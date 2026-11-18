"""Zero-shot structured description of ARCADE angiograms with an open VLM, with or without RAG.

Usage:
  python src/run_vlm.py --cond base --task syntax --n 30
  python src/run_vlm.py --cond rag  --task stenosis --n 30
Works on CPU (slow) or GPU (Colab T4: add --device cuda).
"""
import argparse, json, random, re, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from gt import build_syntax_gt, build_stenosis_gt
from rag import Retriever

SYSTEM = "You are an experienced interventional cardiologist reading a single coronary angiography frame."

SCHEMA = """Describe this coronary angiogram as JSON with exactly these keys:
"side": "left" or "right" or "unknown" - which coronary system is injected,
"arteries": a list of {{"artery": one of "LM", "LAD", "LCx", "RCA", "point": [x, y]}} for every artery that is clearly visible; point is a pixel on that artery,
"stenoses": a list of {{"artery": one of "LM", "LAD", "LCx", "RCA", "point": [x, y], "severity": "mild" or "moderate" or "severe" or "occlusion", "confidence": a number from 0 to 1}}; point is the center of the narrowing,
"not_assessable": a list of short notes about what cannot be judged from this image.
The image is {w}x{h} pixels; coordinates are pixels in this image. Use only what is visible; if unsure, leave the list empty or say so in "not_assessable". Output only the JSON."""

RAG_QUERY = ("left right coronary injection laterality LM LAD LCx RCA segments course "
             "stenosis narrowing location projection report not visible")


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cond", choices=["base", "rag"], default="base")
    ap.add_argument("--task", choices=["syntax", "stenosis"], default="syntax")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=6, help="retrieved chunks for RAG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/home/user/data/arcade")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="T4 GPU: fp16")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    gts = build_syntax_gt(args.root, args.split) if args.task == "syntax" else build_stenosis_gt(args.root, args.split)
    random.Random(args.seed).shuffle(gts)
    gts = gts[: args.n]

    context = ""
    if args.cond == "rag":
        chunks = Retriever(str(Path(__file__).resolve().parent.parent / "kb" / "knowledge.jsonl")).search(RAG_QUERY, k=args.k)
        context = "Reference knowledge:\n" + "\n".join(f"- {c['text']}" for c in chunks) + "\n\n"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.load_4bit:  # e.g. 7B model on a free Colab T4
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForImageTextToText.from_pretrained(args.model, quantization_config=q, device_map="auto").eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 28 * 28, max_pixels=512 * 512)

    tag = f"{Path(args.model).name}_{args.cond}_{args.task}_{args.split}_n{args.n}"
    out_path = Path(args.out) / f"pred_{tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["image_id"] for l in open(out_path)}
    fout = open(out_path, "a")

    for i, g in enumerate(gts):
        if g["image_id"] in done:
            continue
        img = Image.open(g["file"]).convert("RGB")
        # size the model actually sees (Qwen resizes to multiples of 28)
        probe = processor.image_processor(images=[img], return_tensors="pt")
        gh, gw = probe["image_grid_thw"][0][1:].tolist()
        rh, rw = gh * 14, gw * 14
        prompt = context + SCHEMA.format(w=rw, h=rh)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(args.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=350, do_sample=False)
        raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        rec = {"image_id": g["image_id"], "raw": raw, "parsed": parse_json(raw),
               "scale": [g["width"] / rw, g["height"] / rh], "sec": round(time.time() - t0, 1)}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{i+1}/{len(gts)}] id={g['image_id']} {rec['sec']}s parsed={rec['parsed'] is not None}", flush=True)


if __name__ == "__main__":
    main()
