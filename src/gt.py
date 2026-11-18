"""Build structured ground truth from ARCADE COCO annotations.

ARCADE has two independent subsets:
  syntax/   - 25 coronary segments (SYNTAX numbering), no stenosis labels
  stenosis/ - only stenosis polygons, no segment labels
So artery-level facts are evaluated on syntax/, stenosis facts on stenosis/.
"""
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

# SYNTAX segment -> artery
SEGMENT_TO_ARTERY = {
    "1": "RCA", "2": "RCA", "3": "RCA", "4": "RCA",
    "16": "RCA", "16a": "RCA", "16b": "RCA", "16c": "RCA",
    "5": "LM",
    "6": "LAD", "7": "LAD", "8": "LAD", "9": "LAD", "9a": "LAD", "10": "LAD", "10a": "LAD",
    "11": "LCx", "12": "LCx", "12a": "LCx", "12b": "LCx", "13": "LCx",
    "14": "LCx", "14a": "LCx", "14b": "LCx", "15": "LCx",
}
ARTERIES = ["RCA", "LM", "LAD", "LCx"]


def load_coco(root, task, split):
    path = Path(root) / task / split / "annotations" / f"{split}.json"
    d = json.load(open(path))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return imgs, anns, cats


def ann_mask(a, h, w):
    rles = mask_utils.frPyObjects(a["segmentation"], h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def build_syntax_gt(root, split):
    imgs, anns, cats = load_coco(root, "syntax", split)
    out = []
    for iid, im in sorted(imgs.items()):
        h, w = im["height"], im["width"]
        segs = sorted({cats[a["category_id"]] for a in anns.get(iid, [])},
                      key=lambda s: (int("".join(c for c in s if c.isdigit())), s))
        arteries = sorted({SEGMENT_TO_ARTERY[s] for s in segs}, key=ARTERIES.index)
        side = "right" if "RCA" in arteries else "left"
        out.append({
            "image_id": iid, "file": str(Path(root) / "syntax" / split / "images" / im["file_name"]),
            "width": w, "height": h, "side": side, "arteries": arteries, "segments": segs,
        })
    return out


def syntax_artery_masks(root, split, image_id):
    imgs, anns, cats = load_coco(root, "syntax", split)
    im = imgs[image_id]
    masks = {}
    for a in anns.get(image_id, []):
        art = SEGMENT_TO_ARTERY[cats[a["category_id"]]]
        m = ann_mask(a, im["height"], im["width"])
        masks[art] = masks.get(art, np.zeros_like(m)) | m
    return masks


def build_stenosis_gt(root, split):
    imgs, anns, cats = load_coco(root, "stenosis", split)
    out = []
    for iid, im in sorted(imgs.items()):
        boxes = [a["bbox"] for a in anns.get(iid, [])]  # x, y, w, h
        out.append({
            "image_id": iid, "file": str(Path(root) / "stenosis" / split / "images" / im["file_name"]),
            "width": im["width"], "height": im["height"], "stenosis_boxes": boxes,
        })
    return out


if __name__ == "__main__":
    import argparse, collections
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/data/arcade")
    ap.add_argument("--out", default="/home/user/angio-vlm/results/gt_stats.json")
    args = ap.parse_args()
    stats = {}
    for split in ["train", "val", "test"]:
        sg = build_syntax_gt(args.root, split)
        st = build_stenosis_gt(args.root, split)
        seg_c = collections.Counter(s for g in sg for s in g["segments"])
        stats[split] = {
            "syntax_images": len(sg),
            "side": dict(collections.Counter(g["side"] for g in sg)),
            "artery_freq": dict(collections.Counter(a for g in sg for a in g["arteries"])),
            "segments_per_image_mean": round(float(np.mean([len(g["segments"]) for g in sg])), 2),
            "segment_freq": dict(seg_c.most_common()),
            "stenosis_images": len(st),
            "stenoses_per_image_mean": round(float(np.mean([len(g["stenosis_boxes"]) for g in st])), 2),
            "stenosis_box_area_frac_mean": round(float(np.mean(
                [sum(b[2] * b[3] for b in g["stenosis_boxes"]) / (g["width"] * g["height"]) for g in st])), 4),
        }
    json.dump(stats, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats["test"], indent=1))
