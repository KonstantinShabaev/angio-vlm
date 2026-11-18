"""Evaluate VLM predictions against ARCADE ground truth.

Syntax subset  -> laterality, artery-level P/R/F1, grounding of artery claims on segment masks.
Stenosis subset -> pointing precision/recall of stenosis claims on stenosis boxes.
Each metric is compared with a trivial baseline (majority / random point).
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from gt import ARTERIES, build_stenosis_gt, build_syntax_gt, syntax_artery_masks

LEFT = {"LM", "LAD", "LCx"}
TOL_PX = 7      # tolerance around a vessel mask, pixels (512x512 image)
BOX_PAD = 10    # tolerance around a stenosis box, pixels


def load_preds(path):
    return {r["image_id"]: r for r in map(json.loads, open(path))}


def norm_artery(a):
    a = str(a).strip().upper()
    return {"LCX": "LCx", "CX": "LCx", "CIRC": "LCx", "LMCA": "LM", "LEFT MAIN": "LM"}.get(a, a if a in {"LM", "LAD", "RCA"} else None)


def point_of(item, scale):
    try:
        x, y = item["point"][:2]
        return float(x) * scale[0], float(y) * scale[1]
    except Exception:
        return None


def on_mask(mask, p):
    if p is None:
        return False
    x, y = int(round(p[0])), int(round(p[1]))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def eval_syntax(pred_path, root, split):
    preds = load_preds(pred_path)
    gts = {g["image_id"]: g for g in build_syntax_gt(root, split)}
    ids = [i for i in preds if i in gts]
    st = Counter()
    vessel_area = []
    examples = []
    for iid in ids:
        g, r = gts[iid], preds[iid]
        p = r["parsed"]
        st["n"] += 1
        if p is None:
            st["unparsed"] += 1
            p = {}
        side = str(p.get("side", "unknown")).lower()
        st["side_unknown"] += side not in {"left", "right"}
        st["side_correct"] += side == g["side"]
        masks = {a: binary_dilation(m, iterations=TOL_PX) for a, m in syntax_artery_masks(root, split, iid).items()}
        any_vessel = np.any(np.stack(list(masks.values())), axis=0)
        vessel_area.append(any_vessel.mean())
        pred_arts = {}
        for it in p.get("arteries", []) or []:
            a = norm_artery(it.get("artery", "")) if isinstance(it, dict) else None
            if a and a not in pred_arts:
                pred_arts[a] = point_of(it, r["scale"])
        gset = set(g["arteries"])
        for a in ARTERIES:
            st["tp"] += a in pred_arts and a in gset
            st["fp"] += a in pred_arts and a not in gset
            st["fn"] += a not in pred_arts and a in gset
        # claims about the other coronary system (left vs right) - anatomical contradiction
        gt_left = g["side"] == "left"
        st["claims"] += len(pred_arts)
        st["cross_system"] += sum((a in LEFT) != gt_left for a in pred_arts)
        for a, pt in pred_arts.items():
            st["pts_on_vessel"] += on_mask(any_vessel, pt)
            if a in gset:
                st["claims_true"] += 1
                st["pts_on_own_artery"] += on_mask(masks[a], pt)
        supported = sum(a in gset and on_mask(masks[a], pt) for a, pt in pred_arts.items())
        st["supported"] += supported
        examples.append({"image_id": iid, "gt": g["arteries"], "pred": {a: pt for a, pt in pred_arts.items()}, "side": side})
    n = st["n"]
    P, R, F = prf(st["tp"], st["fp"], st["fn"])
    # baselines on the same images
    maj_side = Counter(gts[i]["side"] for i in ids).most_common(1)[0]
    rng = random.Random(0)
    bp = Counter()
    for iid in ids:  # baseline: always predict LM + LAD (most frequent arteries)
        gset = set(gts[iid]["arteries"])
        for a in ARTERIES:
            pa = a in {"LM", "LAD"}
            bp["tp"] += pa and a in gset; bp["fp"] += pa and a not in gset; bp["fn"] += (not pa) and a in gset
    bP, bR, bF = prf(bp["tp"], bp["fp"], bp["fn"])
    return {
        "images": n,
        "parse_rate": round(1 - st["unparsed"] / n, 3),
        "side_accuracy": round(st["side_correct"] / n, 3),
        "side_unknown_rate": round(st["side_unknown"] / n, 3),
        "side_majority_baseline": round(maj_side[1] / n, 3),
        "artery_P_R_F1": [P, R, F],
        "artery_baseline_LM_LAD_P_R_F1": [bP, bR, bF],
        "artery_claims_per_image": round(st["claims"] / n, 2),
        "cross_system_claim_rate": round(st["cross_system"] / max(st["claims"], 1), 3),
        "point_on_any_vessel_rate": round(st["pts_on_vessel"] / max(st["claims"], 1), 3),
        "random_point_on_vessel_baseline": round(float(np.mean(vessel_area)), 3),
        "point_on_correct_artery_rate": round(st["pts_on_own_artery"] / max(st["claims_true"], 1), 3),
        "supported_claim_rate": round(st["supported"] / max(st["claims"], 1), 3),
    }, examples


def eval_stenosis(pred_path, root, split):
    preds = load_preds(pred_path)
    gts = {g["image_id"]: g for g in build_stenosis_gt(root, split)}
    ids = [i for i in preds if i in gts]
    st = Counter()
    area = []
    confs_hit, confs_miss = [], []
    sev = Counter()
    examples = []
    for iid in ids:
        g, r = gts[iid], preds[iid]
        p = r["parsed"] or {}
        st["n"] += 1
        st["unparsed"] += r["parsed"] is None
        boxes = [(x - BOX_PAD, y - BOX_PAD, x + w + BOX_PAD, y + h + BOX_PAD) for x, y, w, h in g["stenosis_boxes"]]
        m = np.zeros((g["height"], g["width"]), bool)
        for x0, y0, x1, y1 in boxes:
            m[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
        area.append(m.mean())
        art_pts = {tuple(point_of(it, r["scale"]) or ()) for it in (p.get("arteries") or []) if isinstance(it, dict)}
        sts = [it for it in (p.get("stenoses") or []) if isinstance(it, dict)]
        st["gt"] += len(boxes)
        st["claims"] += len(sts)
        st["no_claim_images"] += len(sts) == 0
        hit_boxes = set()
        pts = []
        for it in sts:
            pt = point_of(it, r["scale"])
            pts.append(pt)
            sev[str(it.get("severity", "?"))] += 1
            st["copied_artery_point"] += pt is not None and tuple(pt) in art_pts
            hits = [j for j, (x0, y0, x1, y1) in enumerate(boxes) if pt and x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1]
            st["hit_claims"] += bool(hits)
            hit_boxes.update(hits)
            try:
                (confs_hit if hits else confs_miss).append(float(it.get("confidence")))
            except (TypeError, ValueError):
                pass
        st["gt_found"] += len(hit_boxes)
        examples.append({"image_id": iid, "pred_points": pts, "gt_boxes": g["stenosis_boxes"]})
    n = st["n"]
    return {
        "images": n,
        "parse_rate": round(1 - st["unparsed"] / n, 3),
        "gt_stenoses": st["gt"],
        "claims_per_image": round(st["claims"] / n, 2),
        "images_without_claims": round(st["no_claim_images"] / n, 3),
        "point_precision": round(st["hit_claims"] / max(st["claims"], 1), 3),
        "random_point_precision_baseline": round(float(np.mean(area)), 3),
        "stenosis_recall": round(st["gt_found"] / max(st["gt"], 1), 3),
        "stenosis_point_copied_from_artery": round(st["copied_artery_point"] / max(st["claims"], 1), 3),
        "severity_counts": dict(sev),
        "mean_conf_hit": round(float(np.mean(confs_hit)), 3) if confs_hit else None,
        "mean_conf_miss": round(float(np.mean(confs_miss)), 3) if confs_miss else None,
    }, examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/data/arcade")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = {}
    for f in sorted(Path(args.results).glob("pred_*.jsonl")):
        task = "syntax" if "_syntax_" in f.name else "stenosis"
        fn = eval_syntax if task == "syntax" else eval_stenosis
        metrics, ex = fn(f, args.root, args.split)
        out[f.stem.replace("pred_", "")] = metrics
        json.dump(ex, open(Path(args.results) / f"examples_{f.stem[5:]}.json", "w"))
    json.dump(out, open(Path(args.results) / "metrics.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
