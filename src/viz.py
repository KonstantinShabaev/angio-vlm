"""Overlay ground truth (ARCADE masks / boxes) and VLM claims (points) for a few images."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gt import build_stenosis_gt, build_syntax_gt, syntax_artery_masks

COLORS = {"LM": "#e6194b", "LAD": "#3cb44b", "LCx": "#4363d8", "RCA": "#f58231"}


def show_syntax(ax, root, split, g, ex):
    img = np.array(Image.open(g["file"]).convert("L"))
    ax.imshow(img, cmap="gray")
    for a, m in syntax_artery_masks(root, split, g["image_id"]).items():
        rgba = np.zeros((*m.shape, 4))
        rgba[m] = matplotlib.colors.to_rgba(COLORS[a], 0.35)
        ax.imshow(rgba)
    for a, pt in ex["pred"].items():
        if pt:
            ax.scatter([pt[0]], [pt[1]], s=90, marker="X", c=COLORS.get(a, "w"), edgecolors="white", linewidths=1.2)
            ax.annotate(a, pt, xytext=(6, -6), textcoords="offset points", color="white", fontsize=8, weight="bold")
    ax.set_title(f"#{g['image_id']}  GT: {', '.join(g['arteries'])}\nVLM: {', '.join(ex['pred']) or '-'} | side={ex['side']}", fontsize=8)
    ax.axis("off")


def show_stenosis(ax, g, ex):
    img = np.array(Image.open(g["file"]).convert("L"))
    ax.imshow(img, cmap="gray")
    for x, y, w, h in g["stenosis_boxes"]:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#3cb44b", lw=1.8))
    for pt in ex["pred_points"]:
        if pt:
            ax.scatter([pt[0]], [pt[1]], s=90, marker="X", c="#e6194b", edgecolors="white", linewidths=1.2)
    ax.set_title(f"#{g['image_id']}  GT stenoses: {len(g['stenosis_boxes'])} | VLM points: {len(ex['pred_points'])}", fontsize=8)
    ax.axis("off")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/data/arcade")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True, help="e.g. Qwen2.5-VL-3B-Instruct_rag_syntax_test_n50")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    ex = json.load(open(f"results/examples_{args.tag}.json"))[: args.k]
    syntax = "_syntax_" in args.tag
    gts = {g["image_id"]: g for g in (build_syntax_gt if syntax else build_stenosis_gt)(args.root, args.split)}
    cols = 3
    rows = (len(ex) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.95))
    for ax, e in zip(np.ravel(axes), ex):
        if syntax:
            show_syntax(ax, args.root, args.split, gts[e["image_id"]], e)
        else:
            show_stenosis(ax, gts[e["image_id"]], e)
    for ax in np.ravel(axes)[len(ex):]:
        ax.axis("off")
    if syntax:
        handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=a) for a, c in COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                   title="маска = разметка ARCADE, X = точка, указанная VLM", title_fontsize=8)
    else:
        fig.legend(handles=[plt.Line2D([], [], color="#3cb44b", label="стеноз по разметке ARCADE"),
                            plt.Line2D([], [], marker="X", ls="", color="#e6194b", label="точка стеноза от VLM")],
                   loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.out) / f"{args.tag}.png", dpi=130)
    print("saved", Path(args.out) / f"{args.tag}.png")
