"""Evaluate fine-tuned models on the UCF101 split-1 test set (3,783 videos).

Reports clip accuracy (1 centre clip) and video accuracy (mean of 10 uniformly spaced clips),
top-1 and top-5; writes the confusion analysis and per-class accuracy for the best model.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import UCF101
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


@torch.no_grad()
def predict(tag, ds, dev):
    arch, init = tag.rsplit("_", 1)
    model = build(arch, len(ds.classes), pretrained=False).to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{tag}.pt"), map_location=dev))
    model.eval()
    dl = DataLoader(ds, batch_size=1, num_workers=12, pin_memory=True)
    clip_p, vid_p, ys = [], [], []
    for x, y in dl:
        x = x[0].to(dev)                                   # (n_clips, 3, T, H, W)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).float().softmax(1)
        clip_p.append(p[len(p) // 2].cpu()); vid_p.append(p.mean(0).cpu()); ys.append(y.item())
    return torch.stack(clip_p), torch.stack(vid_p), torch.tensor(ys)


def topk(p, y, k):
    return (p.topk(k, 1).indices == y[:, None]).any(1).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["r2plus1d_18_k400", "r3d_18_k400", "r2plus1d_18_scratch"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    ds = UCF101("test", 1)
    metrics, best = {}, None
    for tag in args.tags:
        if not os.path.exists(os.path.join(ROOT, "checkpoints", f"{tag}.pt")):
            continue
        cp, vp, y = predict(tag, ds, dev)
        metrics[tag] = {"clip_top1": topk(cp, y, 1), "video_top1": topk(vp, y, 1), "video_top5": topk(vp, y, 5)}
        print(tag, {k: round(v, 2) for k, v in metrics[tag].items()}, flush=True)
        if best is None or metrics[tag]["video_top1"] > metrics[best[0]]["video_top1"]:
            best = (tag, vp, y)
    os.makedirs(RES, exist_ok=True)
    json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

    tag, vp, y = best
    pred = vp.argmax(1).numpy(); y = y.numpy(); C = len(ds.classes)
    cm = np.zeros((C, C), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    per_class = cm.diagonal() / cm.sum(1)
    order = np.argsort(per_class)
    worst = [(ds.classes[i], float(per_class[i])) for i in order[:12]]
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape))[0]
    confused = [(ds.classes[a], ds.classes[b], int(off[a, b])) for a, b in pairs]
    json.dump({"model": tag, "worst_classes": worst, "most_confused": confused},
              open(os.path.join(RES, "error_analysis.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([w[0] for w in worst][::-1], [100 * w[1] for w in worst][::-1], color="#c0504d")
    ax.set_xlabel("video accuracy (%)"); ax.set_title(f"12 hardest classes ({tag})"); ax.set_xlim(0, 100)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "hardest_classes.png"), dpi=120); plt.close()

    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"UCF101 split-1 confusion matrix ({tag})"); ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=110); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for t in args.tags:
        p = os.path.join(RES, f"history_{t}.json")
        if os.path.exists(p):
            h = json.load(open(p)); ep = range(1, len(h["loss"]) + 1)
            ax[0].plot(ep, h["loss"], marker="o", label=t); ax[1].plot(ep, [100 * a for a in h["acc"]], marker="o", label=t)
    ax[0].set(title="train loss", xlabel="epoch"); ax[1].set(title="train clip accuracy (%)", xlabel="epoch")
    ax[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(RES, "training_curves.png"), dpi=110)


if __name__ == "__main__":
    main()
