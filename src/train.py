"""Fine-tune a video CNN on UCF101 split 1.

    python src/train.py --arch r2plus1d_18                 # Kinetics-400 pretrained (main model)
    python src/train.py --arch r3d_18                      # 3D-conv baseline, pretrained
    python src/train.py --arch r2plus1d_18 --scratch       # same network, random init
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UCF101
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="r2plus1d_18")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hours", type=float, default=None, help="time budget; overrides --epochs")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0075)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    tag = f"{args.arch}_{'scratch' if args.scratch else 'k400'}"
    ckpt_dir = os.path.join(ROOT, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    ds = UCF101("train", args.fold)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    model = build(args.arch, len(ds.classes), pretrained=not args.scratch).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    if args.hours:
        # time budget: measure the real throughput (data loading included), then pick the epoch count
        init = copy.deepcopy(model.state_dict())
        probe = torch.optim.SGD(model.parameters(), lr=0.0)
        it_dl = iter(dl)
        for k in range(50):
            if k == 10:
                torch.cuda.synchronize(); t_start = time.time()
            x, y = next(it_dl)
            with torch.autocast("cuda", dtype=torch.float16):
                l = crit(model(x.to(dev)), y.to(dev))
            probe.zero_grad(); l.backward(); probe.step()
        torch.cuda.synchronize()
        sec_it = (time.time() - t_start) / 40
        model.load_state_dict(init); del probe, it_dl
        args.epochs = max(1, int(args.hours * 3600 / (sec_it * len(dl))))
        print(f"[{tag}] {sec_it:.3f} s/it -> {args.epochs} epochs in {args.hours} h", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = args.epochs * len(dl)
    warm = len(dl) // 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=warm / steps, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {"loss": [], "acc": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for it, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x); loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
            if it % 100 == 0:
                print(f"[{tag}] ep {ep + 1} it {it}/{len(dl)} loss {tl / n:.3f} acc {tc / n:.3f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        hist["loss"].append(tl / n); hist["acc"].append(tc / n)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}.pt"))
        print(f"[{tag}] epoch {ep + 1} done: train loss {tl / n:.3f} acc {tc / n:.3f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"history_{tag}.json"), "w") as f:
        json.dump(hist, f)


if __name__ == "__main__":
    main()
