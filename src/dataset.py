"""UCF101 clip dataset (official train/test split 1/2/3), decoded on the fly with OpenCV.

Training: one random 16-frame clip per video (temporal stride 2, i.e. ~12.5 fps from 25 fps), resized
to 128x171, random 112x112 crop and horizontal flip.
Testing: `n_clips` clips uniformly spaced over the video, centre crop; video score = mean of clip scores.
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

UCF_ROOT = os.environ.get("UCF101_ROOT", os.path.expanduser("~/datasets/portfolio/ucf101"))
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)   # Kinetics-400 statistics (torchvision)
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)


def read_split(split="train", fold=1, root=UCF_ROOT):
    sd = os.path.join(root, "ucfTrainTestlist")
    classes = [l.split()[1] for l in open(os.path.join(sd, "classInd.txt")).read().strip().splitlines()]
    idx = {c: i for i, c in enumerate(classes)}
    fn = f"{'trainlist' if split == 'train' else 'testlist'}{fold:02d}.txt"
    items = []
    for line in open(os.path.join(sd, fn)).read().strip().splitlines():
        rel = line.split()[0].strip()
        items.append((os.path.join(root, "UCF-101", rel), idx[rel.split("/")[0]]))
    return items, classes


def load_frames(path, indices, size=(171, 128)):
    """Decode only the requested frame indices (sorted, may repeat)."""
    cap = cv2.VideoCapture(path)
    want = sorted(set(indices))
    frames, cur, k = {}, 0, 0
    while k < len(want):
        if cur < want[k]:
            if not cap.grab():
                break
            cur += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        frames[cur] = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        cur += 1; k += 1
    cap.release()
    if not frames:
        return np.zeros((len(indices), size[1], size[0], 3), np.uint8)
    last = frames[max(frames)]
    return np.stack([frames.get(i, last) for i in indices])


def num_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def to_tensor(clip):
    """(T, H, W, 3) uint8 -> (3, T, H, W) normalised float."""
    x = (clip.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(3, 0, 1, 2).copy())


class UCF101(Dataset):
    def __init__(self, split="train", fold=1, clip_len=16, stride=2, crop=112, n_clips=10):
        self.items, self.classes = read_split(split, fold)
        self.train = split == "train"
        self.clip_len, self.stride, self.crop, self.n_clips = clip_len, stride, crop, n_clips

    def __len__(self):
        return len(self.items)

    def _clip_indices(self, n, start):
        return [min(start + i * self.stride, n - 1) for i in range(self.clip_len)]

    def __getitem__(self, i):
        path, y = self.items[i]
        n = num_frames(path)
        span = self.clip_len * self.stride
        if self.train:
            start = random.randint(0, max(0, n - span))
            clip = load_frames(path, self._clip_indices(n, start))
            h, w = clip.shape[1:3]
            t, l = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            clip = clip[:, t:t + self.crop, l:l + self.crop]
            if random.random() < 0.5:
                clip = clip[:, :, ::-1]
            return to_tensor(clip), y
        starts = np.linspace(0, max(0, n - span), self.n_clips).astype(int)
        idx = [j for s in starts for j in self._clip_indices(n, s)]
        frames = load_frames(path, idx)
        h, w = frames.shape[1:3]
        t, l = (h - self.crop) // 2, (w - self.crop) // 2
        frames = frames[:, t:t + self.crop, l:l + self.crop]
        clips = torch.stack([to_tensor(frames[k * self.clip_len:(k + 1) * self.clip_len])
                             for k in range(self.n_clips)])
        return clips, y
