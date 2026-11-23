"""Fast CPU tests: 3D-CNN heads, clip index sampling, frame decoding, clip -> video score aggregation."""
import cv2
import numpy as np
import pytest
import torch

import evaluate
from dataset import MEAN, STD, UCF101, load_frames, to_tensor
from model import build

torch.manual_seed(0)


@pytest.mark.parametrize("arch", ["r3d_18", "r2plus1d_18"])
def test_video_model_head_shape(arch):
    m = build(arch, num_classes=101, pretrained=False).eval()      # no Kinetics download
    with torch.no_grad():
        y = m(torch.randn(2, 3, 8, 64, 64))
    assert y.shape == (2, 101)


def _ds(**kw):
    ds = object.__new__(UCF101)                                    # skip the split files
    ds.clip_len, ds.stride, ds.crop, ds.n_clips, ds.train = 16, 2, 112, 10, False
    ds.__dict__.update(kw)
    return ds


def test_clip_indices_stride_and_clamp():
    ds = _ds()
    assert ds._clip_indices(100, 10) == list(range(10, 42, 2))
    assert ds._clip_indices(20, 0) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 19, 19, 19, 19, 19]


@pytest.fixture
def video(tmp_path):
    """60-frame video whose frame k has brightness 4k, so decoded frames reveal their index."""
    path = str(tmp_path / "v_Test_g01_c01.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (171, 128))
    if not w.isOpened():
        pytest.skip("OpenCV build without an AVI/MJPG writer")
    for k in range(60):
        w.write(np.full((128, 171, 3), 4 * k, np.uint8))
    w.release()
    return path


def test_load_frames_repeats_and_out_of_range(video):
    fr = load_frames(video, [5, 5, 30, 200])
    assert fr.shape == (4, 128, 171, 3)
    np.testing.assert_allclose(fr[:3].reshape(3, -1).mean(1), [20, 20, 120], atol=3)
    np.testing.assert_array_equal(fr[3], fr[2])      # past the end -> last decoded frame is repeated


def test_test_time_sampling_uniform_clips(video):
    clips, y = _ds(items=[(video, 7)])[0]
    assert y == 7 and clips.shape == (10, 3, 16, 112, 112)
    # undo normalisation and read back which frame each clip starts on
    first = clips[:, 0, 0].mean((1, 2)).numpy() * STD[0] + MEAN[0]
    np.testing.assert_allclose(first * 255 / 4, np.linspace(0, 60 - 32, 10).astype(int), atol=0.75)


def test_to_tensor_layout_and_normalisation():
    clip = np.zeros((4, 8, 8, 3), np.uint8)
    x = to_tensor(clip)
    assert x.shape == (3, 4, 8, 8)
    torch.testing.assert_close(x[:, 0, 0, 0], torch.from_numpy(-MEAN / STD))


def test_clip_to_video_aggregation_and_topk(tmp_path, monkeypatch):
    n_cls = 5
    (tmp_path / "checkpoints").mkdir()
    torch.save(build("r3d_18", n_cls, pretrained=False).state_dict(), tmp_path / "checkpoints" / "r3d_18_k400.pt")
    monkeypatch.setattr(evaluate, "ROOT", str(tmp_path))
    monkeypatch.setattr(evaluate, "DataLoader", lambda ds, **kw: torch.utils.data.DataLoader(ds, batch_size=1))

    class Fake(torch.utils.data.Dataset):
        classes = [str(i) for i in range(n_cls)]
        clips = torch.randn(2, 3, 3, 8, 32, 32)                    # 2 videos x 3 clips

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return self.clips[i], i

    clip_p, vid_p, ys = evaluate.predict("r3d_18_k400", Fake(), torch.device("cpu"))
    m = build("r3d_18", n_cls, pretrained=False)
    m.load_state_dict(torch.load(tmp_path / "checkpoints" / "r3d_18_k400.pt"))
    with torch.no_grad():
        ref = m.eval()(Fake.clips[1]).softmax(1)
    torch.testing.assert_close(vid_p[1], ref.mean(0), atol=1e-5, rtol=1e-4)   # video = mean of clip scores
    torch.testing.assert_close(clip_p[1], ref[1], atol=1e-5, rtol=1e-4)       # clip = the central clip
    assert ys.tolist() == [0, 1]

    p = torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]])
    y = torch.tensor([1, 2, 0])
    assert evaluate.topk(p, y, 1) == pytest.approx(100 / 3)
    assert evaluate.topk(p, y, 2) == pytest.approx(200 / 3)
    assert evaluate.topk(p, y, 3) == pytest.approx(100.0)
