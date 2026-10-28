"""rfinal.latency_l2: letterbox GPU (dijalankan di CPU) vs Ultralytics 8.3.240, NMS vs non_max_suppression + scale_boxes."""
import math

import cv2
import numpy as np
import pytest
import torch

from rfinal.latency_l2 import GpuLetterbox, nms_gpu


def _ultra_letterbox(im, S):
    """Jalur val Ultralytics: load_image (rect, INTER_LINEAR, ceil) lalu LetterBox(scaleup=False, center, 114), BGR->RGB."""
    from ultralytics.data.augment import LetterBox
    h0, w0 = im.shape[:2]
    r = S / max(h0, w0)
    if r != 1:
        w, h = min(math.ceil(w0 * r), S), min(math.ceil(h0 * r), S)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    out = LetterBox((S, S), auto=False, scale_fill=False, scaleup=False, center=True)(image=im)
    return torch.from_numpy(np.ascontiguousarray(out[..., ::-1].transpose(2, 0, 1))).float()[None] / 255.0


@pytest.mark.parametrize("shape", [(375, 1242), (370, 1224), (765, 1360), (480, 640), (300, 200), (640, 640)])
def test_letterbox_matches_ultralytics(shape):
    rng = np.random.default_rng(sum(shape))
    im = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    im = cv2.GaussianBlur(im, (5, 5), 0)                   # tekstur alami-ish; noise murni memperbesar selisih pembulatan
    S = 640
    pre = GpuLetterbox(S, torch.device("cpu"), pin=False)
    pre.stage(im)
    dst = torch.zeros(1, 3, S, S)
    pre.run(dst)
    ref = _ultra_letterbox(im, S)
    assert dst.shape == ref.shape
    d = (dst - ref).abs() * 255
    assert d.max().item() <= 1.0 + 1e-4                   # cv2 INTER_LINEAR fixed-point vs bilinear FP32: ≤ 1 level
    assert (d > 0.5).float().mean().item() < 0.2
    m = pre.meta                                           # padding persis sama (baris/kolom 114)
    pad = torch.ones_like(dst, dtype=torch.bool)
    pad[:, :, m["top"]:m["top"] + m["h"], m["left"]:m["left"] + m["w"]] = False
    assert torch.equal(dst[pad], ref[pad])


def test_nms_matches_eval_and_predict():
    """multi_label=True == rfinal.data.nms_one (jalur akurasi suite); False == non_max_suppression predict. Skala balik
    per sumbu == rfinal.data._scale_xyxy."""
    from ultralytics.utils.nms import non_max_suppression
    from rfinal.data import _scale_xyxy, nms_one
    g = torch.Generator().manual_seed(0)
    N, nc = 2000, 3
    xy = torch.rand(2, N, generator=g) * 600 + 20
    wh = torch.rand(2, N, generator=g) * 80 + 4
    sc = torch.rand(nc, N, generator=g) ** 3
    y = torch.cat([xy, wh, sc])[None]                      # (1, 4 + nc, N), xywh piksel letterbox
    h0, w0 = 375, 1242
    pre = GpuLetterbox(640, torch.device("cpu"), pin=False)
    pre.stage(np.zeros((h0, w0, 3), np.uint8))
    m = pre.meta
    rp = ((m["gh"], m["gw"]), (m["left"], m["top"]))
    for conf, ml in ((0.25, False), (0.001, True), (0.25, True)):
        ours = nms_gpu(y, N, m, conf=conf, iou=0.7, max_det=300, multi_label=ml)
        if ml and conf == 0.001:
            ref = nms_one(y.clone(), conf=conf, iou=0.7, max_det=300)
        else:
            ref = non_max_suppression(y.clone(), conf_thres=conf, iou_thres=0.7, max_det=300, nc=nc, multi_label=ml)[0]
        ref = torch.cat([_scale_xyxy(ref[:, :4], rp, (h0, w0)), ref[:, 4:6]], 1)
        assert ours.shape == ref.shape, (conf, ml, ours.shape, ref.shape)
        o = ours[torch.argsort(ours[:, 4] * 10 + ours[:, 5], descending=True)]
        r = ref[torch.argsort(ref[:, 4] * 10 + ref[:, 5], descending=True)]
        assert torch.allclose(o[:, 4], r[:, 4]) and torch.equal(o[:, 5], r[:, 5])
        assert (o[:, :4] - r[:, :4]).abs().max().item() < 1e-3


def test_nms_route_b_cut():
    """Rute B: anchor setelah n diabaikan (padding nol di engine A3)."""
    y = torch.zeros(1, 6, 10)
    y[0, :4, :] = torch.tensor([100.0, 100.0, 20.0, 20.0])[:, None]
    y[0, 4, 3] = 0.9
    y[0, 5, 8] = 0.95                                       # di luar n=5 -> tidak boleh muncul
    pre = GpuLetterbox(640, torch.device("cpu"), pin=False)
    pre.stage(np.zeros((640, 640, 3), np.uint8))
    out = nms_gpu(y, 5, pre.meta, conf=0.25)
    assert out.shape == (1, 6) and out[0, 5].item() == 0 and abs(out[0, 4].item() - 0.9) < 1e-6
