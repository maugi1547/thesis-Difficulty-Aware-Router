"""NMS cepat (rfinal.data.nms_one, fast_nms_k) harus identik dengan NMS Ultralytics penuh bila skor kontinu (tanpa tie)."""
import numpy as np
import torch

from rfinal import data as D
from rfinal.data import nms_one, to_dets


def _scene(seed, n=1500, nc=2, clusters=25, img=640):
    """Keluaran Detect sintetis (1, 4+nc, n): kotak xywh berkerumun (banyak tumpang tindih), skor kontinu."""
    g = torch.Generator().manual_seed(seed)
    ctr = torch.rand(clusters, 2, generator=g) * img
    k = torch.randint(0, clusters, (n,), generator=g)
    xy = ctr[k] + torch.randn(n, 2, generator=g) * 6
    wh = 12 + torch.rand(n, 2, generator=g) * 30
    sc = torch.rand(nc, n, generator=g) ** 4                    # mayoritas skor rendah, seperti conf 0.001
    return torch.cat([xy.T, wh.T, sc], 0)[None].float()


def test_fast_nms_identical_and_both_paths_used():
    for p in D.NMS_STATS:
        D.NMS_STATS[p] = 0
    rp, osh = ((1.0, 1.0), (0.0, 0.0)), (640, 640)
    for seed in range(4):
        y = _scene(seed)
        for md in (50, 300):
            ref = nms_one(y, 0.001, 0.7, md)
            ref_d = to_dets(y, rp, osh, 0.001, 0.7, md)
            for k in (100, 500, 1500, 20000):
                got = nms_one(y, 0.001, 0.7, md, fast_nms_k=k)
                assert torch.equal(ref, got), (seed, md, k)
                assert np.array_equal(ref_d, to_dets(y, rp, osh, 0.001, 0.7, md, fast_nms_k=k)), (seed, md, k)
    s = D.NMS_STATS
    assert s["cepat"] > 0 and s["fallback"] > 0 and s["penuh_langsung"] > 0, s


def test_prefix_property_used_by_suite():
    """Suite memakai d[:max_det_gate] dari NMS ber-max_det lebih besar sebagai pengganti NMS ber-max_det gate."""
    rp, osh = ((1.0, 1.0), (0.0, 0.0)), (640, 640)
    for seed in range(2):
        y = _scene(seed + 10)
        d500 = to_dets(y, rp, osh, 0.001, 0.7, 500)
        d300 = to_dets(y, rp, osh, 0.001, 0.7, 300)
        assert np.array_equal(d500[:300], d300), seed
