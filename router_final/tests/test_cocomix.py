"""CocoMix / UnionBoot / FastUnionBoot == COCOeval langsung (data sintetis, CPU, < 30 s)."""
import json

import numpy as np

from rfinal.cocomix import FastUnionBoot, Router, UnionBoot, random_masks
from rfinal.data import coco_eval, jdict


def _synthetic(tmp_path, n_img=40, seed=0):
    from pycocotools.coco import COCO
    rng = np.random.default_rng(seed)
    images, anns, dA, dB, aid = [], [], [], [], 1
    for i in range(n_img):
        images.append({"id": i, "file_name": f"{i}.jpg", "height": 480, "width": 640})
        gts = []
        for _ in range(rng.integers(0, 6)):
            w, h = float(rng.choice([12, 20, 40, 90])), float(rng.choice([10, 24, 50, 120]))
            x, y = float(rng.uniform(0, 640 - w)), float(rng.uniform(0, 480 - h))
            c = int(rng.integers(1, 3))
            anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            gts.append((x, y, w, h, c)); aid += 1
        for store, jit, miss in ((dA, 2.0, 0.1), (dB, 6.0, 0.35)):        # A lebih akurat dari B
            d = []
            for x, y, w, h, c in gts:
                if rng.random() > miss:
                    d.append([x + rng.normal(0, jit), y + rng.normal(0, jit), w, h, rng.uniform(0.3, 1), c])
            for _ in range(rng.integers(0, 4)):                              # FP
                d.append([rng.uniform(0, 600), rng.uniform(0, 440), 30, 30, rng.uniform(0, 0.6), int(rng.integers(1, 3))])
            store.append(np.asarray(d, np.float32).reshape(-1, 6))
    gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    p = tmp_path / "gt.json"; p.write_text(json.dumps(gt))
    return COCO(str(p)), dA, dB, list(range(n_img))


def test_router_union_fast_equal_cocoeval(tmp_path):
    gt, dA, dB, ids = _synthetic(tmp_path)
    _, EA = coco_eval(gt, jdict(dA, ids)); _, EB = coco_eval(gt, jdict(dB, ids))
    R = Router(EA, EB, ids)
    U = UnionBoot(R.bA, R.bB); F = FastUnionBoot(U)
    n = len(ids)
    for m in random_masks(n, 0.4, 3, 1) + [np.ones(n, bool), np.zeros(n, bool)]:
        st, _ = coco_eval(gt, jdict([a if u else b for a, b, u in zip(dA, dB, m)], ids))
        assert st["mAP_small"] > 0.01 and st["mAP50"] > 0.1, st          # data tidak trivial
        r = R.metrics(m); su = R.sorted_mask(m)
        u, f = U.metrics(su, np.arange(n)), F.metrics(su, np.arange(n))
        for k in ("mAP50-95", "mAP50", "mAP_small"):
            assert abs(r[k] - st[k]) < 1e-12 and abs(u[k] - st[k]) < 1e-12 and abs(f[k] - st[k]) < 1e-12, (k, r[k], u[k], f[k], st[k])
    rng = np.random.default_rng(3)
    for _ in range(5):                                                       # resample berulang: Union == Fast
        idx = rng.integers(0, n, n); su = R.sorted_mask(random_masks(n, 0.5, 1, 9)[0])
        u, f = U.metrics(su, idx), F.metrics(su, idx)
        assert all(abs(u[k] - f[k]) < 1e-12 for k in u), (u, f)
