"""
Gate post-hoc (router_final) — dilatih SETELAH detektor dibekukan; tidak menyentuh bobot detektor.

    fitur  : f = [GAP‖GMP](p3_neck) ‖ [GAP‖GMP](p2_backbone) ‖ [GAP‖GMP](p4_neck) ‖ [GAP‖GMP](p5_sppf)   (960)
    target : y = 1[Δ soft_small > 0],  Δ = soft_small(Branch A) − soft_small(Branch B) per gambar TRAIN
             soft_small = Σ skor deteksi TP (IoU 0.5) pada GT kecil COCO (< 32² px asli), dari evalImgs COCOeval
    model  : StandardScaler + LogisticRegression(C = 0.001, lbfgs, max_iter 5000), fit float64
    deploy : dilipat menjadi satu Gemm: skor = f · w + b,  w = coef / scale,  b = intercept − Σ coef · mean / scale
             (keputusan = skor > threshold B2, lihat rfinal/budget.py)

Diporting dari eksperimen/tahap4_final.py (cache_split, fit_gate) + tahap3a.py (fit_logreg, folded_weights).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import build_coco_gt, coco_eval, image_id_of, jdict, per_image_soft_small, to_dets
from .model import BranchA, BranchB, pool4


@torch.no_grad()
def cache_split(model, loader, device, nms, dets=True, log_every=1000, log=print):
    """Satu forward per gambar (batch 1, urutan loader): fitur gate (960) + deteksi pasca-NMS Branch A & B."""
    a, b_ = BranchA(model).eval(), BranchB(model).eval()
    out = {k: [] for k in ("file", "image_id", "feat", "detA", "detB")}
    t0 = time.perf_counter()
    for i, b in enumerate(loader):
        assert b["img"].shape[0] == 1, "cache_split butuh batch 1"
        im = b["img"].to(device).float() / 255.0
        p3, p2, p4, p5 = model.trunk(im)
        out["feat"].append(pool4(p3, p2, p4, p5)[0].cpu().numpy())
        if dets:
            rp, osh = b["ratio_pad"][0], b["ori_shape"][0]
            out["detA"].append(to_dets(a(p3, p2, p4, p5), rp, osh, **nms))
            out["detB"].append(to_dets(b_(p3, p4, p5), rp, osh, **nms))
        out["file"].append(Path(b["im_file"][0]).name)
        out["image_id"].append(image_id_of(b["im_file"][0]))
        if log_every and (i + 1) % log_every == 0:
            log(f"  {i + 1} gambar | {time.perf_counter() - t0:.0f} s")
    out["feat"] = np.stack(out["feat"])
    return out


def soft_small_delta(cache, dataset, names, gt_path, max_dets=(1, 10, 100)):
    """Δ soft_small (A − B) per gambar, urutan cache. Juga statistik A/B dan jumlah GT kecil."""
    gt = build_coco_gt(dataset, names, gt_path)
    ids = cache["image_id"]
    stA, EA = coco_eval(gt, jdict(cache["detA"], ids), max_dets)
    stB, EB = coco_eval(gt, jdict(cache["detB"], ids), max_dets)
    sA, n_small = per_image_soft_small(EA, ids)
    sB, n_small_b = per_image_soft_small(EB, ids)
    assert np.array_equal(n_small, n_small_b)
    return dict(delta=sA - sB, n_gt_small=n_small, stats_A=stA, stats_B=stB)


def fit_gate(feat, delta, C=0.001, max_iter=5000):
    """Returns dict(w float64 (D,), b float, info). Lipatan scaler + logreg diperiksa terhadap sklearn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(feat, np.float64)
    y = (np.asarray(delta) > 0).astype(int)
    assert 0 < y.sum() < len(y), "target gate hanya satu kelas"
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter)).fit(X, y)
    sc, lr = pipe[0], pipe[1]
    scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    w = lr.coef_[0] / scale
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / scale))
    ref = pipe.decision_function(X[:256])
    assert np.allclose(X[:256] @ w + b, ref, atol=1e-6 * max(1.0, np.abs(ref).max())), "lipatan scaler + logreg salah"
    import sklearn
    info = dict(C=C, max_iter=max_iter, n_iter=int(np.max(lr.n_iter_)), n_train=int(len(y)), pos_rate=float(y.mean()),
                n_features=int(X.shape[1]), solver=lr.solver, sklearn=sklearn.__version__)
    return dict(w=w, b=b, info=info)


def save_gate(path, gate, extra=None):
    """JSON (bobot float64 dengan repr penuh -> dimuat ulang bit-identik)."""
    d = dict(w=[float(v) for v in np.asarray(gate["w"], np.float64)], b=float(gate["b"]), info=gate.get("info", {}),
             **(extra or {}))
    Path(path).write_text(json.dumps(d, indent=1))


def load_gate(path):
    d = json.loads(Path(path).read_text())
    d["w"] = np.asarray(d["w"], np.float64)
    return d


def score_np(feat, gate):
    """Skor gate di numpy float64 (dipakai untuk threshold val, sama dengan eksperimen)."""
    return np.asarray(feat, np.float32) @ np.asarray(gate["w"], np.float64) + float(gate["b"])


class GateHead(nn.Module):
    """(p3, p2, p4, p5) -> skor (B,) FP32 = pool4 · w + b. Dipakai di Stage 1 / engine A3."""

    def __init__(self, w, b):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(np.asarray(w), dtype=torch.float32))
        self.register_buffer("b", torch.tensor([float(b)], dtype=torch.float32))

    def forward(self, p3, p2, p4, p5):
        return pool4(*(t.float() for t in (p3, p2, p4, p5))) @ self.w + self.b
