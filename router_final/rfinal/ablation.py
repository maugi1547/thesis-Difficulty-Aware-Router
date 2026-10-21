"""
Ablasi pasca-training (router_final): kenapa tiap pilihan desain gate & anggaran masuk sistem final.
Semua di atas detektor BEKU; pilihan dinilai di VAL (gain vs routing acak pada aktivasi sama), test hanya laporan.

  G1  target gate   : soft_small (final) vs soft_all vs net25_all vs selisih loss (loss_A < loss_B) vs selisih loss
                      − median (offset adaptif ala DynamicDet) + plafon ber-GT di val (oracle Δsoft_small, jumlah GT kecil)
  G3  subset fitur  : 4 peta GAP‖GMP (final) vs tanpa p2_backbone vs hanya p3 vs hanya GAP vs hanya GMP
  G4  classifier    : LogReg C ∈ {1e-4 … 1} vs HistGradientBoosting
  G5  biaya label   : n gambar train berlabel ∈ {50 … semua}, 5 subset acak per n
  B   anggaran      : threshold val tetap vs B2 (window 50 / 200 / 500) pada aliran test: aktivasi & gain vs acak
Ukuran gain = metrik(routing) − rata-rata metrik(20 routing acak, aktivasi sama), CocoMix (= COCOeval eksak).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .budget import b2_decide, thr0_from_val
from .cocomix import METRICS, Router, random_masks, top_rho
from .data import build_coco_gt, coco_eval, jdict
from .gate import cache_split, fit_gate, score_np

# kanal YOLOv8n: p3_neck 64, p2_backbone 32, p4_neck 128, p5_sppf 256; pool4 = per peta [GAP | GMP]
CH = dict(p3=64, p2=32, p4=128, p5=256)


def feature_columns():
    idx, o = {}, 0
    for k, c in CH.items():
        idx[k] = dict(gap=np.arange(o, o + c), gmp=np.arange(o + c, o + 2 * c)); o += 2 * c
    assert o == 960
    cols = lambda maps, kinds=("gap", "gmp"): np.concatenate([idx[m][t] for m in maps for t in kinds])
    return {"final: 4 peta, GAP+GMP": cols(CH), "tanpa p2_backbone": cols(["p3", "p4", "p5"]),
            "hanya p3_neck": cols(["p3"]), "hanya GAP": cols(CH, ("gap",)), "hanya GMP": cols(CH, ("gmp",))}


@torch.no_grad()
def per_image_losses(model, loader, device, max_images=None):
    """loss deteksi stok per gambar (box·7.5 + cls·0.5 + dfl·1.5) untuk Branch A dan B (batch 1, model eval)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from .loss import BranchLoss
    if not hasattr(getattr(model, "args", None), "box"):
        model.args = get_cfg(DEFAULT_CFG)
    LA, LB = BranchLoss(model, model.detect_A), BranchLoss(model, model.detect_B)
    out = np.zeros((0, 2))
    rows = []
    for i, b in enumerate(loader):
        if max_images and i >= max_images:
            break
        im = b["img"].to(device).float() / 255.0
        oa, ob = model.forward_dual(im)
        batch = {k: b[k].to(device) for k in ("batch_idx", "cls", "bboxes")}
        la, _ = LA(oa, batch); lb, _ = LB(ob, batch)
        rows.append((float(la.sum()), float(lb.sum())))
    return np.asarray(rows, float) if rows else out


class Split:
    """Cache satu split: fitur, deteksi A/B, Router (CocoMix), target per gambar."""

    def __init__(self, model, loader, device, nms, gt_path, max_dets, log=print):
        t = time.perf_counter()
        self.C = cache_split(model, loader, device, nms, log=log)
        gt = build_coco_gt(loader.dataset, loader.dataset.data["names"], gt_path)
        ids = self.C["image_id"]
        _, EA = coco_eval(gt, jdict(self.C["detA"], ids), max_dets)
        _, EB = coco_eval(gt, jdict(self.C["detB"], ids), max_dets)
        self.R = Router(EA, EB, ids)
        self.T = self.R.targets()
        self.X = np.asarray(self.C["feat"], np.float64)
        self.n = len(ids)
        log(f"  cache {self.n} gambar ({time.perf_counter() - t:.0f} s)")

    def gain(self, use_a, n_random, seed):
        m = self.R.metrics(use_a)
        base = [self.R.metrics(r) for r in random_masks(self.n, float(np.mean(use_a)), n_random, seed)]
        return {f"gain_{k}": m[k] - float(np.mean([b[k] for b in base])) for k in METRICS}


def _fit_score(kind, Xtr, ytr_delta, Xev, C=0.001):
    if kind == "logreg":
        g = fit_gate(Xtr, ytr_delta, C=C)
        return score_np(Xev, g), g
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = (np.asarray(ytr_delta) > 0).astype(int)
    clf = HistGradientBoostingClassifier(random_state=0).fit(Xtr, y)
    return clf.predict_proba(Xev)[:, 1], None


def _two_class(t):
    k = int((np.asarray(t) > 0).sum())
    return 0 < k < len(t)


def run_ablation(tr: Split, va: Split, te: Split | None, losses_tr, rho=0.5, C=0.001, n_random=20, seed=0,
                 b2=(200, 20), log=print):
    rows = []
    add = lambda grp, varian, split, sp, use, **kw: rows.append(
        dict(seri=grp, varian=varian, split=split, aktivasi=float(np.mean(use)), **sp.gain(use, n_random, seed + 3), **kw))

    # ---- G1 target
    targets = {"soft_small (final)": tr.T["soft_small"], "soft_all": tr.T["soft_all"], "net25_all": tr.T["net25_all"]}
    if losses_tr is not None and len(losses_tr):
        d = losses_tr[:, 1] - losses_tr[:, 0]                       # > 0 -> A lebih kecil loss-nya
        targets["selisih loss (loss_A < loss_B)"] = d
        targets["selisih loss − median (offset ala DynamicDet)"] = d - np.median(d)
    for nm, tgt in targets.items():
        if _two_class(tgt):
            s, _ = _fit_score("logreg", tr.X, tgt, va.X, C)
            add("G1 target", nm, "val", va, top_rho(s, rho), pos_rate_train=float((np.asarray(tgt) > 0).mean()))
    add("G1 target", "plafon: oracle Δsoft_small (GT val)", "val", va, top_rho(va.T["soft_small"], rho))
    add("G1 target", "plafon: jumlah GT kecil (GT val)", "val", va, top_rho(va.T["n_gt_small"], rho))

    if not _two_class(tr.T["soft_small"]):
        log("⚠ target soft_small train hanya satu kelas: G3–G5 dan B dilewati")
        return pd.DataFrame(rows)

    # ---- G3 subset fitur
    for nm, cols in feature_columns().items():
        s, _ = _fit_score("logreg", tr.X[:, cols], tr.T["soft_small"], va.X[:, cols], C)
        add("G3 fitur", nm, "val", va, top_rho(s, rho), n_fitur=len(cols))

    # ---- G4 classifier
    for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        s, _ = _fit_score("logreg", tr.X, tr.T["soft_small"], va.X, c)
        add("G4 classifier", f"LogReg C={c:g}" + (" (final)" if c == C else ""), "val", va, top_rho(s, rho))
    s, _ = _fit_score("hgb", tr.X, tr.T["soft_small"], va.X)
    add("G4 classifier", "HistGradientBoosting", "val", va, top_rho(s, rho))

    # ---- G5 biaya label
    rng = np.random.default_rng(seed + 5)
    for n_lab in [n for n in (50, 100, 250, 500, 1000, 2500) if n < tr.n] + [tr.n]:
        for rep in range(5 if n_lab < tr.n else 1):
            sel = rng.choice(tr.n, n_lab, replace=False) if n_lab < tr.n else np.arange(tr.n)
            y = tr.T["soft_small"][sel]
            if not 0 < int((y > 0).sum()) < len(y):
                continue
            s, _ = _fit_score("logreg", tr.X[sel], y, va.X, C)
            add("G5 label", f"n = {n_lab}", "val", va, top_rho(s, rho), n_label=n_lab, ulangan=rep)

    # ---- B anggaran: aliran test (urutan file) dengan gate final
    if te is not None:
        g = fit_gate(tr.X, tr.T["soft_small"], C=C)
        s_va, s_te = score_np(va.X, g), score_np(te.X, g)
        thr0 = thr0_from_val(s_va, rho)
        add("B anggaran", "threshold val tetap", "test", te, np.float32(s_te) > np.float32(thr0))
        for win in (50, 200, 500):
            d_, _ = b2_decide(s_te, rho, win, b2[1], thr0)
            add("B anggaran", f"B2 window {win}" + (" (final)" if win == b2[0] else ""), "test", te, d_)
        add("B anggaran", "top-ρ test (oracle anggaran, transduktif)", "test", te, top_rho(s_te, rho))
    return pd.DataFrame(rows)
