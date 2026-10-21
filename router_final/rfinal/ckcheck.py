"""
Cek checkpoint: last.pt (dipakai semua tahap) vs best.pt (dipilih Ultralytics dari fitness Branch A saja) di VAL.

Kenapa: L = L_A + 0,7·L_B, tetapi best.pt = argmax fitness (0,1·mAP50 + 0,9·mAP50-95) Branch A. Epoch terbaik A
belum tentu terbaik B, dan memilih epoch dari A bisa membesarkan jarak A − B (target gate Δ soft_small).
Aturan (dikunci sebelum training): last.pt TETAP dipakai; skrip ini hanya MELAPORKAN selisihnya (bukan seleksi).
Test tidak pernah disentuh.

Uji (bootstrap berpasangan per gambar val, indeks resample sama untuk keempat sumber):
  A_last − A_best, B_last − B_best, dan selisih jarak (A_last − B_last) − (A_best − B_best).
Tambahan (titik, GT val): plafon oracle Δ soft_small pada ρ — gain vs routing acak aktivasi sama, per checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cocomix import METRICS, FastUnionBoot, Router, UnionBoot, _pval, boot_engine, random_masks, top_rho
from .data import build_coco_gt, coco_eval, jdict
from .gate import cache_split
from .repro import state_hash


def best_epoch_from_results(run_dir):
    """Epoch best.pt menurut results.csv (fitness bawaan 0,1·mAP50 + 0,9·mAP50-95; seri -> epoch terakhir)."""
    p = Path(run_dir) / "results.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    j = int(np.flatnonzero(fit.values == fit.max())[-1])
    return int(df["epoch"].iloc[j]), df.assign(fitness_A=fit)


def branch_b_log(run_dir):
    p = Path(run_dir) / "branch_b_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def same_weights(path_a, path_b):
    import torch
    h = []
    for p in (path_a, path_b):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        k = "ema" if ck.get("ema") is not None else "model"
        h.append(state_hash(ck[k]))
    return h[0] == h[1]


def evaluate_pair(models, loader, device, nms, gt_path, max_dets, rho=0.5, n_boot=1000, seed=20261021,
                  n_jobs=1, n_random=20, log=print):
    """models: {'last': model, 'best': model}. Returns (titik, uji, plafon) DataFrame."""
    gt = build_coco_gt(loader.dataset, loader.dataset.data["names"], gt_path)
    E, ids = {}, None
    for nm, m in models.items():
        log(f"  cache val [{nm}] ...")
        C = cache_split(m, loader, device, nms, log=log)
        assert ids is None or ids == C["image_id"], "urutan gambar beda antar-checkpoint"
        ids = C["image_id"]
        for br in ("A", "B"):
            _, E[f"{br}_{nm}"] = coco_eval(gt, jdict(C[f"det{br}"], ids), max_dets)

    # plafon oracle per checkpoint (A vs B dari checkpoint yang sama)
    plafon = []
    for nm in models:
        R = Router(E[f"A_{nm}"], E[f"B_{nm}"], ids)
        use = top_rho(R.targets()["soft_small"], rho)
        mo = R.metrics(use)
        base = [R.metrics(r) for r in random_masks(R.n, float(np.mean(use)), n_random, 3)]
        plafon.append(dict(checkpoint=nm, rho=rho, aktivasi=float(np.mean(use)),
                           **{f"oracle_{k}": mo[k] for k in METRICS},
                           **{f"gain_vs_acak_{k}": mo[k] - float(np.mean([b[k] for b in base])) for k in METRICS}))

    # bootstrap berpasangan: satu UnionBoot per branch (sumber 1 = last, sumber 2 = best)
    mixes = {}
    for br in ("A", "B"):
        R = Router(E[f"{br}_last"], E[f"{br}_best"], ids)
        U = FastUnionBoot(UnionBoot(R.bA, R.bB))
        mixes[f"{br}_last"] = (U, np.ones(R.n, bool))
        mixes[f"{br}_best"] = (U, np.zeros(R.n, bool))
    log(f"  bootstrap {n_boot} resample ...")
    point, smp = boot_engine(mixes, n_boot, seed, n_jobs)

    rows = []
    tests = {"A: last − best": (("A_last", 1), ("A_best", -1)),
             "B: last − best": (("B_last", 1), ("B_best", -1)),
             "jarak A−B: last − best": (("A_last", 1), ("B_last", -1), ("A_best", -1), ("B_best", 1))}
    for label, terms in tests.items():
        for m in METRICS:
            pt = sum(s * point[k][m] for k, s in terms)
            sm = sum(s * smp[k][m] for k, s in terms); sm = sm[np.isfinite(sm)]
            lo, hi = np.percentile(sm, [2.5, 97.5])
            rows.append(dict(uji=label, metric=m, diff=pt, ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm), n_boot=len(sm)))
    titik = pd.DataFrame([dict(sumber=k, **v) for k, v in point.items()])
    return titik, pd.DataFrame(rows), pd.DataFrame(plafon)


def verdict(uji: pd.DataFrame, metric="mAP_small"):
    """Ringkasan teks. Tidak mengubah pilihan checkpoint (last.pt dikunci)."""
    out = []
    for _, r in uji[uji.metric == metric].iterrows():
        sig = not (r.ci_lo <= 0 <= r.ci_hi)
        arah = "last lebih baik" if r["diff"] > 0 else "best lebih baik"
        out.append(f"{r.uji:<24} {metric} Δ={r['diff']:+.4f} CI95 [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
                   + (f"⚠ signifikan ({arah})" if sig else "≈ dalam noise"))
    return "\n".join(out)
