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

Uji berpasangan (boot_ablation): bootstrap per gambar (indeks resample sama untuk semua routing di satu split) atas
selisih gain dua varian, gain = metrik − rata-rata n_random_boot routing acak pada aktivasi yang sama (set acak
dipakai bersama bila aktivasinya sama, sehingga untuk pasangan top-ρ selisih gain = selisih metrik persis).
p dua sisi + Holm di seluruh keluarga pasangan. Pasangan dipilih setelah titik ablasi seed 42 terlihat (pasca-hasil):
dilaporkan sebagai konfirmasi ketidakpastian, bukan seleksi desain (desain tetap terkunci).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .budget import b2_decide, thr0_from_val
from .cocomix import (METRICS, FastUnionBoot, Router, UnionBoot, _pval, boot_engine, cluster_index, holm,
                      random_masks, top_rho)
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
    rows, masks = [], {}

    def add(grp, varian, split, sp, use, **kw):
        use = np.asarray(use, bool)
        masks.setdefault((split, varian), use)                     # G5: hanya ulangan pertama per n
        rows.append(dict(seri=grp, varian=varian, split=split, aktivasi=float(np.mean(use)),
                         **sp.gain(use, n_random, seed + 3), **kw))

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
        return pd.DataFrame(rows), masks

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
    return pd.DataFrame(rows), masks


# =============================================================================
# Uji berpasangan
# =============================================================================
F_SS, F_LOSS, F_LOSSM = "soft_small (final)", "selisih loss (loss_A < loss_B)", "selisih loss − median (offset ala DynamicDet)"
F_FEAT, F_C = "final: 4 peta, GAP+GMP", "LogReg C=0.001 (final)"
B_FINAL = "B2 window 200 (final)"

# (label, split, varian_a, varian_b): selisih gain a − b. varian_b = None -> a − acak (gain itu sendiri).
PAIRS = [
    ("G1 soft_small − acak", "val", F_SS, None),
    ("G1 soft_small − selisih loss", "val", F_SS, F_LOSS),
    ("G1 soft_small − selisih loss − median", "val", F_SS, F_LOSSM),
    ("G1 soft_small − soft_all", "val", F_SS, "soft_all"),
    ("G1 soft_small − net25_all", "val", F_SS, "net25_all"),
    ("G1 selisih loss − acak", "val", F_LOSS, None),
    ("G1 plafon jumlah GT kecil − oracle Δsoft_small", "val", "plafon: jumlah GT kecil (GT val)",
     "plafon: oracle Δsoft_small (GT val)"),
    ("G3 final − tanpa p2_backbone", "val", F_FEAT, "tanpa p2_backbone"),
    ("G3 final − hanya p3_neck", "val", F_FEAT, "hanya p3_neck"),
    ("G4 LogReg C=0.001 − HistGradientBoosting", "val", F_C, "HistGradientBoosting"),
    ("G4 LogReg C=0.001 − C=0.01", "val", F_C, "LogReg C=0.01"),
    ("G5 semua label − n = 500", "val", "n = {n_all}", "n = 500"),
    ("B B2 window 200 − threshold val tetap", "test", B_FINAL, "threshold val tetap"),
    ("B B2 window 200 − window 500", "test", B_FINAL, "B2 window 500"),
    ("B B2 window 200 − acak", "test", B_FINAL, None),
]


def video_groups(sp):
    """Klaster per video sumber (token pertama nama file, sama dengan suite.py) -> posisi imgIds terurut (UnionBoot)."""
    g = {}
    for j, f in enumerate(sp.C["file"]):
        g.setdefault(f.split("_")[0], []).append(int(sp.R.perm[j]))
    return [np.asarray(v) for v in g.values()]


def boot_ablation(splits, masks, n_boot=1000, seed=20261016, n_random=5, rand_seed=11, n_jobs=1, log=print,
                  cluster_seed=None):
    """
    splits: {'val': Split, 'test': Split|None}; masks: {(split, varian): use_a (urutan cache)} dari run_ablation.
    cluster_seed: None = bootstrap per gambar; angka = bootstrap KLASTER per video (video_groups, cluster_index),
    estimasi titik identik, hanya indeks resample yang berbeda.
    Returns DataFrame: uji, metric, diff, ci_lo, ci_hi, p_two_sided, p_holm (per metrik), n_boot, aktivasi_a/b.
    """
    n_all = max((int(v.split("= ")[1]) for (sp, v) in masks if v.startswith("n = ")), default=None)
    pairs = []
    for lab, sp, a, b in PAIRS:
        a = a.format(n_all=n_all); b = b.format(n_all=n_all) if b else None
        if splits.get(sp) is not None and (sp, a) in masks and (b is None or (sp, b) in masks):
            pairs.append((lab, sp, a, b))
        else:
            log(f"  lewati uji '{lab}' (varian tidak ada)")
    rows = []
    for sp in ("val", "test"):
        P = [p for p in pairs if p[1] == sp]
        if not P:
            continue
        S = splits[sp]
        U = FastUnionBoot(UnionBoot(S.R.bA, S.R.bB))
        names = sorted({v for _, _, a, b in P for v in (a, b) if v})
        mixes, rand_of = {}, {}
        for v in names:
            use = masks[(sp, v)]
            mixes[v] = (U, S.R.sorted_mask(use))
            k = int(use.sum())
            if k not in rand_of:                                       # set acak dipakai bersama per aktivasi
                rand_of[k] = [f"acak_k{k}_{j}" for j in range(n_random)]
                for j, r in enumerate(random_masks(S.n, k / S.n, n_random, rand_seed + k)):
                    mixes[rand_of[k][j]] = (U, S.R.sorted_mask(r))
        if cluster_seed is None:
            log(f"  bootstrap {sp}: {len(mixes)} routing × {n_boot} resample ...")
            point, smp = boot_engine(mixes, n_boot, seed, n_jobs)
        else:
            G = video_groups(S)
            log(f"  bootstrap klaster {sp}: {len(G)} video (terbesar {max(map(len, G))}/{S.n}), "
                f"{len(mixes)} routing × {n_boot} resample ...")
            point, smp = boot_engine(mixes, None, None, n_jobs, idx=cluster_index(G, n_boot, cluster_seed))
        gain = lambda d, v: d[v][m] - np.mean([d[r][m] for r in rand_of[int(masks[(sp, v)].sum())]], 0)
        for lab, _, a, b in P:
            for m in METRICS:
                pt = gain(point, a) - (gain(point, b) if b else 0.0)
                sm = gain(smp, a) - (gain(smp, b) if b else 0.0)
                sm = sm[np.isfinite(sm)]
                lo, hi = np.percentile(sm, [2.5, 97.5])
                rows.append(dict(uji=lab, split=sp, metric=m, diff=pt, ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm),
                                 n_boot=len(sm), aktivasi_a=float(masks[(sp, a)].mean()),
                                 aktivasi_b=float(masks[(sp, b)].mean()) if b else np.nan))
    T = pd.DataFrame(rows)
    if len(T):
        T["p_holm"] = np.nan
        for m in METRICS:
            sel = T.metric == m
            T.loc[sel, "p_holm"] = holm(T.loc[sel, "p_two_sided"].values)
    return T
