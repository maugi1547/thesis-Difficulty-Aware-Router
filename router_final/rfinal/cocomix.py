"""
Metrik routing & statistik (router_final). Murni numpy di atas evalImgs pycocotools — tidak bergantung pada model.

Diporting tanpa perubahan semantik dari kode yang menghasilkan angka Tahap 4/4B:
  CocoBootstrap, BOOT_METRICS, holm           <- eksperimen/tahap2_eval.py
  CocoMix, Router, top_rho, random_masks      <- eksperimen/tahap2b_plafon.py
  UnionBoot, boot_engine, boot_tests, _pval,
  partial_spearman_p, pooled_tests            <- eksperimen/tahap4b.py
Kesetaraan diuji pada intermediate Kaggle Tahap 4 (tests/test_suite_replay.py): bootstrap_test.csv identik.

Konvensi urutan: array "per gambar" = urutan cache/loader; CocoBootstrap & UnionBoot memakai urutan imgIds COCO
(sorted). Router.sorted_mask() memetakan antar keduanya.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

BOOT_METRICS = ("mAP50-95", "mAP50", "mAP_small")
METRICS = BOOT_METRICS
TARGETS = ("soft_small", "soft_all", "net25_all")


# =============================================================================
# evalImgs -> AP untuk subset gambar (boleh berulang)
# =============================================================================
class CocoBootstrap:
    """AP COCO untuk subset gambar langsung dari evalImgs (replikasi accumulate + summarize, maxDet terakhir)."""
    AREAS = {"all": 0, "small": 1, "medium": 2, "large": 3}

    def __init__(self, E):
        p = E.params
        self.T, self.R, self.I = len(p.iouThrs), p.recThrs, len(p.imgIds)
        self.A, self.K, self.maxDet = len(p.areaRng), len(p.catIds), p.maxDets[-1]
        self.iou50 = int(np.where(np.isclose(p.iouThrs, 0.5))[0][0])
        self.stats_full = np.asarray(E.stats, float)
        self.cells = {}
        for k in range(self.K):
            for name, a in self.AREAS.items():
                base = k * self.A * self.I + a * self.I
                lst = []
                for i in range(self.I):
                    e = E.evalImgs[base + i]
                    if e is None:
                        lst.append(None); continue
                    lst.append((np.asarray(e["dtScores"][:self.maxDet], float),
                                np.asarray(e["dtMatches"][:, :self.maxDet]) > 0,
                                np.asarray(e["dtIgnore"][:, :self.maxDet], bool),
                                int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))))
                self.cells[(k, name)] = lst

    def _precision(self, idx, area):
        out = []
        for k in range(self.K):
            cell = self.cells[(k, area)]
            E = [cell[i] for i in idx if cell[i] is not None]
            if not E:
                out.append(None); continue
            npig = sum(e[3] for e in E)
            if npig == 0:
                out.append(None); continue
            scores = np.concatenate([e[0] for e in E])
            order = np.argsort(-scores, kind="mergesort")
            dtm = np.concatenate([e[1] for e in E], axis=1)[:, order]
            dtig = np.concatenate([e[2] for e in E], axis=1)[:, order]
            tp = np.cumsum(dtm & ~dtig, axis=1, dtype=float)
            fp = np.cumsum(~dtm & ~dtig, axis=1, dtype=float)
            nd = tp.shape[1]
            q = np.zeros((self.T, len(self.R)))
            if nd:
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                for t in range(self.T):
                    ii = np.searchsorted(rc[t], self.R, side="left")
                    ok = ii < nd
                    q[t, ok] = pr[t, ii[ok]]
            out.append(q)
        return out

    def metrics(self, idx):
        P_all = [p for p in self._precision(idx, "all") if p is not None]
        P_s = [p for p in self._precision(idx, "small") if p is not None]
        m = {}
        if P_all:
            P = np.stack(P_all); m["mAP50-95"] = float(P.mean()); m["mAP50"] = float(P[:, self.iou50].mean())
        else:
            m["mAP50-95"] = m["mAP50"] = np.nan
        m["mAP_small"] = float(np.stack(P_s).mean()) if P_s else np.nan
        return m


class CocoMix(CocoBootstrap):
    """Metrik COCO untuk routing per gambar dari evalImgs A & B (indeks = urutan imgIds)."""

    def __init__(self, bA, bB, use_a_sorted):
        for k in ("T", "R", "I", "A", "K", "maxDet", "iou50"):
            setattr(self, k, getattr(bA, k))
        assert bA.I == bB.I == len(use_a_sorted)
        self.cells = {key: [ca if u else cb for ca, cb, u in zip(bA.cells[key], bB.cells[key], use_a_sorted)]
                      for key in bA.cells}


class Router:
    """Dua sumber deteksi (A: use_a True, B: False) -> metrik routing apa pun + target per gambar."""

    def __init__(self, E_A, E_B, image_ids):
        self.bA, self.bB = CocoBootstrap(E_A), CocoBootstrap(E_B)
        ids_sorted = list(E_A.params.imgIds)
        assert ids_sorted == list(E_B.params.imgIds)
        pos = {iid: j for j, iid in enumerate(ids_sorted)}
        self.perm = np.array([pos[i] for i in image_ids])
        self.n = len(image_ids)

    def sorted_mask(self, use_a):
        m = np.zeros(self.n, bool); m[self.perm] = np.asarray(use_a, bool)
        return m

    def mix(self, use_a):
        return CocoMix(self.bA, self.bB, self.sorted_mask(use_a))

    def metrics(self, use_a, idx=None):
        return self.mix(use_a).metrics(np.arange(self.n) if idx is None else idx)

    def targets(self, conf_thr=0.25):
        """Δ per gambar (A − B), urutan cache, + jumlah GT kecil & semua (non-ignored)."""
        def per_image(b):
            res = {t: np.zeros(self.n) for t in TARGETS}
            res["n_gt_small"] = np.zeros(self.n); res["n_gt_all"] = np.zeros(self.n)
            t50 = b.iou50
            for (k, area), cells in b.cells.items():
                if area not in ("all", "small"):
                    continue
                for j, c in enumerate(cells):
                    if c is None:
                        continue
                    sc, dtm, dtig, npig = c
                    tp = dtm[t50] & ~dtig[t50]
                    if area == "small":
                        res["soft_small"][j] += sc[tp].sum(); res["n_gt_small"][j] += npig
                    else:
                        res["soft_all"][j] += sc[tp].sum(); res["n_gt_all"][j] += npig
                        fp = ~dtm[t50] & ~dtig[t50]
                        res["net25_all"][j] += (tp & (sc >= conf_thr)).sum() - (fp & (sc >= conf_thr)).sum()
            return res
        rA, rB = per_image(self.bA), per_image(self.bB)
        out = {t: (rA[t] - rB[t])[self.perm] for t in TARGETS}
        out["n_gt_small"] = rA["n_gt_small"][self.perm]; out["n_gt_all"] = rA["n_gt_all"][self.perm]
        return out


def top_rho(score, rho, rng=None):
    """use_a = ρ·N gambar dengan skor tertinggi (seri dipecah acak tetapi deterministik)."""
    n = len(score); k = int(round(rho * n))
    rng = rng or np.random.default_rng(0)
    order = np.lexsort((rng.random(n), -np.asarray(score, float)))
    m = np.zeros(n, bool); m[order[:k]] = True
    return m


def random_masks(n, rho, n_draw, seed):
    rng = np.random.default_rng(seed)
    k = int(round(rho * n))
    out = []
    for _ in range(n_draw):
        m = np.zeros(n, bool); m[rng.choice(n, k, replace=False)] = True; out.append(m)
    return out


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


# =============================================================================
# Bootstrap berpasangan cepat (indeks resample sama untuk semua routing)
# =============================================================================
class UnionBoot:
    """
    Deteksi dua sumber diurutkan SEKALI per (kategori, area) dengan kunci (−skor, gambar, sumber, posisi) = urutan
    accumulate COCOeval untuk routing apa pun; resample dengan multiplisitas c_i = titik berbobot c_i (eksak untuk
    presisi terinterpolasi). Pada himpunan penuh == COCOeval persis.
    """
    AREAS = ("all", "small")

    def __init__(self, bA, bB):
        self.I, self.T, self.R, self.K, self.iou50 = bA.I, bA.T, np.asarray(bA.R), bA.K, bA.iou50
        assert bB.I == self.I and bB.K == self.K
        self.cells = {}
        for k in range(self.K):
            for area in self.AREAS:
                sc, tp, fp, img, src, pos = [], [], [], [], [], []
                npig = np.zeros(self.I, np.int64)
                for s_i, b in enumerate((bA, bB)):
                    for i, c in enumerate(b.cells[(k, area)]):
                        if c is None:
                            continue
                        s_, dtm, dtig, n_ = c
                        npig[i] = n_
                        if len(s_):
                            sc.append(s_); tp.append(dtm & ~dtig); fp.append(~dtm & ~dtig)
                            img.append(np.full(len(s_), i, np.int64)); src.append(np.full(len(s_), s_i, np.int8))
                            pos.append(np.arange(len(s_)))
                if sc:
                    sc, img, src, pos = map(np.concatenate, (sc, img, src, pos))
                    tp, fp = np.concatenate(tp, 1), np.concatenate(fp, 1)
                    o = np.lexsort((pos, src, img, -sc))
                    o = o[(tp[:, o] | fp[:, o]).any(0)]
                    self.cells[(k, area)] = (tp[:, o], fp[:, o], img[o], src[o], npig)
                else:
                    self.cells[(k, area)] = (np.zeros((self.T, 0), bool), np.zeros((self.T, 0), bool),
                                             np.zeros(0, np.int64), np.zeros(0, np.int8), npig)

    def metrics(self, use_a_sorted, idx):
        c = np.bincount(idx, minlength=self.I)
        P_ = {"all": [], "small": []}
        for (k, area), (tp, fp, img, src, npig) in self.cells.items():
            npg = int((c * npig).sum())
            if npg == 0:
                continue
            q = np.zeros((self.T, len(self.R)))
            if len(img):
                w = c[img] * ((src == 0) == use_a_sorted[img])
                sel = np.flatnonzero(w)
                if len(sel):
                    ww = w[sel].astype(np.int32)
                    tpc = np.cumsum(tp[:, sel] * ww, axis=1, dtype=np.int32).astype(float)
                    fpc = np.cumsum(fp[:, sel] * ww, axis=1, dtype=np.int32).astype(float)
                    rc, pr = tpc / npg, tpc / (fpc + tpc + np.spacing(1))
                    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                    nd = len(sel)
                    for t in range(self.T):
                        ii = np.searchsorted(rc[t], self.R, side="left"); ok = ii < nd
                        q[t, ok] = pr[t, ii[ok]]
            P_[area].append(q)
        m = {}
        if P_["all"]:
            Pa = np.stack(P_["all"]); m["mAP50-95"] = float(Pa.mean()); m["mAP50"] = float(Pa[:, self.iou50].mean())
        else:
            m["mAP50-95"] = m["mAP50"] = np.nan
        m["mAP_small"] = float(np.stack(P_["small"]).mean()) if P_["small"] else np.nan
        return m


class FastUnionBoot:
    """
    Metrik identik dengan UnionBoot, tetapi hanya dihitung di posisi TP (presisi terinterpolasi hanya perlu
    dievaluasi di TP); jumlah deteksi non-abaikan sampai posisi p = cumsum bobot basis + koreksi jarang.
    ±7× lebih cepat di VisDrone (dipakai bootstrap klaster). Kesetaraan: tests/test_cocomix.py.
    """

    def __init__(self, U):
        self.I, self.T, self.R, self.K, self.iou50 = U.I, U.T, U.R, U.K, U.iou50
        self.cells = {}
        for key, (tp, fp, img, src, npig) in U.cells.items():
            nonig = tp | fp
            base = nonig.all(0)
            tp_pos = [np.flatnonzero(tp[t]) for t in range(self.T)]
            extra = [np.flatnonzero(nonig[t] & ~base) for t in range(self.T)]
            self.cells[key] = (img, src.astype(bool) == False, npig, base, tp_pos, extra)  # noqa: E712

    def metrics(self, use_a_sorted, idx):
        c = np.bincount(idx, minlength=self.I)
        P_ = {"all": [], "small": []}
        for (k, area), (img, is_a, npig, base, tp_pos, extra) in self.cells.items():
            npg = int((c * npig).sum())
            if npg == 0:
                continue
            q = np.zeros((self.T, len(self.R)))
            if len(img):
                w = c[img] * (is_a == use_a_sorted[img])
                Nb = np.cumsum(w * base)
                for t in range(self.T):
                    pos = tp_pos[t]
                    wt = w[pos]; keep = wt > 0
                    if not keep.any():
                        continue
                    pos, wt = pos[keep], wt[keep]
                    tpc = np.cumsum(wt).astype(float)
                    ex = extra[t]
                    if len(ex):
                        cex = np.cumsum(w[ex]); kk = np.searchsorted(ex, pos, side="right")
                        corr = np.where(kk > 0, cex[np.maximum(kk - 1, 0)], 0)
                    else:
                        corr = 0
                    N = (Nb[pos] + corr).astype(float)
                    rc, pr = tpc / npg, tpc / (N + np.spacing(1))
                    pr = np.maximum.accumulate(pr[::-1])[::-1]
                    ii = np.searchsorted(rc, self.R, side="left"); ok = ii < len(pos)
                    q[t, ok] = pr[ii[ok]]
            P_[area].append(q)
        m = {}
        if P_["all"]:
            Pa = np.stack(P_["all"]); m["mAP50-95"] = float(Pa.mean()); m["mAP50"] = float(Pa[:, self.iou50].mean())
        else:
            m["mAP50-95"] = m["mAP50"] = np.nan
        m["mAP_small"] = float(np.stack(P_["small"]).mean()) if P_["small"] else np.nan
        return m


_BCTX = {}


def _boot_rows(rows):
    mixes, idx = _BCTX["mixes"], _BCTX["idx"]
    return [{k: U.metrics(u, idx[r]) for k, (U, u) in mixes.items()} for r in rows]


def boot_engine(mixes, n_boot, seed, n_jobs=1, idx=None):
    """
    mixes: {nama: (UnionBoot, use_a urutan imgIds)}. idx: indeks resample (n_boot, I) — default = rng(seed).integers,
    sama untuk semua model (pooling antar-seed). Returns (titik, sampel {nama: {metrik: array}}).
    """
    U0 = next(iter(mixes.values()))[0]
    I = U0.I
    assert all(U.I == I for U, _ in mixes.values())
    if idx is None:
        idx = np.random.default_rng(seed).integers(0, I, (n_boot, I))
    n_boot = len(idx)
    t1 = time.perf_counter()
    point = {k: U.metrics(u, np.arange(I)) for k, (U, u) in mixes.items()}
    t1 = time.perf_counter() - t1
    _BCTX.update(mixes=mixes, idx=idx)
    chunks = [c for c in np.array_split(np.arange(n_boot), max(1, 4 * n_jobs)) if len(c)]
    parts = None
    try:
        if n_jobs > 1:
            timeout = max(300.0, 4.0 * t1 * n_boot / n_jobs)
            pool = mp.get_context("fork").Pool(n_jobs)
            try:
                parts = pool.map_async(_boot_rows, chunks).get(timeout=timeout)
                pool.close()
            except mp.TimeoutError:
                print(f"⚠ bootstrap paralel melewati {timeout:.0f} s -> diulang serial", flush=True)
            finally:
                pool.terminate(); pool.join()
        if parts is None:
            parts = [_boot_rows(c) for c in chunks]
    finally:
        _BCTX.clear()
    flat = [d for p_ in parts for d in p_]
    samples = {k: {m: np.array([d[k][m] for d in flat], float) for m in METRICS} for k in mixes}
    return point, samples


def _pval(s):
    return float(min(1.0, max(2 * min((s <= 0).mean(), (s >= 0).mean()), 1.0 / len(s))))


def boot_tests(point, samples, tests):
    """tests: [(label, a, [b...])] -> DataFrame selisih a − rata-rata(b) + CI persentil 95%."""
    rows = []
    for label, a, bs in tests:
        if a not in point or not bs or any(b not in point for b in bs):
            continue
        for m in METRICS:
            pt = point[a][m] - np.nanmean([point[b][m] for b in bs])
            sm = samples[a][m] - np.nanmean(np.stack([samples[b][m] for b in bs]), 0)
            sm = sm[np.isfinite(sm)]
            lo, hi = np.percentile(sm, [2.5, 97.5])
            rows.append(dict(uji=label, metric=m, diff=pt, ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm), n_boot=len(sm)))
    return pd.DataFrame(rows)


def partial_spearman_p(x, y, controls):
    """ρ Spearman parsial (rank kontrol diregresikan keluar) + p dua sisi (t, df = n − 2 − k)."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    R = lambda v: pd.Series(np.asarray(v, float)).rank().values
    Xc = np.c_[np.ones(len(x)), *[R(c) for c in controls]]
    res = lambda v: R(v) - Xc @ np.linalg.lstsq(Xc, R(v), rcond=None)[0]
    rx, ry = res(x), res(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(rx, ry)[0, 1]); df = len(x) - 2 - len(controls)
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df)), len(x)


def pooled_tests(dirs, tests_wanted):
    """Selisih rata-rata antar-model per resample (indeks resample sama -> sah bila test sama, mis. KITTI 3 seed)."""
    S, Pt, Tdef = {}, {}, None
    for d in map(Path, dirs):
        z = np.load(d / "boot_samples.npz"); S[d.name] = {k: z[k] for k in z.files}
        Pt[d.name] = json.loads((d / "boot_point.json").read_text())
        Tdef = Tdef or json.loads((d / "boot_tests.json").read_text())
    rows = []
    for label, a, bs in Tdef:
        if label not in tests_wanted:
            continue
        ok = [m for m in S if all(f"{x}|mAP_small" in S[m] for x in [a, *bs])]
        if len(ok) != len(S):
            continue
        for met in METRICS:
            diffs = [S[m][f"{a}|{met}"] - np.nanmean(np.stack([S[m][f"{b}|{met}"] for b in bs]), 0) for m in ok]
            pts = [Pt[m][a][met] - np.nanmean([Pt[m][b][met] for b in bs]) for m in ok]
            sm = np.mean(np.stack(diffs), 0); sm = sm[np.isfinite(sm)]
            lo, hi = np.percentile(sm, [2.5, 97.5])
            rows.append(dict(uji=label, metric=met, diff_rata2=float(np.mean(pts)), ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm),
                             n_model=len(ok)))
    return pd.DataFrame(rows)


def cluster_index(groups, n_boot, seed):
    """Indeks resample bootstrap KLASTER: groups = list array posisi imgIds per klaster (mis. per video VisDrone)."""
    rng = np.random.default_rng(seed)
    G = [np.asarray(g) for g in groups]
    return [np.concatenate([G[k] for k in rng.integers(0, len(G), len(G))]) for _ in range(n_boot)]
