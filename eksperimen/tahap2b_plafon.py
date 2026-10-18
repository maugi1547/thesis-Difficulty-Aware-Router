"""
tahap2b_plafon.py — Uji plafon routing + gate post-hoc (Opsi A) untuk Router-P2 Dual-Branch.

Pertanyaan:
  (1) PLAFON: adakah routing per gambar (A vs B) yang mengalahkan routing acak secara berarti pada aktivasi
      yang sama, bila target diukur dari HASIL DETEKSI (bukan loss)? Oracle memakai GT test -> batas atas,
      termasuk optimisme seleksi; bukan klaim deployment.
  (2) DAPAT DIPELAJARI: gate baru di atas detektor BEKU (fitur pooled dari keluaran Stage 1 — sudah tersedia
      di engine Stage 1, jadi deployable), dilatih di TRAIN, dipilih di VAL dengan aturan tetap, dinilai
      SEKALI di TEST.

Desain:
  - Satu forward per gambar (letterbox persegi 640, format deployment): z gate lama, fitur pooled
    (GAP+GMP dari p3_neck, p2_backbone, p4_neck, p5_sppf), deteksi pasca-NMS Branch A & B (conf 0.001,
    IoU 0.7, max_det 300 — sama dengan Tahap 2), dikonversi ke koordinat asli PER SUMBU.
  - COCOeval dijalankan SEKALI untuk A dan SEKALI untuk B per split. Karena evaluateImg COCO independen per
    gambar, metrik routing apa pun = gabungan evalImgs A/B per gambar (CocoMix) -> ratusan routing murah.
    Diverifikasi sama persis dengan COCOeval pada jdict ter-route (verify_mix).
  - Target per gambar dari evalImgs (IoU 0.5): soft-recall objek kecil (jumlah skor deteksi TP pada GT kecil),
    soft-recall semua objek, dan TP−FP semua objek pada conf ≥ 0.25. Δ = nilai A − nilai B.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tahap2_eval as E
from router_train_utils import find_router, _raw_gate_logit

TARGETS = ("soft_small", "soft_all", "net25_all")


# =============================================================================
# CACHE: deteksi pasca-NMS A/B (koordinat asli), z, fitur pooled
# =============================================================================
def _scale_xyxy(b, ratio_pad, ori_shape):
    (gh, gw), (px, py) = ratio_pad
    b = b.clone()
    b[:, [0, 2]] = ((b[:, [0, 2]] - px) / gw).clamp(0, ori_shape[1])
    b[:, [1, 3]] = ((b[:, [1, 3]] - py) / gh).clamp(0, ori_shape[0])
    return b


def _to_dets(y, ratio_pad, ori_shape, conf, iou, max_det):
    """y (1,4+nc,N) -> array (n,6): x, y, w, h (COCO, piksel asli), skor, category_id (=cls+1)."""
    from ultralytics.utils.nms import non_max_suppression
    x = non_max_suppression(y.clone(), conf, iou, nc=0, multi_label=True, agnostic=False, max_det=max_det)[0]
    if not len(x):
        return np.zeros((0, 6), np.float32)
    b = _scale_xyxy(x[:, :4], ratio_pad, ori_shape)
    out = torch.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], x[:, 4], x[:, 5] + 1], 1)
    return out.float().cpu().numpy()


def _pool(t):
    return torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1)


@torch.no_grad()
def cache_dets(model, stages, loader, device, conf=0.001, iou=0.7, max_det=300, max_images=None, log_every=1000):
    """Loader batch 1, urutan tetap. Returns dict file, image_id, z, feat (N,D), detA, detB (list array (n,6))."""
    s1, s2a, s2b = stages
    rt = find_router(model)
    out = dict(file=[], image_id=[], z=[], feat=[], detA=[], detB=[])
    t0 = time.perf_counter()
    for i, b in enumerate(loader):
        if max_images and i >= max_images:
            break
        assert b["img"].shape[0] == 1
        im = b["img"].to(device).float() / 255.0
        _, p3, p2, p4, p5 = s1(im)
        rp, osh = b["ratio_pad"][0], b["ori_shape"][0]
        out["detA"].append(_to_dets(s2a(p3, p2, p4, p5), rp, osh, conf, iou, max_det))
        out["detB"].append(_to_dets(s2b(p3, p4, p5), rp, osh, conf, iou, max_det))
        out["z"].append(float(_raw_gate_logit(rt, p3, p2)[0]))
        out["feat"].append(torch.cat([_pool(p3), _pool(p2), _pool(p4), _pool(p5)], 1)[0].float().cpu().numpy())
        stem = Path(b["im_file"][0]).stem
        out["file"].append(Path(b["im_file"][0]).name)
        out["image_id"].append(int(stem) if stem.isnumeric() else stem)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1} gambar | {time.perf_counter() - t0:.0f}s", flush=True)
    out["z"] = np.asarray(out["z"]); out["feat"] = np.stack(out["feat"])
    return out


def jdict(dets, image_ids, use_a=None):
    """dets: dict detA/detB; use_a None -> pakai detA semua. Returns list COCO results."""
    js = []
    for i, iid in enumerate(image_ids):
        d = dets["detA"][i] if (use_a is None or use_a[i]) else dets["detB"][i]
        for x, y, w, h, s, c in d:
            js.append({"image_id": iid, "category_id": int(c), "bbox": [float(x), float(y), float(w), float(h)], "score": float(s)})
    return js


def subset_coco_gt(dataset, names, path, n):
    """GT COCO untuk n gambar pertama dataset (sesuai cache_dets dengan max_images)."""
    ds = copy.copy(dataset); ds.labels = dataset.labels[:n]
    return E.build_coco_gt(ds, names, path)


# =============================================================================
# CAMPURAN evalImgs A/B -> metrik routing apa pun
# =============================================================================
class CocoMix(E.CocoBootstrap):
    """Metrik COCO untuk routing per gambar dari evalImgs A & B (indeks = urutan E.params.imgIds)."""
    def __init__(self, bA, bB, use_a_sorted):
        for k in ("T", "R", "I", "A", "K", "maxDet", "iou50"):
            setattr(self, k, getattr(bA, k))
        assert bA.I == bB.I == len(use_a_sorted)
        self.cells = {key: [ca if u else cb for ca, cb, u in zip(bA.cells[key], bB.cells[key], use_a_sorted)]
                      for key in bA.cells}


class Router:
    """Mengelola urutan: array per gambar (urutan cache/loader) <-> urutan imgIds COCO (sorted)."""
    def __init__(self, E_A, E_B, image_ids):
        self.bA, self.bB = E.CocoBootstrap(E_A), E.CocoBootstrap(E_B)
        ids_sorted = list(E_A.params.imgIds)
        assert ids_sorted == list(E_B.params.imgIds)
        pos = {iid: j for j, iid in enumerate(ids_sorted)}
        self.perm = np.array([pos[i] for i in image_ids])          # cache i -> posisi COCO
        self.n = len(image_ids)

    @classmethod
    def from_boots(cls, bA, bB, ids_sorted, image_ids):
        """Dari dua CocoBootstrap yang sudah ada (urutan = ids_sorted = E.params.imgIds)."""
        r = cls.__new__(cls)
        r.bA, r.bB = bA, bB
        pos = {iid: j for j, iid in enumerate(ids_sorted)}
        r.perm = np.array([pos[i] for i in image_ids]); r.n = len(image_ids)
        return r

    def sorted_mask(self, use_a):
        m = np.zeros(self.n, bool); m[self.perm] = np.asarray(use_a, bool)
        return m

    def mix(self, use_a):
        return CocoMix(self.bA, self.bB, self.sorted_mask(use_a))

    def metrics(self, use_a, idx=None):
        return self.mix(use_a).metrics(np.arange(self.n) if idx is None else idx)

    def targets(self, conf_thr=0.25):
        """Δ per gambar (A − B), urutan cache. Juga jumlah GT kecil & semua (non-ignored)."""
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


def verify_mix(router, coco_gt, dets, image_ids, use_a, atol=1e-9):
    """Metrik CocoMix == COCOeval langsung pada jdict ter-route (bukti campuran evalImgs sah)."""
    st, _ = E.coco_eval(coco_gt, jdict(dets, image_ids, use_a))
    m = router.metrics(use_a)
    for k, kk in (("mAP50-95", "mAP50-95"), ("mAP50", "mAP50"), ("mAP_small", "mAP_small")):
        assert abs(m[k] - st[kk]) <= atol, f"CocoMix {k} {m[k]} != COCOeval {st[kk]}"
    return m


# =============================================================================
# ROUTING DARI SKOR, ACAK, KURVA
# =============================================================================
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


def curve(router, scores, rhos, n_random=20, seed=0):
    """DataFrame: per ρ dan per skor -> metrik COCO + selisih terhadap rata-rata acak (n_random undian)."""
    rows = []
    for rho in rhos:
        rnd = [router.metrics(m) for m in random_masks(router.n, rho, n_random, seed + int(rho * 1000))]
        base = {k: float(np.mean([r[k] for r in rnd])) for k in E.BOOT_METRICS}
        sd = {k: float(np.std([r[k] for r in rnd])) for k in E.BOOT_METRICS}
        rows.append(dict(rho=rho, routing="acak (rata-rata)", **base, **{f"sd_{k}": v for k, v in sd.items()}))
        for name, sc in scores.items():
            m = router.metrics(top_rho(sc, rho))
            rows.append(dict(rho=rho, routing=name, **m, **{f"gain_{k}": m[k] - base[k] for k in E.BOOT_METRICS}))
    return pd.DataFrame(rows)


def paired_bootstrap_routing(router, use_a, rho, n_random=5, n_boot=1000, seed=0):
    """Bootstrap berpasangan per gambar: routing − rata-rata acak (n_random undian), pada resample yang sama."""
    mix_a = router.mix(use_a)
    mix_r = [router.mix(m) for m in random_masks(router.n, rho, n_random, seed + 7)]
    return E.paired_bootstrap(mix_a, mix_r, n_boot=n_boot, seed=seed)


# =============================================================================
# GATE POST-HOC (detektor beku)
# =============================================================================
def candidate_models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    c = {}
    for a in (1.0, 10.0, 100.0, 1000.0):
        c[f"ridge_a{a:g}"] = ("reg", lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a)))
    for C in (0.001, 0.01, 0.1):
        c[f"logreg_C{C:g}"] = ("clf", lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))
    c["hgb"] = ("reg", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                              min_samples_leaf=40, l2_regularization=1.0, random_state=0))
    return c


def fit_score(kind, make, Xtr, ytr, X):
    m = make()
    if kind == "clf":
        yb = (ytr > 0).astype(int)
        if yb.min() == yb.max():
            return None, None
        m.fit(Xtr, yb)
        return m, m.decision_function(X)
    m.fit(Xtr, ytr)
    return m, m.predict(X)


def features(cache, with_z=True):
    X = cache["feat"]
    return np.concatenate([X, cache["z"][:, None]], 1) if with_z else X


def select_gate(cache_tr, tgt_tr, cache_va, router_va, rhos, n_random=20, seed=0, select_metric="mAP_small"):
    """
    Latih semua kandidat (model x target) di TRAIN, nilai di VAL: rata-rata gain metrik vs acak pada rhos.
    Aturan pemilihan TETAP: argmax rata-rata gain `select_metric` di val. Returns (tabel, nama terbaik, objek).
    """
    Xtr, Xva = features(cache_tr), features(cache_va)
    rnd = {rho: np.mean([router_va.metrics(m)[select_metric]
                         for m in random_masks(router_va.n, rho, n_random, seed + int(rho * 1000))]) for rho in rhos}
    rows, fitted = [], {}
    for tname in TARGETS:
        for mname, (kind, make) in candidate_models().items():
            m, s = fit_score(kind, make, Xtr, tgt_tr[tname], Xva)
            if m is None:
                continue
            gains = [router_va.metrics(top_rho(s, rho))[select_metric] - rnd[rho] for rho in rhos]
            key = f"{tname}|{mname}"
            fitted[key] = (kind, m)
            rows.append(dict(kandidat=key, target=tname, model=mname, mean_gain_val=float(np.mean(gains)),
                             **{f"gain_rho{rho:g}": g for rho, g in zip(rhos, gains)}))
    tab = pd.DataFrame(rows).sort_values("mean_gain_val", ascending=False).reset_index(drop=True)
    best = tab.kandidat.iloc[0]
    return tab, best, fitted[best]


def gate_score(fitted, cache):
    kind, m = fitted
    X = features(cache)
    return m.decision_function(X) if kind == "clf" else m.predict(X)
