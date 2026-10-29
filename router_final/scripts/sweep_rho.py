"""
Sensitivitas titik operasi ρ (pasca-hasil; ρ = 0,5 TETAP terkunci) — offline dari cache, tanpa GPU.

    # per model: test dari intermediate suite (+ val dari cache ablasi, opsional)
    python scripts/sweep_rho.py run --config configs/kitti.yaml --suite-dir SUITE_INTER/kitti_seed42 \
        --ablation-cache RUN/ablasi/cache_ablasi.pkl --out SWEEP/kitti_seed42 [--ref SUITE/kitti_seed42/bootstrap_test.csv]
    # pooling KITTI 3 seed (indeks resample sama -> sah, test sama)
    python scripts/sweep_rho.py pool --dirs SWEEP/kitti_seed42 SWEEP/kitti_seed7 SWEEP/kitti_seed2026 --out SWEEP/kitti_pool

Untuk tiap ρ di grid:
  VAL (deskriptif, hanya model yang punya cache ablasi): top-ρ skor gate final; gain = metrik − rata-rata 20 routing
      acak beraktivasi sama, plus plafon oracle (top-ρ Δsoft_small GT). Menunjukkan bentuk kurva, BUKAN untuk memilih ρ.
  TEST (konfirmasi): sistem deployment persis — B2 efisien (window, min_hist dari config) pada aliran test dengan skor gate
      TensorRT dari pass akurasi suite (rec["score_g"]), thr0 = kuantil (1 − ρ) skor val. Uji (bootstrap berpasangan,
      indeks resample = suite):
        H2/H7(ρ): router − acak A/B beraktivasi sama (undian seed + 11, sama dengan suite)
        H6(ρ):    router − campuran acak Vanilla/P2-Static pada f(ρ) (undian seed + 19)
      f(ρ) dari MODEL latensi linear: T_router(a) = a·T_A + (1 − a)·T_B + h, dengan T_A, T_B = rute paksa A3 (ujung-ke-ujung,
      rata-rata build) dan h = overhead host/keputusan yang dikalibrasi dari titik terukur
      (h = T_router_B2efisien − [a0·T_A + (1 − a0)·T_B], a0 = aktivasi pass ujung-ke-ujung). Asumsi: h tidak bergantung ρ
      (biaya B2 = kuantil jendela 200, tidak bergantung ρ). Di ρ = 0,5 model ini mereproduksi f H6 utama persis, sehingga
      baris ρ = 0,5 harus identik dengan suite (paritas, --ref; di ρ terkunci f terukur dipakai langsung, karena aktivasi
      pass ujung-ke-ujung dan pass akurasi bisa berbeda ±0,005). Titik ρ lain = perkiraan, bukan latensi terukur.
VisDrone (analysis.cluster_by = video_prefix): bootstrap klaster per video juga dihitung.
"""
import argparse
import contextlib
import io
import json
import pickle
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd

from rfinal.budget import B2Controller, thr0_from_val
from rfinal.cocomix import (METRICS, FastUnionBoot, UnionBoot, _pval, boot_engine, boot_tests, cluster_index,
                            random_masks, top_rho)
from rfinal.env import load_config, provenance, write_json
from rfinal.gate import score_np
from rfinal.suite import _STATE, H6_UTAMA, router_of, spec_seed
from suite import suite_cfg

GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H2_SUITE = "H2/H7 | router (B2) − acak (aktivasi sama)"
SHOW = ("mAP_small", "mAP50", "mAP50-95")


def lat_model(res):
    LE = pd.DataFrame([dict(varian=k.split("|", 1)[1], mean=v["mean"]) for k, v in res["latency_e2e_builds"].items()])
    t = LE.groupby("varian")["mean"].mean().to_dict()
    a0 = float(np.mean([v for k, v in res["e2e_activation"].items() if k.endswith("efisien (e2e)")]))
    TV, TP, TA, TB, TR = (t[k] for k in ("Vanilla", "P2-Static", "A3 gate rute A", "A3 gate rute B",
                                         "Router B2 efisien (e2e)"))
    h = TR - (a0 * TA + (1 - a0) * TB)
    T = lambda a: a * TA + (1 - a) * TB + h
    f = lambda a: float(np.clip((T(a) - TV) / (TP - TV), 0, 1))
    return dict(TV=TV, TP=TP, TA=TA, TB=TB, TR=TR, a0=a0, h=h), T, f


def b2_stream(scores, rho, window, min_hist, thr0):
    c = B2Controller(rho, window, min_hist, thr0, efficient=True)
    return np.array([c.decide(s)[0] for s in scores], bool)


def val_curve(cache, gate, grid, n_random, seed):
    C = pickle.loads(Path(cache).read_bytes())
    va = C["va"]
    s = score_np(va.X, gate)
    rows = []
    for rho in grid:
        use = top_rho(s, rho)
        g = va.gain(use, n_random, seed + 3)
        o = va.gain(top_rho(va.T["soft_small"], rho), n_random, seed + 3)
        rows.append(dict(rho=rho, aktivasi=float(use.mean()), **g, **{f"plafon_{k}": v for k, v in o.items()}))
    return pd.DataFrame(rows)


def cmd_run(a):
    cfg = suite_cfg(load_config(a.config), {})
    if a.n_boot:
        cfg["n_boot"] = a.n_boot
    D_ = Path(a.suite_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = pickle.loads((D_ / "stageA.pkl").read_bytes())
    res = json.loads((D_ / "job_res.json").read_text())
    J = pickle.loads((D_ / "job_dets.pkl").read_bytes())
    te, rec, D = A["test"], J["rec"], J["dets"]
    assert rec["file"] == te["file"], "urutan gambar job != cache test"
    ids, n, seed = te["image_id"], len(te["image_id"]), int(spec_seed(A))
    grid = sorted(set(a.grid) | {cfg["rho"]})
    W, MH = cfg["b2_window"], cfg["b2_min_hist"]
    gate = dict(w=np.asarray(A["gate"]["w"], np.float64), b=float(A["gate"]["b"]))
    s_val = np.asarray(A["val"]["score_g32"], np.float64)
    s_te = np.asarray(rec["score_g"], np.float32)

    # paritas keputusan: ρ terkunci harus mereproduksi keputusan pass akurasi suite persis
    d_dep = np.asarray(rec["use_g"], bool)
    d_chk = b2_stream(s_te, cfg["rho"], W, MH, A["thr_val_gate"])
    par_dec = float((d_chk == d_dep).mean())
    assert par_dec == 1.0, f"keputusan B2 ulang != suite ({par_dec:.4f})"

    L, T, f = lat_model(res)
    _STATE["max_dets"] = tuple(A["spec"]["coco_max_dets"])          # = analyze suite (VisDrone: 500)
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(D_ / "gt_test.json"))
    R_AB, _, _ = router_of(gt, D["A"], D["B"], ids)
    R_VP, _, _ = router_of(gt, D["P2S"], D["V"], ids)
    U_AB, U_VP = UnionBoot(R_AB.bA, R_AB.bB), UnionBoot(R_VP.bA, R_VP.bB)
    mixes = {"A": (U_AB, R_AB.sorted_mask(np.ones(n, bool))), "B": (U_AB, R_AB.sorted_mask(np.zeros(n, bool))),
             "V": (U_VP, R_VP.sorted_mask(np.zeros(n, bool))), "P2S": (U_VP, R_VP.sorted_mask(np.ones(n, bool)))}
    tests, info = [], []
    for rho in grid:
        thr0 = A["thr_val_gate"] if rho == cfg["rho"] else thr0_from_val(s_val, rho)
        d = d_dep if rho == cfg["rho"] else b2_stream(s_te, rho, W, MH, thr0)
        act = float(d.mean())
        a_lat = L["a0"] if rho == cfg["rho"] else act          # ρ terkunci: latensi & f terukur persis (= H6 utama suite)
        fr = f(a_lat)
        r = f"router|{rho}"
        mixes[r] = (U_AB, R_AB.sorted_mask(d))
        ga = [f"acak|{rho}|{j}" for j in range(cfg["n_draws"])]
        for k, m_ in zip(ga, random_masks(n, int(d.sum()) / n, cfg["n_draws"], seed + 11)):
            mixes[k] = (U_AB, R_AB.sorted_mask(m_))
        gc = [f"campuran|{rho}|{j}" for j in range(cfg["n_draws"])]
        for k, m_ in zip(gc, random_masks(n, fr, cfg["n_draws"], seed + 19)):
            mixes[k] = (U_VP, R_VP.sorted_mask(m_))
        tests += [(f"H2(ρ)|{rho}", r, ga), (f"H6(ρ)|{rho}", r, gc), (f"router − P2-Static|{rho}", r, ["P2S"]),
                  (f"router − Vanilla|{rho}", r, ["V"])]
        info.append(dict(rho=rho, aktivasi_test=act, thr0_val=float(thr0), T_router_model_ms=T(a_lat), f=fr,
                         hemat_vs_P2S_persen=100 * (1 - T(a_lat) / L["TP"]),
                         tambah_vs_V_persen=100 * (T(a_lat) / L["TV"] - 1)))
    fast_ = {}
    fmix = {}
    for k_, (U_, u_) in mixes.items():
        fast_.setdefault(id(U_), FastUnionBoot(U_)); fmix[k_] = (fast_[id(U_)], u_)
    print(f"bootstrap: {len(mixes)} routing × {cfg['n_boot']} resample ({'cepat' if a.fast else 'UnionBoot'}) ...", flush=True)
    point, samples = boot_engine(fmix if a.fast else mixes, cfg["n_boot"], cfg["boot_seed"], cfg["boot_jobs"])
    boots = {"per gambar": (point, samples)}
    if cfg.get("cluster_by") == "video_prefix":
        pos_ = {iid: j for j, iid in enumerate(sorted(ids))}
        grp_ = {}
        for f_, iid in zip(te["file"], ids):
            grp_.setdefault(f_.split("_")[0], []).append(pos_[iid])
        _, S_cl = boot_engine(fmix, None, None, cfg["boot_jobs"],
                              idx=cluster_index([np.asarray(v) for v in grp_.values()], cfg["n_boot"], cfg["cluster_seed"]))
        boots["klaster video"] = (point, S_cl)
    tabs, npz = [], {}
    for bname, (pt, sm) in boots.items():
        BS = boot_tests(pt, sm, tests)
        BS[["uji", "rho"]] = BS.uji.str.split("|", expand=True); BS["rho"] = BS.rho.astype(float)
        BS["bootstrap"] = bname
        tabs.append(BS)
        for lab, a_, bs in tests:                          # sampel selisih untuk pooling antar-seed
            for m in METRICS:
                npz[f"{bname}|{lab}|{m}"] = (sm[a_][m] - np.nanmean(np.stack([sm[b][m] for b in bs]), 0)).astype(np.float32)
                npz[f"titik|{lab}|{m}"] = np.float64(pt[a_][m] - np.nanmean([pt[b][m] for b in bs]))
    TT = pd.concat(tabs, ignore_index=True).merge(pd.DataFrame(info), on="rho")
    TT.to_csv(out / "sweep_test.csv", index=False, float_format="%.6g")
    np.savez_compressed(out / "sweep_boot.npz", **npz)
    pd.DataFrame(info).to_csv(out / "sweep_latensi.csv", index=False, float_format="%.6g")
    lines = [f"model latensi (ms): {json.dumps({k: round(v, 4) for k, v in L.items()})}",
             f"paritas keputusan B2 ρ={cfg['rho']} vs suite: {par_dec:.4f}", "",
             pd.DataFrame(info).round(4).to_string(index=False), ""]
    for bname in boots:
        sub = TT[(TT.bootstrap == bname) & TT.metric.isin(SHOW)]
        piv = sub.assign(sel=lambda x: x["diff"].map("{:+.4f}".format) + " [" + x.ci_lo.map("{:+.4f}".format) + ", "
                         + x.ci_hi.map("{:+.4f}".format) + "]").pivot_table(index=["uji", "metric"], columns="rho",
                                                                              values="sel", aggfunc="first")
        lines += [f"TEST, bootstrap {bname} (selisih [CI 95%]):", piv.to_string(), ""]
    par = None
    if a.ref:
        R = pd.read_csv(a.ref)
        r0 = TT[(TT.rho == cfg["rho"]) & (TT.bootstrap == "per gambar")]
        par = {}
        for mine, theirs in (("H6(ρ)", H6_UTAMA), ("H2(ρ)", H2_SUITE)):
            m_ = r0[r0.uji == mine].set_index("metric"); t_ = R[R.uji == theirs].set_index("metric")
            par[mine] = float(max(abs(m_.loc[k, c] - t_.loc[k, c]) for k in m_.index for c in ("diff", "ci_lo", "ci_hi")))
        lines.append(f"Paritas baris ρ = {cfg['rho']} vs suite (selisih maks; harus 0): {json.dumps(par)}")
    if a.ablation_cache:
        V = val_curve(a.ablation_cache, gate, grid, 20, 0)
        V.to_csv(out / "sweep_val.csv", index=False, float_format="%.6g")
        lines += ["", "VAL (deskriptif; gain vs 20 routing acak, top-ρ skor gate final; plafon = oracle Δsoft_small GT):",
                  V[["rho", "aktivasi"] + [f"gain_{k}" for k in SHOW] + [f"plafon_gain_{k}" for k in SHOW]].round(4)
                  .to_string(index=False)]
    txt = "\n".join(lines)
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[a.config, *(str(D_ / x) for x in ("stageA.pkl", "job_res.json",
                                                                                              "job_dets.pkl"))],
                                                   extra=dict(grid=grid, n_boot=cfg["n_boot"], n_draws=cfg["n_draws"],
                                                              paritas=par, paritas_keputusan=par_dec, model_latensi=L)))


def cmd_pool(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Z = {Path(d).name: np.load(Path(d) / "sweep_boot.npz") for d in a.dirs}
    keys = set.intersection(*[{k for k in z.files if k.startswith("per gambar|")} for z in Z.values()])
    n = {len(z[next(iter(keys))]) for z in Z.values()}
    assert len(n) == 1, "jumlah resample berbeda"
    rows = []
    for k in sorted(keys):
        _, uji, rho, m = k.split("|")
        lab = f"{uji}|{rho}"
        sm = np.mean(np.stack([z[k] for z in Z.values()]), 0); sm = sm[np.isfinite(sm)]
        pt = float(np.mean([float(z[f"titik|{lab}|{m}"]) for z in Z.values()]))
        lo, hi = np.percentile(sm, [2.5, 97.5])
        rows.append(dict(uji=uji, rho=float(rho), metric=m, diff_rata2=pt, ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm),
                         n_model=len(Z)))
    P = pd.DataFrame(rows)
    P.to_csv(out / "sweep_pool.csv", index=False, float_format="%.6g")
    sub = P[P.metric.isin(SHOW)]
    piv = sub.assign(sel=lambda x: x.diff_rata2.map("{:+.4f}".format) + " [" + x.ci_lo.map("{:+.4f}".format) + ", "
                     + x.ci_hi.map("{:+.4f}".format) + "]").pivot_table(index=["uji", "metric"], columns="rho",
                                                                          values="sel", aggfunc="first")
    txt = f"pooling {list(Z)} (rata-rata selisih per resample)\n" + piv.to_string()
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--suite-dir", required=True)
    r.add_argument("--out", required=True); r.add_argument("--ablation-cache", default=None)
    r.add_argument("--ref", default=None, help="bootstrap_test.csv suite (paritas baris ρ terkunci)")
    r.add_argument("--grid", type=float, nargs="+", default=GRID)
    r.add_argument("--n-boot", type=int, default=None)
    r.add_argument("--fast", action="store_true", help="FastUnionBoot juga untuk bootstrap per gambar (VisDrone)")
    p = sub.add_parser("pool")
    p.add_argument("--dirs", nargs="+", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_pool(a)


if __name__ == "__main__":
    main()
