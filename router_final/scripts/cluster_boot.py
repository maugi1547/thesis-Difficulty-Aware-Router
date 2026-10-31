"""
A2 — bootstrap KLASTER per rekaman (KITTI) dari intermediate suite, tanpa GPU. Pasca-hasil; dilaporkan di samping CI
per citra, tidak menggantikannya.

    # per model (KITTI: klaster = rekaman mentah dari pemetaan devkit, verifikasi B3)
    python scripts/cluster_boot.py run --config configs/kitti.yaml --suite-dir SUITE_INTER/kitti_seed42 \
        --map-csv kitti_peta_rekaman.csv --ref SUITE/kitti_seed42/bootstrap_test.csv --out A2/kitti_seed42
    # gabungan 3 seed KITTI (indeks resample sama untuk semua model -> sah karena data uji dan klaster sama)
    python scripts/cluster_boot.py pool --dirs A2/kitti_seed42 A2/kitti_seed7 A2/kitti_seed2026 --out A2/kitti_pool \
        [--ref-pool SUITE_GABUNG/ringkasan_pool.csv]
    # validasi kode klaster: VisDrone dengan klaster video harus mereproduksi bootstrap_klaster.csv suite persis
    python scripts/cluster_boot.py run --config configs/visdrone.yaml --suite-dir SUITE_INTER/visdrone_seed42 \
        --cluster video_prefix --ref SUITE/visdrone_seed42/bootstrap_test.csv \
        --ref-cluster SUITE/visdrone_seed42/bootstrap_klaster.csv --out A2/visdrone_seed42

Routing & pembanding direkonstruksi persis seperti rfinal.suite.analyze (keputusan pass akurasi rec["use_g"]; acak
beraktivasi sama: undian seed + 11; campuran V/P2S pada f ujung-ke-ujung B2 efisien: undian seed + 19; n_draws dari
config). Indeks bootstrap per citra = rng(analysis.boot_seed), sama dengan suite -> kolom per citra WAJIB identik dengan
bootstrap_test.csv (paritas). Indeks klaster = cluster_index(klaster, n_boot, --cluster-seed), default 20261017 (= VisDrone).

Aturan keputusan (daftar tindakan A2, ditulis sebelum dijalankan): laporkan CI klaster di samping CI per citra untuk H2 dan
H5. Bila CI klaster H2 gabungan KITTI memuat nol, klaim H2 KITTI diturunkan menjadi "konsisten dengan VisDrone"; bila H5
tetap memuat nol, status "tidak kalah" tidak berubah.
Penomoran paper/tesis 30 Okt: H5 = campuran berlatensi sama (label suite: "H6 utama").
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

from rfinal.cocomix import METRICS, FastUnionBoot, UnionBoot, _pval, boot_engine, boot_tests, cluster_index, random_masks
from rfinal.env import load_config, provenance, write_json
from rfinal.suite import _STATE, H6_UTAMA, router_of, spec_seed
from suite import suite_cfg

H2_SUITE = "H2/H7 | router (B2) − acak (aktivasi sama)"
# (label laporan, label suite, a, pembanding-prefix/list)
TESTS = [("H2 | router − acak (aktivasi sama)", H2_SUITE),
         ("H5 | router − campuran V/P2S (latensi ujung-ke-ujung)", H6_UTAMA),
         ("router − Branch A", "router (B2) − Branch A (P2 selalu)"),
         ("router − Branch B", "router (B2) − Branch B (P2 tidak pernah)"),
         ("Branch A − P2-Static", "Branch A − P2-Static"),
         ("Branch B − Vanilla", "Branch B − Vanilla")]
KEY = ("H2 | router − acak (aktivasi sama)", "H5 | router − campuran V/P2S (latensi ujung-ke-ujung)")


def f_e2e_efisien(res):
    """= lat['f_mix_e2e_efisien'] di analyze: rata-rata build latensi ujung-ke-ujung."""
    LE = pd.DataFrame([dict(varian=k.split("|", 1)[1], mean=v["mean"]) for k, v in res["latency_e2e_builds"].items()])
    t = LE.groupby("varian")["mean"].mean().to_dict()
    return float(np.clip((t["Router B2 efisien (e2e)"] - t["Vanilla"]) / (t["P2-Static"] - t["Vanilla"]), 0, 1))


def groups_of(te, ids, how, map_csv=None, key_col="image_id", cluster_col="rekaman"):
    """Posisi (urutan imgIds terurut, = indeks UnionBoot) per klaster + label klaster per gambar (urutan cache)."""
    if how == "video_prefix":
        lab = [Path(f).name.split("_")[0] for f in te["file"]]
    else:
        M = pd.read_csv(map_csv, dtype=str)
        mp = dict(zip(M[key_col], M[cluster_col]))
        stems = [Path(f).stem for f in te["file"]]
        miss = [s for s in stems if s not in mp]
        assert not miss, f"{len(miss)} gambar uji tidak ada di {map_csv}: {miss[:5]}"
        lab = [mp[s] for s in stems]
    pos = {iid: j for j, iid in enumerate(sorted(ids))}
    g = {}
    for l_, iid in zip(lab, ids):
        g.setdefault(l_, []).append(pos[iid])
    names = sorted(g)
    return [np.asarray(sorted(g[k])) for k in names], names, lab


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
    ids, n, seed, nd = te["image_id"], len(te["image_id"]), int(spec_seed(A)), cfg["n_draws"]
    _STATE["max_dets"] = tuple(A["spec"]["coco_max_dets"])
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(D_ / "gt_test.json"))
    R_AB, _, _ = router_of(gt, D["A"], D["B"], ids)
    R_VP, _, _ = router_of(gt, D["P2S"], D["V"], ids)
    U_AB, U_VP = UnionBoot(R_AB.bA, R_AB.bB), UnionBoot(R_VP.bA, R_VP.bB)
    F_AB, F_VP = FastUnionBoot(U_AB), FastUnionBoot(U_VP)
    d_dep = np.asarray(rec["use_g"], bool)
    f_mix = f_e2e_efisien(res)
    masks = {"A": ("AB", np.ones(n, bool)), "B": ("AB", np.zeros(n, bool)), "V": ("VP", np.zeros(n, bool)),
             "P2S": ("VP", np.ones(n, bool)), "router": ("AB", d_dep)}
    for j, m_ in enumerate(random_masks(n, int(d_dep.sum()) / n, nd, seed + 11)):
        masks[f"acak_router_{j}"] = ("AB", m_)
    for j, m_ in enumerate(random_masks(n, f_mix, nd, seed + 19)):
        masks[f"campuran_utama_{j}"] = ("VP", m_)
    R = {"AB": R_AB, "VP": R_VP}
    grp = lambda p_: [k for k in masks if k.startswith(p_)]
    comp = {"H2 | router − acak (aktivasi sama)": ("router", grp("acak_router_")),
            "H5 | router − campuran V/P2S (latensi ujung-ke-ujung)": ("router", grp("campuran_utama_")),
            "router − Branch A": ("router", ["A"]), "router − Branch B": ("router", ["B"]),
            "Branch A − P2-Static": ("A", ["P2S"]), "Branch B − Vanilla": ("B", ["V"])}
    tests = [(lab, *comp[lab]) for lab, _ in TESTS]
    slow = {k: ({"AB": U_AB, "VP": U_VP}[u], R[u].sorted_mask(m_)) for k, (u, m_) in masks.items()}
    quick = {k: ({"AB": F_AB, "VP": F_VP}[u], R[u].sorted_mask(m_)) for k, (u, m_) in masks.items()}

    print(f"[A2] {n} gambar, f campuran = {f_mix:.4f}, aktivasi = {d_dep.mean():.4f}; bootstrap per citra "
          f"({cfg['n_boot']} resample, seed {cfg['boot_seed']}) ...", flush=True)
    point, S_img = boot_engine(slow if a.slow else quick, cfg["n_boot"], cfg["boot_seed"], cfg["boot_jobs"])
    G, names, lab = groups_of(te, ids, a.cluster, a.map_csv, a.key_col, a.cluster_col)
    sizes = np.array([len(g) for g in G])
    kish = float(sizes.sum() ** 2 / (sizes ** 2).sum())
    print(f"[A2] {len(G)} klaster (terbesar {sizes.max()}, median {np.median(sizes):.0f}, n efektif Kish {kish:.1f}); "
          f"bootstrap klaster seed {a.cluster_seed} ...", flush=True)
    _, S_cl = boot_engine(quick, None, None, cfg["boot_jobs"], idx=cluster_index(G, cfg["n_boot"], a.cluster_seed))
    BI, BK = boot_tests(point, S_img, tests), boot_tests(point, S_cl, tests)
    big = int(np.argmax(sizes)); keep = np.concatenate([g for j_, g in enumerate(G) if j_ != big])
    pt_nb = {k: U.metrics(u, keep) for k, (U, u) in quick.items()}
    jk = [dict(uji=lb, metric=m, diff_tanpa_klaster_terbesar=pt_nb[x][m] - np.nanmean([pt_nb[b][m] for b in bs]))
          for lb, x, bs in tests for m in METRICS]
    T = BI[["uji", "metric", "diff", "ci_lo", "ci_hi", "p_two_sided"]].rename(
        columns={"ci_lo": "lo_gambar", "ci_hi": "hi_gambar", "p_two_sided": "p_gambar"}).merge(
        BK[["uji", "metric", "ci_lo", "ci_hi", "p_two_sided"]].rename(
            columns={"ci_lo": "lo_klaster", "ci_hi": "hi_klaster", "p_two_sided": "p_klaster"}), on=["uji", "metric"]).merge(
        pd.DataFrame(jk), on=["uji", "metric"])
    T["rasio_lebar"] = (T.hi_klaster - T.lo_klaster) / (T.hi_gambar - T.lo_gambar)
    for c in ("gambar", "klaster"):
        T[f"status_{c}"] = np.where(T[f"lo_{c}"] > 0, "CI > 0", np.where(T[f"hi_{c}"] < 0, "CI < 0", "memuat 0"))
    T["label_suite"] = T.uji.map(dict(TESTS))
    T.to_csv(out / "bootstrap_klaster_a2.csv", index=False, float_format="%.6g")

    npz = {}
    for lb, x, bs in tests:
        for m in METRICS:
            for nm, S in (("gambar", S_img), ("klaster", S_cl)):
                npz[f"{nm}|{lb}|{m}"] = (S[x][m] - np.nanmean(np.stack([S[b][m] for b in bs]), 0)).astype(np.float64)
            npz[f"titik|{lb}|{m}"] = np.float64(point[x][m] - np.nanmean([point[b][m] for b in bs]))
            npz[f"titik_tanpa_terbesar|{lb}|{m}"] = np.float64(pt_nb[x][m] - np.nanmean([pt_nb[b][m] for b in bs]))
    np.savez_compressed(out / "a2_boot.npz", **npz)
    pd.DataFrame(dict(klaster=names, n_gambar=sizes)).sort_values("n_gambar", ascending=False).to_csv(
        out / "klaster_ukuran.csv", index=False)
    info = dict(n_gambar=n, n_klaster=len(G), terbesar=dict(klaster=names[big], n=int(sizes[big])),
                median=float(np.median(sizes)), n_efektif_kish=kish, cara_klaster=a.cluster, cluster_seed=a.cluster_seed,
                n_boot=cfg["n_boot"], boot_seed=cfg["boot_seed"], n_draws=nd, f_campuran=f_mix,
                aktivasi=float(d_dep.mean()), sha_klaster=hashlib_ids(names, sizes))

    lines = [json.dumps(info, ensure_ascii=False)]
    par = {}
    if a.ref:
        Rf = pd.read_csv(a.ref)
        for lb, ls in TESTS:
            m_ = T[T.uji == lb].set_index("metric"); r_ = Rf[Rf.uji == ls].set_index("metric")
            if len(r_):
                par[lb] = float(max(abs(m_.loc[k, c] - r_.loc[k, c2]) for k in METRICS
                                    for c, c2 in (("diff", "diff"), ("lo_gambar", "ci_lo"), ("hi_gambar", "ci_hi"))))
        lines.append(f"Paritas per citra vs suite (selisih maks; harus 0): {json.dumps(par, ensure_ascii=False)}")
    par_cl = {}
    if a.ref_cluster:
        Rc = pd.read_csv(a.ref_cluster)
        for lb, ls in TESTS:
            m_ = T[T.uji == lb].set_index("metric"); r_ = Rc[Rc.uji == ls].set_index("metric")
            if len(r_):
                par_cl[lb] = float(max(abs(m_.loc[k, c] - r_.loc[k, c]) for k in METRICS for c in ("lo_klaster", "hi_klaster")))
        lines.append(f"Paritas klaster vs bootstrap_klaster.csv suite (selisih maks; harus 0): {json.dumps(par_cl, ensure_ascii=False)}")
    show = T[T.metric == "mAP_small"][["uji", "diff", "lo_gambar", "hi_gambar", "lo_klaster", "hi_klaster", "rasio_lebar",
                                      "status_gambar", "status_klaster", "diff_tanpa_klaster_terbesar"]]
    lines += ["", "mAP_small:", show.round(4).to_string(index=False)]
    txt = "\n".join(lines)
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    files = [a.config, *(str(D_ / x) for x in ("stageA.pkl", "job_res.json", "job_dets.pkl"))] + ([a.map_csv] if a.map_csv else [])
    write_json(out / "provenance.json", provenance(files=files, extra=dict(info=info, paritas=par, paritas_klaster=par_cl)))


def hashlib_ids(names, sizes):
    import hashlib
    return hashlib.sha256(json.dumps([[str(k), int(s)] for k, s in zip(names, sizes)]).encode()).hexdigest()[:16]


def cmd_pool(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Z = {str(Path(d)): np.load(Path(d) / "a2_boot.npz") for d in a.dirs}
    assert len(Z) == len(a.dirs), "folder ganda di --dirs"
    infos = [json.loads((Path(d) / "provenance.json").read_text())["info"] for d in a.dirs]
    assert len({i["sha_klaster"] for i in infos}) == 1 and len({i["cluster_seed"] for i in infos}) == 1, \
        "klaster / seed klaster berbeda antar-model -> indeks resample tidak sama, pooling tidak sah"
    rows = []
    for lb, ls in TESTS:
        for m in METRICS:
            row = dict(uji=lb, metric=m, label_suite=ls,
                       diff_rata2=float(np.mean([float(z[f"titik|{lb}|{m}"]) for z in Z.values()])),
                       diff_tanpa_klaster_terbesar=float(np.mean([float(z[f"titik_tanpa_terbesar|{lb}|{m}"]) for z in Z.values()])))
            for nm in ("gambar", "klaster"):
                sm = np.mean(np.stack([z[f"{nm}|{lb}|{m}"] for z in Z.values()]), 0); sm = sm[np.isfinite(sm)]
                lo, hi = np.percentile(sm, [2.5, 97.5])
                row.update({f"lo_{nm}": lo, f"hi_{nm}": hi, f"p_{nm}": _pval(sm),
                            f"status_{nm}": "CI > 0" if lo > 0 else ("CI < 0" if hi < 0 else "memuat 0")})
            row["rasio_lebar"] = (row["hi_klaster"] - row["lo_klaster"]) / (row["hi_gambar"] - row["lo_gambar"])
            rows.append(row)
    P = pd.DataFrame(rows)
    P.to_csv(out / "bootstrap_klaster_a2_pool.csv", index=False, float_format="%.6g")
    lines = [f"pooling {[Path(d).name for d in a.dirs]}; klaster {infos[0]['n_klaster']} (n efektif Kish "
             f"{infos[0]['n_efektif_kish']:.1f}), seed klaster {infos[0]['cluster_seed']}"]
    if a.ref_pool:
        Rp = pd.read_csv(a.ref_pool)
        par = {}
        for lb, ls in TESTS:
            m_ = P[P.uji == lb].set_index("metric"); r_ = Rp[Rp.uji == ls].set_index("metric")
            if len(r_):
                par[lb] = float(max(abs(m_.loc[k, c] - r_.loc[k, c2]) for k in METRICS
                                    for c, c2 in (("diff_rata2", "diff_rata2"), ("lo_gambar", "ci_lo"), ("hi_gambar", "ci_hi"))))
        lines.append(f"Paritas gabungan per citra vs ringkasan_pool.csv suite (harus 0): {json.dumps(par, ensure_ascii=False)}")
    h2 = P[(P.uji == KEY[0]) & (P.metric == "mAP_small")].iloc[0]
    h5 = P[(P.uji == KEY[1]) & (P.metric == "mAP_small")].iloc[0]
    lines += ["", "mAP_small:", P[P.metric == "mAP_small"][["uji", "diff_rata2", "lo_gambar", "hi_gambar", "lo_klaster",
                                                              "hi_klaster", "rasio_lebar", "status_klaster"]].round(4).to_string(index=False),
              "", "Aturan keputusan A2 (ditulis sebelum dijalankan):",
              f"  H2 gabungan KITTI, CI klaster [{h2.lo_klaster:+.4f}; {h2.hi_klaster:+.4f}] -> "
              + ("klaim H2 KITTI TETAP (CI klaster > 0)" if h2.lo_klaster > 0 else
                 "klaim H2 KITTI DITURUNKAN menjadi 'konsisten dengan VisDrone'"),
              f"  H5 gabungan KITTI, CI klaster [{h5.lo_klaster:+.4f}; {h5.hi_klaster:+.4f}] -> "
              + ("status 'tidak kalah' tidak berubah" if h5.lo_klaster <= 0 <= h5.hi_klaster else
                 ("CI klaster > 0: dilaporkan apa adanya, BUKAN pengganti kesimpulan utama" if h5.lo_klaster > 0 else
                  "CI klaster < 0: status berubah menjadi 'kalah' di bawah bootstrap klaster"))]
    txt = "\n".join(lines)
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--suite-dir", required=True); r.add_argument("--out", required=True)
    r.add_argument("--cluster", choices=("csv", "video_prefix"), default="csv")
    r.add_argument("--map-csv", default=None); r.add_argument("--key-col", default="image_id")
    r.add_argument("--cluster-col", default="rekaman")
    r.add_argument("--cluster-seed", type=int, default=20261017)
    r.add_argument("--ref", default=None, help="bootstrap_test.csv suite (paritas per citra)")
    r.add_argument("--ref-cluster", default=None, help="bootstrap_klaster.csv suite (paritas klaster, VisDrone)")
    r.add_argument("--n-boot", type=int, default=None)
    r.add_argument("--slow", action="store_true", help="UnionBoot (bukan FastUnionBoot) untuk bootstrap per citra")
    p = sub.add_parser("pool")
    p.add_argument("--dirs", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ref-pool", default=None, help="ringkasan_pool.csv suite gabungan (paritas per citra)")
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.cluster == "video_prefix" or a.map_csv, "--map-csv wajib untuk --cluster csv"
        cmd_run(a)
    else:
        cmd_pool(a)


if __name__ == "__main__":
    main()
