"""
H6 pada fraksi campuran dari ukuran latensi lain (mis. L2) — offline dari intermediate suite, tanpa GPU.

    python scripts/h6_latency.py --config configs/kitti.yaml --suite-dir SUITE_INTERMEDIATE/kitti_seed42 \
        --l2-json PROFIL/l2.json --out /kaggle/working/h6_l2_kitti_seed42 [--ref SUITE/kitti_seed42/bootstrap_test.csv]

Akurasi (deteksi V, P2-Static, A, B dan keputusan router) diambil dari pass akurasi suite (job_dets.pkl), sehingga hanya
fraksi campuran f yang berubah: f = (T_router − T_V) / (T_P2S − T_V) dari latensi yang dipilih. Undian campuran dan
indeks bootstrap memakai seed yang sama dengan suite (seed + 19, analysis.boot_seed), sehingga "H6 utama (L1, ulang)"
harus identik dengan baris H6 utama di bootstrap_test.csv suite (pemeriksaan paritas, --ref).
Status di tesis: analisis sensitivitas pasca-hasil; H6 utama tetap L1.
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

from rfinal.cocomix import FastUnionBoot, UnionBoot, boot_engine, boot_tests, cluster_index, random_masks
from rfinal.env import load_config, provenance, write_json
from rfinal.suite import H6_UTAMA, router_of, spec_seed
from suite import suite_cfg

L1_LABEL = "H6 utama (L1, ulang)"
MIN_AGREE = 0.99            # kesepakatan keputusan router L2 vs pass akurasi suite (atau L1) minimum agar akurasi pass suite berlaku untuk L2


def f_l1(res):
    """f H6 utama dari job_res.json (sama dengan analyze: rata-rata build, latensi ujung-ke-ujung B2 efisien)."""
    LE = pd.DataFrame([dict(varian=k.split("|", 1)[1], mean=v["mean"]) for k, v in res["latency_e2e_builds"].items()])
    t = LE.groupby("varian")["mean"].mean().to_dict()
    return float(np.clip((t["Router B2 efisien (e2e)"] - t["Vanilla"]) / (t["P2-Static"] - t["Vanilla"]), 0, 1)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite-dir", required=True, help="folder berisi stageA.pkl, job_res.json, job_dets.pkl, gt_test.json")
    ap.add_argument("--l2-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="bootstrap_test.csv suite untuk paritas H6 utama")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args()
    cfg = suite_cfg(load_config(a.config), {})
    if a.n_boot:
        cfg["n_boot"] = a.n_boot
    D_ = Path(a.suite_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = pickle.loads((D_ / "stageA.pkl").read_bytes())
    res = json.loads((D_ / "job_res.json").read_text())
    J = pickle.loads((D_ / "job_dets.pkl").read_bytes())
    L2 = json.loads(Path(a.l2_json).read_text())
    te, rec, D = A["test"], J["rec"], J["dets"]
    assert rec["file"] == te["file"], "urutan gambar job != cache test"
    ids, n, seed = te["image_id"], len(te["image_id"]), int(spec_seed(A))
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(D_ / "gt_test.json"))
    R_AB, _, _ = router_of(gt, D["A"], D["B"], ids)
    R_VP, _, _ = router_of(gt, D["P2S"], D["V"], ids)
    U_AB, U_VP = UnionBoot(R_AB.bA, R_AB.bB), UnionBoot(R_VP.bA, R_VP.bB)
    d_dep = np.asarray(rec["use_g"], bool)

    f1, t1 = f_l1(res)
    # kesepakatan keputusan: utamakan vs keputusan pass akurasi suite (keputusan_l2.csv), selain itu vs L1 sesi profil
    kp = Path(a.l2_json).parent / "keputusan_l2.csv"
    K, agree_suite = None, {}
    if kp.exists():
        K = pd.read_csv(kp)
        pos = {Path(f_).name: j for j, f_ in enumerate(te["file"])}
        assert all(f_ in pos for f_ in K.file), "berkas keputusan_l2.csv tidak ada di cache test suite"
        ref_dec = d_dep[[pos[f_] for f_ in K.file]]
        agree_suite = {c: float(np.mean(K[c].astype(bool).to_numpy() == ref_dec)) for c in K.columns if c != "file"}
    fs, agree = {L1_LABEL: f1}, {L1_LABEL: 1.0}
    for conf, d in L2["per_conf"].items():
        lab = f"H6 L2 (NMS conf {conf})"
        fs[lab] = float(d["f_L2"])
        cols = [c for c in agree_suite if c.startswith("L2_") and c.endswith(f"_c{conf}")]
        agree[lab] = (min(agree_suite[c] for c in cols) if cols else float(d.get("kesepakatan_L2_L1", float("nan"))))
    mixes = {"router": (U_AB, R_AB.sorted_mask(d_dep))}
    tests = []
    for lab, f in fs.items():
        grp = []
        for j, m_ in enumerate(random_masks(n, f, cfg["n_draws"], seed + 19)):
            k = f"campuran|{lab}|{j}"
            mixes[k] = (U_VP, R_VP.sorted_mask(m_)); grp.append(k)
        tests.append((lab, "router", grp))
    point, samples = boot_engine(mixes, cfg["n_boot"], cfg["boot_seed"], cfg["boot_jobs"])
    BS = boot_tests(point, samples, tests)
    BS["f"] = BS.uji.map(fs)
    BS["bootstrap"] = "per gambar"
    tabs = [BS]
    if cfg.get("cluster_by") == "video_prefix":
        pos_ = {iid: j for j, iid in enumerate(sorted(ids))}
        grp_ = {}
        for f_, iid in zip(te["file"], ids):
            grp_.setdefault(f_.split("_")[0], []).append(pos_[iid])
        G_ = [np.asarray(v) for v in grp_.values()]
        fast_ = {}
        fmix = {}
        for k_, (U_, u_) in mixes.items():
            fast_.setdefault(id(U_), FastUnionBoot(U_)); fmix[k_] = (fast_[id(U_)], u_)
        _, S_cl = boot_engine(fmix, None, None, cfg["boot_jobs"], idx=cluster_index(G_, cfg["n_boot"], cfg["cluster_seed"]))
        BK = boot_tests(point, S_cl, tests)
        BK["f"] = BK.uji.map(fs); BK["bootstrap"] = "klaster video"
        tabs.append(BK)
    T = pd.concat(tabs, ignore_index=True)
    T["kesepakatan_keputusan"] = T.uji.map(agree)
    T["setara_L1"] = T["kesepakatan_keputusan"] >= MIN_AGREE          # aturan baca: < 0,99 -> bukan uji, hanya deskriptif
    T.to_csv(out / "h6_latensi.csv", index=False, float_format="%.6g")
    lines = [f"f: {json.dumps({k: round(v, 4) for k, v in fs.items()})}",
             "kesepakatan keputusan vs " + ("pass akurasi suite: " + json.dumps({k: round(v, 4) for k, v in agree_suite.items()})
                                           if agree_suite else "L1 sesi profil (keputusan_l2.csv tidak ada): "
                                           + json.dumps({k: round(v, 4) for k, v in agree.items()})),
             f"latensi L1 (ms): {json.dumps({k: round(v, 4) for k, v in t1.items() if k in ('Vanilla', 'P2-Static', 'Router B2 efisien (e2e)')})}"]
    for conf, d in L2["per_conf"].items():
        lines.append(f"latensi L2 conf {conf} (ms): " + json.dumps({k: round(v, 4) for k, v in d["T"].items()}))
    show = T[T.metric == "mAP_small"][["bootstrap", "uji", "f", "kesepakatan_keputusan", "diff", "ci_lo", "ci_hi", "p_two_sided"]]
    bad = [k for k, v in agree.items() if not v >= MIN_AGREE]
    if bad:
        lines.append(f"PERINGATAN: kesepakatan keputusan < {MIN_AGREE} untuk {bad} -> baris itu deskriptif, bukan uji")
    lines += ["", show.round(4).to_string(index=False)]
    par = None
    if a.ref:
        R = pd.read_csv(a.ref)
        r = R[R.uji == H6_UTAMA].set_index("metric")
        m = T[(T.uji == L1_LABEL) & (T.bootstrap == "per gambar")].set_index("metric")
        par = float(max(abs(m.loc[k, c] - r.loc[k, c]) for k in m.index for c in ("diff", "ci_lo", "ci_hi")))
        lines.append(f"\nParitas H6 utama (L1, ulang) vs suite: selisih maks {par:.2e} (harus 0 bila B dan seed sama)")
    txt = "\n".join(lines)
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[a.config, a.l2_json, *(str(D_ / x) for x in (
        "stageA.pkl", "job_res.json", "job_dets.pkl"))], extra=dict(f=fs, n_boot=cfg["n_boot"], paritas=par, kesepakatan=agree, kesepakatan_vs_suite=agree_suite)))


if __name__ == "__main__":
    main()
