"""
Bandingkan dua folder model hasil suite (mis. eksperimen/tahap4b.py di fork lama pada bobot M0 vs
rfinal.suite pada bobot M0 yang dipetakan tools/convert_m0.py). Cache, deteksi job, skor, keputusan, dan
bootstrap harus IDENTIK; yang boleh berbeda hanya kolom latensi (pengukuran waktu).

    python tools/compare_suite_runs.py OLD/kitti_s42 NEW/kitti_s42
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def same(a, b):
    if isinstance(a, list) and a and isinstance(a[0], np.ndarray):
        return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return np.array_equal(np.asarray(a), np.asarray(b))


def main(o, n):
    O, N = Path(o), Path(n)
    Ao, An = (pickle.loads((d / "stageA.pkl").read_bytes()) for d in (O, N))
    bad = []
    for part in ("test", "val"):
        for k in Ao[part]:
            if not same(Ao[part][k], An[part][k]):
                bad.append(f"{part}.{k}")
    for k in ("thr_val_gate", "thr_val_heur"):
        if Ao[k] != An[k]:
            bad.append(k)
    if not same(Ao["casc"]["w"], An["casc"]["w"]):
        bad.append("casc.w")
    Jo, Jn = (pickle.loads((d / "job_dets.pkl").read_bytes()) for d in (O, N))
    bad += [f"job.dets.{k}" for k in Jo["dets"] if not same(Jo["dets"][k], Jn["dets"][k])]
    bad += [f"job.rec.{k}" for k in Jo["rec"] if not same(Jo["rec"][k], Jn["rec"][k])]
    bo, bn = pd.read_csv(O / "bootstrap_test.csv"), pd.read_csv(N / "bootstrap_test.csv")
    m = bo.merge(bn, on=["uji", "metric"], suffixes=("_o", "_n"))
    lat_dep = m.uji.str.contains("campuran|kaskade")               # fraksi campuran ikut latensi terukur
    d = (m.diff_o - m.diff_n).abs() + (m.ci_lo_o - m.ci_lo_n).abs() + (m.ci_hi_o - m.ci_hi_n).abs()
    bad += [f"bootstrap: {u}" for u in m.uji[(d > 0) & ~lat_dep].unique()]
    for f in ("per_gambar_test.csv", "h3_perilaku_skor.csv", "cek_fp32.csv", "b2_urutan.csv"):
        a, b = pd.read_csv(O / f), pd.read_csv(N / f)
        num = a.select_dtypes("number").columns.intersection(b.columns)
        if f != "cek_fp32.csv" and float((a[num] - b[num]).abs().max().max()) > 0:
            bad.append(f)
    print("gate info baru:", An["gate"]["info"])
    print("GFLOPs sama:", all(abs(Ao["gflops"][k] - An["gflops"][k]) < 1e-9 for k in Ao["gflops"]))
    print("❌ BERBEDA: " + ", ".join(bad) if bad else "✅ IDENTIK (selain latensi)")
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
