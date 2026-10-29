"""
Pooling H6 L1/L2 antar-seed KITTI (indeks resample sama -> sah karena test sama), dari keluaran h6_latency.py:

    python scripts/h6_latency_pool.py --dirs H6/kitti_seed42 H6/kitti_seed7 H6/kitti_seed2026 --out H6/kitti_pool

Selisih rata-rata antar-model per resample, CI persentil 95% (= cocomix.pooled_tests).
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd

from rfinal.cocomix import _pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Z = {Path(d).name: np.load(Path(d) / "h6_boot.npz") for d in a.dirs}
    keys = sorted(set.intersection(*[{k for k in z.files if not k.startswith("titik|")} for z in Z.values()]))
    assert len({len(z[keys[0]]) for z in Z.values()}) == 1, "jumlah resample berbeda"
    rows = []
    for k in keys:
        uji, m = k.rsplit("|", 1)
        sm = np.mean(np.stack([z[k] for z in Z.values()]), 0); sm = sm[np.isfinite(sm)]
        lo, hi = np.percentile(sm, [2.5, 97.5])
        rows.append(dict(uji=uji, metric=m, diff_rata2=float(np.mean([float(z[f"titik|{k}"]) for z in Z.values()])),
                         ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm), n_model=len(Z)))
    P = pd.DataFrame(rows)
    P.to_csv(out / "h6_pool.csv", index=False, float_format="%.6g")
    txt = f"pooling {list(Z)}\n" + P[P.metric == "mAP_small"].round(4).to_string(index=False)
    print(txt); (out / "ringkasan.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
