"""
Analisis rfinal.suite == Tahap 4/4B (eksperimen/tahap4_final.py) pada intermediate Kaggle yang sama.

    RFINAL_SUITE_REF=/path/visdrone_s42 pytest tests/test_suite_replay.py
Folder acuan berisi stageA.pkl, job_res.json, job_dets.pkl, gt_test.json (dari *_intermediate.zip Tahap 4) DAN
boot_point.json, boot_samples.npz, per_gambar_test.csv, b2_urutan.csv, h3_perilaku_skor.csv (dari zip hasil).
Bootstrap dibatasi K resample pertama (indeks sama dengan Kaggle) supaya cepat; harus identik per resample.
Dilewati bila variabel tidak diisi.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REF = os.environ.get("RFINAL_SUITE_REF")
K = int(os.environ.get("RFINAL_SUITE_K", "8"))
pytestmark = pytest.mark.skipif(not REF, reason="RFINAL_SUITE_REF tidak diisi")


def test_analyze_replays_tahap4(tmp_path):
    from rfinal import cocomix as C
    from rfinal import suite as S
    D = Path(REF)
    A = pickle.loads((D / "stageA.pkl").read_bytes()); A["gt_test"] = str(D / "gt_test.json")
    res = json.loads((D / "job_res.json").read_text()); J = pickle.loads((D / "job_dets.pkl").read_bytes())
    orig = C.boot_engine

    def head_only(mixes, n_boot, seed, n_jobs=1, idx=None):
        I = next(iter(mixes.values()))[0].I
        return orig(mixes, K, None, n_jobs, idx=np.random.default_rng(seed).integers(0, I, (n_boot, I))[:K])
    S.boot_engine = head_only
    try:
        cfg = dict(rho=0.5, n_draws=5, n_draws_sekunder=3, b2_window=200, b2_min_hist=20, b2_orders=5, fp32_check=True,
                   n_boot=1000, boot_seed=20261016, boot_jobs=1, h5_ms=0.15, h7_dev=0.05)
        S.analyze(A, res, J, cfg, tmp_path, log=lambda *a: None)
    finally:
        S.boot_engine = orig
    pn, pr = (json.loads((d / "boot_point.json").read_text()) for d in (tmp_path, D))
    common = set(pn) & set(pr)
    assert len(common) >= 20
    assert max(abs(pn[k][m] - pr[k][m]) for k in common for m in C.METRICS) == 0.0
    zn, zr = np.load(tmp_path / "boot_samples.npz"), np.load(D / "boot_samples.npz")
    assert max(float(np.abs(zn[k][:K] - zr[k][:K]).max()) for k in zn.files if k in zr.files) == 0.0
    for f in ("per_gambar_test.csv", "b2_urutan.csv", "h3_perilaku_skor.csv"):
        a, r = pd.read_csv(tmp_path / f), pd.read_csv(D / f)
        num = a.select_dtypes("number").columns.intersection(r.columns)
        assert float((a[num] - r[num]).abs().max().max()) < 1e-12, f
