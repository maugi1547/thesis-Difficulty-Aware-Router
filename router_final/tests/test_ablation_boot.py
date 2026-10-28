"""boot_ablation: estimasi titik = selisih gain langsung dari Router (COCOeval eksak), set acak bersama per aktivasi."""
from types import SimpleNamespace

import numpy as np

from rfinal.ablation import F_LOSS, F_SS, boot_ablation
from rfinal.cocomix import Router, random_masks, top_rho
from rfinal.data import coco_eval, jdict
from test_cocomix import _synthetic


def test_boot_ablation_point_exact(tmp_path):
    gt, dA, dB, ids = _synthetic(tmp_path, n_img=60)
    _, EA = coco_eval(gt, jdict(dA, ids)); _, EB = coco_eval(gt, jdict(dB, ids))
    R = Router(EA, EB, ids); n = R.n
    sp = SimpleNamespace(R=R, n=n)
    rng = np.random.default_rng(1)
    m1, m2 = top_rho(rng.random(n), 0.5), top_rho(rng.random(n), 0.5)
    m3 = np.zeros(n, bool); m3[:20] = True                                   # aktivasi lain (seri B)
    masks = {("val", F_SS): m1, ("val", F_LOSS): m2, ("test", "B2 window 200 (final)"): m3,
             ("test", "threshold val tetap"): m1}
    T = boot_ablation({"val": sp, "test": sp}, masks, n_boot=30, seed=0, n_random=3, rand_seed=11, log=lambda *a: None)
    assert set(T.uji) == {"G1 soft_small − acak", "G1 soft_small − selisih loss", "G1 selisih loss − acak",
                          "B B2 window 200 − threshold val tetap", "B B2 window 200 − acak"}
    gain = lambda u: {k: R.metrics(u)[k] - np.mean([R.metrics(r)[k] for r in
                                                    random_masks(n, u.sum() / n, 3, 11 + int(u.sum()))])
                      for k in ("mAP50-95", "mAP50", "mAP_small")}
    g1, g2, g3 = gain(m1), gain(m2), gain(m3)
    for lab, ref in (("G1 soft_small − selisih loss", {k: g1[k] - g2[k] for k in g1}),
                     ("G1 soft_small − acak", g1),
                     ("B B2 window 200 − threshold val tetap", {k: g3[k] - g1[k] for k in g1})):
        for _, r in T[T.uji == lab].iterrows():
            assert abs(r["diff"] - ref[r.metric]) < 1e-12, (lab, r.metric, r["diff"], ref[r.metric])
    # pasangan top-ρ (aktivasi sama): set acak bersama -> selisih gain == selisih metrik
    d = T[(T.uji == "G1 soft_small − selisih loss") & (T.metric == "mAP_small")]["diff"].iloc[0]
    assert abs(d - (R.metrics(m1)["mAP_small"] - R.metrics(m2)["mAP_small"])) < 1e-12
    assert T.p_holm.between(0, 1).all() and (T.p_holm >= T.p_two_sided - 1e-15).all()


def test_boot_ablation_cluster_same_point(tmp_path):
    """Bootstrap klaster: estimasi titik identik dengan per gambar; video = token pertama nama file."""
    gt, dA, dB, ids = _synthetic(tmp_path, n_img=60)
    _, EA = coco_eval(gt, jdict(dA, ids)); _, EB = coco_eval(gt, jdict(dB, ids))
    R = Router(EA, EB, ids); n = R.n
    sp = SimpleNamespace(R=R, n=n, C={"file": [f"v{i % 7}_{i:05d}.jpg" for i in range(n)]})
    rng = np.random.default_rng(2)
    masks = {("val", F_SS): top_rho(rng.random(n), 0.5), ("val", F_LOSS): top_rho(rng.random(n), 0.5)}
    kw = dict(n_boot=40, n_random=3, rand_seed=11, log=lambda *a: None)
    T1 = boot_ablation({"val": sp}, masks, seed=0, **kw)
    T2 = boot_ablation({"val": sp}, masks, seed=None, cluster_seed=5, **kw)
    assert np.allclose(T1["diff"].values, T2["diff"].values, atol=1e-15)
    assert not np.allclose(T1["ci_lo"].values, T2["ci_lo"].values)
