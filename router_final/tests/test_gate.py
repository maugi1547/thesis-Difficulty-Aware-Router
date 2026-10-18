import numpy as np
import torch

from rfinal.gate import GateHead, fit_gate, load_gate, save_gate, score_np


def test_fold_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 960)).astype(np.float32) * rng.uniform(0.1, 5, 960).astype(np.float32)
    delta = X[:, :5].sum(1) + rng.normal(size=400)
    g = fit_gate(X, delta)
    save_gate(tmp_path / "g.json", g, extra=dict(thr0=0.1))
    g2 = load_gate(tmp_path / "g.json")
    assert np.array_equal(g2["w"], g["w"]) and g2["b"] == g["b"]
    s_np = score_np(X, g2)
    head = GateHead(g2["w"], g2["b"])
    # Gemm FP32 (engine) vs numpy float64 (threshold val): beda hanya pembulatan
    s_t = (torch.from_numpy(X) @ head.w + head.b).numpy()
    assert np.allclose(s_t, s_np, atol=1e-4 * (1 + np.abs(s_np).max()))
