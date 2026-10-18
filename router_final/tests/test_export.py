import numpy as np
import torch

from rfinal.export import export_a3, verify_a3
from rfinal.model import DualBranchDetector


def test_a3_onnx_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    m = DualBranchDetector(nc=2, verbose=False).eval()
    rng = np.random.default_rng(0)
    gate = dict(w=rng.normal(0, 0.05, 960), b=0.1)
    imgs = [torch.rand(1, 3, 640, 640) for _ in range(2)]
    paths, mods = export_a3(m, gate, imgs[0], tmp_path, "fp32", "cpu")
    worst = verify_a3(paths["fp32"]["a3"], mods, imgs)
    assert worst["y"] < 1e-4 and worst["score"] < 1e-4
