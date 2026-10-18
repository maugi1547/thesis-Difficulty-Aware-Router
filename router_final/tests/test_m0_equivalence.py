"""
Graf final == jaringan deteksi M0 (bobot M0 dipetakan lewat rfinal.compat).

Butuh berkas acuan dari tools/m0_reference.py (dijalankan dengan fork lama):
    RFINAL_M0_REF=/path/m0_ref.pt pytest tests/test_m0_equivalence.py
Dilewati bila variabel itu tidak diisi.
"""
import os

import pytest
import torch

from rfinal.compat import map_m0_state_dict
from rfinal.loss import BranchLoss
from rfinal.model import BranchA, BranchB, DualBranchDetector

REF = os.environ.get("RFINAL_M0_REF")
pytestmark = pytest.mark.skipif(not REF, reason="RFINAL_M0_REF tidak diisi")


@pytest.fixture(scope="module")
def ref_and_model():
    ref = torch.load(REF, map_location="cpu", weights_only=False)
    sd, dropped = map_m0_state_dict(ref["state_dict"])
    m = DualBranchDetector(nc=2, verbose=False)
    m.load_state_dict(sd, strict=True)                    # strict: tidak ada bobot final yang tertinggal
    assert dropped and all(k.startswith("model.16.") for k in dropped)
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    m.args = get_cfg(DEFAULT_CFG)
    return ref, m.eval()


def test_inference_identical(ref_and_model):
    ref, m = ref_and_model
    a, b = BranchA(m), BranchB(m)
    with torch.no_grad():
        for r in ref["eval"]:
            p3, p2, p4, p5 = m.trunk(r["x"])
            for got, want in ((p3, r["p3"]), (p2, r["p2"]), (p4, r["p4"]), (p5, r["p5"])):
                assert torch.equal(got, want)
            assert torch.equal(a(p3, p2, p4, p5), r["yA"]), "Branch A berbeda dari M0"
            assert torch.equal(b(p3, p4, p5), r["yB"]), "Branch B berbeda dari M0"


def test_detection_loss_identical(ref_and_model):
    """Item loss deteksi stok (box, cls, dfl) == item loss lama [0:3] per branch, pada keluaran Detect yang sama."""
    ref, m = ref_and_model
    L = ref["loss"]
    for head, det, want in ((m.detect_A, L["det_A"], L["items_A"]), (m.detect_B, L["det_B"], L["items_B"])):
        _, items = BranchLoss(m, head)(det, L["batch"])
        assert torch.equal(items, want), f"loss {items} != lama {want}"
