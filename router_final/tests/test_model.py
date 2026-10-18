import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from rfinal.env import require_stock_ultralytics
from rfinal.model import BranchA, BranchB, DualBranchDetector, pool4


def _model():
    torch.manual_seed(0)
    m = DualBranchDetector(nc=2, verbose=False)
    m.args = get_cfg(DEFAULT_CFG)
    return m


def test_stock_ultralytics():
    require_stock_ultralytics()


def test_strides_and_state_dict_has_no_router():
    m = _model()
    assert m.detect_A.stride.tolist() == [4, 8, 16, 32] and m.detect_B.stride.tolist() == [8, 16, 32]
    keys = m.state_dict().keys()
    assert not [k for k in keys if not k.startswith("model.")], "alias / modul di luar model.* tidak boleh ada"
    banned = ("squeeze", "classifier", "proxy", "calib", "tau")
    assert not [k for k in keys if any(b in k for b in banned)]


def test_stages_equal_full_forward():
    m = _model().eval()
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        (yA, _), (yB, _) = m.forward_dual(x)
        p3, p2, p4, p5 = m.trunk(x)
        assert torch.equal(BranchA(m)(p3, p2, p4, p5), yA) and torch.equal(BranchB(m)(p3, p4, p5), yB)
        assert yA.shape == (1, 6, 34000) and yB.shape == (1, 6, 8400)
        assert pool4(p3, p2, p4, p5).shape == (1, 960)
        assert torch.equal(m(x)[0], yA)                       # jalur bawaan Ultralytics = Branch A
        m.eval_branch = "B"
        assert torch.equal(m(x)[0], yB)


def test_loss_six_items_and_backward():
    m = _model().train()
    b = dict(img=torch.rand(2, 3, 320, 320), batch_idx=torch.tensor([0., 1.]), cls=torch.tensor([[0.], [1.]]),
             bboxes=torch.tensor([[.5, .5, .2, .3], [.3, .4, .05, .05]]))
    total, items = m(b)
    assert total.ndim == 0 and items.shape == (6,)
    total.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad), "semua parameter harus menerima gradien (tidak ada modul mati)"


def test_init_deterministic():
    from rfinal.repro import state_hash
    assert state_hash(_model()) == state_hash(_model())
