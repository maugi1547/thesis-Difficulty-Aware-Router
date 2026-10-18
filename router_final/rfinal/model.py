"""
Detektor dual-branch YOLOv8n-P2 — arsitektur final (router_final).

Satu graf Ultralytics stok (configs/yolov8n-dualbranch-p2.yaml) dengan DUA Detect head:
  - Detect_A (layer 34): P2, P3, P4, P5 — Branch A / WithP2.
  - Detect_B (layer 35): P3, P4, P5     — Branch B / NoP2.
Backbone + neck top-down (layer 0–15) dipakai bersama. Expert P2 (16–18) dan bottom-up Branch A (19–27) hanya
dibutuhkan Branch A; bottom-up Branch B (28–33) hanya dibutuhkan Branch B.

Tidak ada router, gate, atau loss tambahan di dalam model: saat training kedua branch selalu dihitung penuh dan
dilatih dengan loss deteksi saja (rfinal/loss.py). Routing per gambar dilakukan gate post-hoc (rfinal/gate.py)
di atas detektor yang sudah beku.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel

CFG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_YAML = CFG_DIR / "yolov8n-dualbranch-p2.yaml"

# Indeks layer (harus cocok dengan YAML; diperiksa di DualBranchDetector._check_layout)
L_P2_BACKBONE, L_P5, L_P4, L_P3 = 2, 9, 12, 15
TRUNK = range(0, 16)                 # Stage 1
BRANCH_A = range(16, 28)             # expert P2 (16–18) + bottom-up A (19–27)
BRANCH_B = range(28, 34)
L_DET_A, L_DET_B = 34, 35
FEATS_A = ("p3_neck", "p2_backbone", "p4_neck", "p5_sppf")
FEATS_B = ("p3_neck", "p4_neck", "p5_sppf")


def _run(layers, x, y):
    """Jalankan `layers` (urut) seperti BaseModel._predict_once; `y` = dict {indeks layer: output} (diisi)."""
    for m in layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y[m.i] = x
    return x


class DualBranchDetector(DetectionModel):
    """DetectionModel dengan dua Detect head. Jalur bawaan Ultralytics (val/predict) = Branch A."""

    def __init__(self, cfg=DEFAULT_YAML, ch=3, nc=None, verbose=True):
        super().__init__(cfg=str(cfg) if isinstance(cfg, Path) else cfg, ch=ch, nc=nc, verbose=verbose)
        self._check_layout()
        self._init_done = True

        # DetectionModel.__init__ hanya mengukur stride model[-1] (= Detect_B). Stride Detect_A diukur dengan cara
        # yang sama (forward dummy 256 px, Detect dalam mode training -> feature map mentah).
        s = 256
        self.model.eval()
        self.detect_A.training = self.detect_B.training = True
        with torch.no_grad():
            out_a, out_b = self.forward_dual(torch.zeros(1, ch, s, s))
        self.detect_A.stride = torch.tensor([s / x.shape[-2] for x in out_a])
        assert torch.equal(self.detect_B.stride, torch.tensor([s / x.shape[-2] for x in out_b]))
        self.detect_A.bias_init()     # Detect_B sudah di super().__init__()
        self.stride = self.detect_A.stride
        for d in (self.detect_A, self.detect_B):
            d.shape = None            # buang anchors cache dari forward dummy (dihitung ulang sesuai input)
        self.model.train()
        assert self.detect_A.stride.tolist() == [4.0, 8.0, 16.0, 32.0], self.detect_A.stride
        assert self.detect_B.stride.tolist() == [8.0, 16.0, 32.0], self.detect_B.stride

    # Properti, BUKAN atribut submodule: menyimpan modul sebagai atribut mendaftarkannya dua kali di state_dict
    # (alias "detect_A.*" di samping "model.34.*", seperti model lama).
    @property
    def detect_A(self):
        return self.model[L_DET_A]

    @property
    def detect_B(self):
        return self.model[L_DET_B]

    def _apply(self, fn):
        # BaseModel._apply hanya memindahkan stride/anchors/strides model[-1] (= Detect_B); Detect_A ikut di sini,
        # supaya .to(device) / .half() juga berlaku untuk anchors Detect_A (mis. saat final_eval memuat best.pt).
        self = super()._apply(fn)
        if getattr(self, "_init_done", False):
            m = self.model[L_DET_A]
            m.stride, m.anchors, m.strides = fn(m.stride), fn(m.anchors), fn(m.strides)
        return self

    def _check_layout(self):
        dets = [i for i, m in enumerate(self.model) if isinstance(m, Detect)]
        assert dets == [L_DET_A, L_DET_B], f"Detect harus di layer {L_DET_A} & {L_DET_B}, bukan {dets}"
        assert list(self.model[L_DET_A].f) == [18, 21, 24, 27], self.model[L_DET_A].f
        assert list(self.model[L_DET_B].f) == [L_P3, 30, 33], self.model[L_DET_B].f
        assert self.model[28].f == L_P3, "Branch B harus mulai dari P3_neck (layer 15)"

    # ------------------------------------------------------------------ forward
    def trunk(self, x):
        """Layer 0–15 -> (p3_neck, p2_backbone, p4_neck, p5_sppf)."""
        y = {}
        p3 = _run(self.model[TRUNK.start:TRUNK.stop], x, y)
        return p3, y[L_P2_BACKBONE], y[L_P4], y[L_P5]

    def branch_a(self, p3, p2, p4, p5):
        """Expert P2 + bottom-up A + Detect_A."""
        y = {L_P3: p3, L_P2_BACKBONE: p2, L_P4: p4, L_P5: p5}
        _run(self.model[BRANCH_A.start:BRANCH_A.stop], p3, y)
        return _run([self.model[L_DET_A]], None, y)

    def branch_b(self, p3, p4, p5):
        """Bottom-up B + Detect_B."""
        y = {L_P3: p3, L_P4: p4, L_P5: p5}
        _run(self.model[BRANCH_B.start:BRANCH_B.stop], p3, y)
        return _run([self.model[L_DET_B]], None, y)

    def forward_dual(self, x):
        """(out_A, out_B). Bentuk = bentuk keluaran Detect (training: list feature map; eval: (y, list))."""
        p3, p2, p4, p5 = self.trunk(x)
        return self.branch_a(p3, p2, p4, p5), self.branch_b(p3, p4, p5)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # Selama super().__init__() (layout belum diperiksa) -> jalur bawaan (output layer terakhir = Detect_B),
        # dipakai DetectionModel untuk mengukur stride Detect_B.
        if not getattr(self, "_init_done", False) or profile or visualize or embed is not None:
            return super()._predict_once(x, profile, visualize, embed)
        p3, p2, p4, p5 = self.trunk(x)
        if getattr(self, "eval_branch", "A") == "B":          # dipakai BranchBValidator (pemantauan training)
            return self.branch_b(p3, p4, p5)
        return self.branch_a(p3, p2, p4, p5)

    # ------------------------------------------------------------------ loss
    def loss(self, batch, preds=None):
        """`preds` diabaikan (validator hanya memberi output Branch A): kedua branch selalu dihitung ulang."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        return self.criterion(self.forward_dual(batch["img"]), batch)

    def init_criterion(self):
        from .loss import DualBranchLoss
        return DualBranchLoss(self)


# ---------------------------------------------------------------------- stage untuk ekspor / evaluasi
class Trunk(nn.Module):
    def __init__(self, model: DualBranchDetector):
        super().__init__(); self.m = model

    def forward(self, x):
        return self.m.trunk(x)


class BranchA(nn.Module):
    """(p3, p2, p4, p5) -> y_A (1, 4 + nc, 34000 @640) — keluaran Detect ter-decode (eval)."""
    def __init__(self, model: DualBranchDetector):
        super().__init__(); self.m = model

    def forward(self, p3, p2, p4, p5):
        out = self.m.branch_a(p3, p2, p4, p5)
        return out[0] if isinstance(out, tuple) else out


class BranchB(nn.Module):
    """(p3, p4, p5) -> y_B (1, 4 + nc, 8400 @640)."""
    def __init__(self, model: DualBranchDetector):
        super().__init__(); self.m = model

    def forward(self, p3, p4, p5):
        out = self.m.branch_b(p3, p4, p5)
        return out[0] if isinstance(out, tuple) else out


def pool4(p3, p2, p4, p5):
    """Fitur gate: [GAP‖GMP] dari p3_neck, p2_backbone, p4_neck, p5_sppf -> (B, 960) FP32 untuk YOLOv8n."""
    pool = lambda t: torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1)
    return torch.cat([pool(p3), pool(p2), pool(p4), pool(p5)], 1).float()


def load_detector(path, device="cpu", key=None):
    """
    Muat checkpoint DualBranchDetector (hasil scripts/train.py) -> model FP32, eval, tanpa gradien.
    key: 'ema' / 'model' / None (None = 'ema' bila ada, sama dengan yang dipakai Ultralytics saat val).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = key or ("ema" if ck.get("ema") is not None else "model")
    src = ck[key]
    nc = int(src.yaml["nc"])
    m = DualBranchDetector(cfg=src.yaml, nc=nc, verbose=False)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in src.state_dict().items()}, strict=True)
    m.names = getattr(src, "names", m.names)
    m.args = getattr(src, "args", None)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, dict(ckpt=str(path), weights_key=key, epoch=ck.get("epoch"), train_args=ck.get("train_args"))


def reset_anchor_cache(m):
    """Detect menyimpan anchors sebagai atribut biasa; panggil setelah .to(device) / .half() / deepcopy."""
    for mod in m.modules():
        if isinstance(mod, Detect):
            mod.shape = None
            mod.anchors, mod.strides = torch.empty(0), torch.empty(0)
    return m
