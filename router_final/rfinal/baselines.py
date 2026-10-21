"""
Pembanding sistem final (router_final), semuanya di lingkungan Ultralytics stok:
  - Vanilla (yolov8n) dan P2-Static (yolov8n-p2): model stok, dimuat apa adanya dari last.pt;
  - heuristik kepadatan objek kecil tanpa label (head P3 Detect_B di Stage 1);
  - fitur trunk Vanilla untuk gate kaskade Vanilla -> P2-Static.
Diporting tanpa perubahan semantik dari eksperimen/tahap4b.py (P3Density, S1Heur, static_with_feats) dan
tahap2_eval.py (load_static, _StaticOut). Indeks Detect_B: 33 (graf lama) -> 35 (graf final).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import SMALL_PX
from .model import L_DET_B, pool4

HEUR_CONF = 0.25         # ambang skor kelas maks anchor P3 (dikunci Tahap 4)
HEUR_EPS = 0.01          # skor = log(1 + n_kecil) + HEUR_EPS · log(1 + n_semua)


def load_static(path, device="cpu", nc=None):
    """Vanilla / P2-Static (checkpoint Ultralytics) -> DetectionModel FP32 eval, tanpa gradien."""
    from ultralytics import YOLO
    p = Path(path)
    assert p.exists(), f"checkpoint baseline tidak ada: {p}"
    m = YOLO(str(p)).model.float().to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    if nc is not None:
        assert int(m.model[-1].nc) == int(nc), f"{p}: nc {m.model[-1].nc} != dataset {nc} (baseline dataset lain?)"
    return m


class StaticOut(nn.Module):
    """Model statis -> y (1, 4 + nc, N) ter-decode."""

    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def static_with_feats(m, x):
    """Vanilla/P2-Static -> (y, (p3 = layer 15, p2 = 2, p4 = 12, p5 = 9)); fitur dipakai gate kaskade."""
    y = []
    for mod in m.model:
        if mod.f != -1:
            x = y[mod.f] if isinstance(mod.f, int) else [x if j == -1 else y[j] for j in mod.f]
        x = mod(x)
        y.append(x)
    out = x[0] if isinstance(x, (tuple, list)) else x
    return out, (y[15], y[2], y[12], y[9])


class P3Density(nn.Module):
    """
    Heuristik kepadatan objek kecil TANPA label & TANPA fit, dari head P3 Detect_B (stride 8) di Stage 1:
    n_semua = anchor P3 dengan skor kelas maks ≥ HEUR_CONF; n_kecil = yang luas kotak decode < small_area
    (piksel letterbox² = 32² · gain_h · gain_w). skor = log(1 + n_kecil) + HEUR_EPS · log(1 + n_semua).
    """

    def __init__(self, det, conf=HEUR_CONF, eps=HEUR_EPS):
        super().__init__()
        self.cv2, self.cv3, self.dfl = det.cv2[0], det.cv3[0], det.dfl
        self.reg_max, self.stride = int(det.reg_max), float(det.stride[0])
        assert self.stride == 8.0, f"head P3 Detect_B harus stride 8, bukan {self.stride}"
        self.logit_thr, self.eps = float(math.log(conf / (1 - conf))), float(eps)

    def counts(self, p3, small_area):
        box, cls = self.cv2(p3), self.cv3(p3)
        b, _, h, w = cls.shape
        hit = cls.amax(1).reshape(b, h * w) > self.logit_thr
        d = self.dfl(box.reshape(b, 4 * self.reg_max, h * w)).float() * self.stride
        area = (d[:, 0] + d[:, 2]) * (d[:, 1] + d[:, 3])
        small = area < small_area.float().reshape(-1, 1)
        return hit.float().sum(1), (hit & small).float().sum(1)

    def combine(self, n_all, n_small):
        return torch.log(1.0 + n_small) + self.eps * torch.log(1.0 + n_all)

    def forward(self, p3, small_area):
        return self.combine(*self.counts(p3, small_area))


def density_head(model):
    return P3Density(model.model[L_DET_B])


class S1Heur(nn.Module):
    """(image, small_area) -> (skor heuristik, p3, p2, p4, p5); trunk ikut dtype model, fitur FP32."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.dens = density_head(model)
        self._dt = next(model.parameters()).dtype

    def forward(self, x, small_area):
        p = self.m.trunk(x.to(self._dt))
        return (self.dens(p[0], small_area), *[t.float() for t in p])


def small_area_tensor(ratio_pad, device="cpu"):
    (gh, gw), _ = ratio_pad
    return torch.tensor([SMALL_PX ** 2 * float(gh) * float(gw)], device=device)


__all__ = ["HEUR_CONF", "HEUR_EPS", "load_static", "StaticOut", "static_with_feats", "P3Density", "density_head",
           "S1Heur", "small_area_tensor", "pool4", "np"]
