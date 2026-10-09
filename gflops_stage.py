"""
gflops_stage.py — GFLOPs per stage deployment (Stage1 / Stage2A / Stage2B) dan GFLOPs efektif router.

Keputusan K8 (2026-10-09): GFLOPs dihitung dari graf yang SAMA dengan yang diekspor ke TensorRT
(pembagian layer identik dengan Stage1_BackboneNeckGate / Stage2A_WithP2 / Stage2B_NoP2 di
uji-akurasi-latensi-model-dualbranch_diperbaiki.ipynb sel 11), lalu:

    G_eff(r) = G_Stage1 + r * G_Stage2A + (1 - r) * G_Stage2B

dengan r = aktivasi P2 TERUKUR (fraksi gambar test yang dirutekan ke Branch A oleh engine yang
sama dengan pengukuran latensi), bukan aktivasi target.

Definisi hitungan (dicatat agar tidak dicampur dengan angka GFLOPs dari sumber lain):
  - torch.utils.flop_counter.FlopCounterMode, batch 1, input 3 x imgsz x imgsz, mode eval, FP32.
  - Yang dihitung: konvolusi, matmul/linear, (de)konvolusi; 1 MAC = 2 FLOP.
  - Yang TIDAK dihitung: BatchNorm/GroupNorm, aktivasi, penjumlahan, upsample, concat, topk,
    softmax/sigmoid, decode box Detect. Karena itu angkanya bisa sedikit berbeda dari
    `model.info()` Ultralytics (thop); untuk perbandingan antar-model selalu pakai fungsi ini
    pada SEMUA model (Vanilla, P2-Static, Router).
  - NMS dan pra-pemrosesan gambar tidak termasuk (sama dengan cakupan pengukuran latensi).

Pemakaian di notebook (setelah sel 11 dan sel 28 benchmark latensi):

    from gflops_stage import stage_gflops, static_gflops, effective_gflops
    G = stage_gflops(model, imgsz=CFG["imgsz"])                 # dict stage1/stage2a/stage2b/...
    G["vanilla"] = static_gflops(vanilla_pt, CFG["imgsz"])
    G["p2static"] = static_gflops(p2static_pt, CFG["imgsz"])
    G["router_eff"] = effective_gflops(G, ACT_TRT)               # aktivasi terukur TensorRT
"""
import copy

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


# --------------------------------------------------------------------------------------------
# Pembagian stage — HARUS identik dengan notebook uji (sel 11). Bila notebook berubah, ubah juga.
# --------------------------------------------------------------------------------------------
def _run_layers(layers, x, y):
    """Jalankan layer berurutan dengan routing m.f Ultralytics; y = dict indeks layer -> output."""
    for m in layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y[m.i] = x
    return x


class _Stage1(nn.Module):
    """Layer 0–15 + gate (compute_gate_only). Output: gate, p3_neck, p2_backbone, p4_neck, p5_sppf."""

    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.pre = nn.ModuleList(ml[0:16])
        self.router = ml[16]
        assert "DifficultyAwareRouter" in type(self.router).__name__, "Layer 16 bukan router"

    def forward(self, x):
        y = {}
        x = _run_layers(self.pre, x, y)
        gate = self.router.compute_gate_only(x, y[2])
        return gate, x, y[2], y[12], y[9]


class _Stage2A(nn.Module):
    """Ekspert P2 (C2f-P2) + layer 17–25 + Detect_A."""

    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.router, self.branch, self.detect = ml[16], nn.ModuleList(ml[17:26]), ml[32]

    def forward(self, p3, p2, p4, p5):
        p2_out = self.router.compute_expert_only(p3, p2)
        y = {15: p3, 12: p4, 9: p5, 16: p2_out}
        _run_layers(self.branch, p2_out, y)
        out = self.detect([p2_out, y[19], y[22], y[25]])
        return out[0] if isinstance(out, tuple) else out


class _Stage2B(nn.Module):
    """Layer 26–31 + Detect_B."""

    def __init__(self, full_model):
        super().__init__()
        ml = list(full_model.model)
        self.branch, self.detect = nn.ModuleList(ml[26:32]), ml[33]

    def forward(self, p3, p4, p5):
        y = {15: p3, 12: p4, 9: p5}
        _run_layers(self.branch, p3, y)
        out = self.detect([p3, y[28], y[31]])
        return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------------------------
def _gflops(fn, *args):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops() / 1e9


def stage_gflops(model, imgsz=640):
    """
    GFLOPs Stage1/Stage2A/Stage2B pada salinan CPU FP32 mode eval (model asli tidak disentuh).

    Returns dict: stage1, stage2a, stage2b, route_A (=stage1+stage2a), route_B (=stage1+stage2b),
    dual_full (forward PyTorch kedua branch, referensi), definisi.
    Sanity check: stage1 + stage2a + stage2b == dual_full (gate dihitung sekali di keduanya).
    """
    m = copy.deepcopy(model).float().cpu().eval()
    s1, s2a, s2b = _Stage1(m).eval(), _Stage2A(m).eval(), _Stage2B(m).eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1(x)
        # kesetaraan output stage vs model utuh (sama dengan assert di notebook sel 11)
        yA_ref, yB_ref = (o[0] for o in m._predict_once_dual(x))
        assert torch.allclose(s2a(p3, p2, p4, p5), yA_ref, rtol=1e-4, atol=1e-4), "Stage2A != model utuh"
        assert torch.allclose(s2b(p3, p4, p5), yB_ref, rtol=1e-4, atol=1e-4), "Stage2B != model utuh"
    g1 = _gflops(s1, x)
    g2a = _gflops(s2a, p3, p2, p4, p5)
    g2b = _gflops(s2b, p3, p4, p5)
    g_dual = _gflops(m._predict_once_dual, x)
    assert abs((g1 + g2a + g2b) - g_dual) <= 1e-6 * max(g_dual, 1.0), (
        f"Stage1+2A+2B ({g1 + g2a + g2b:.6f}) != forward dual ({g_dual:.6f}) — pembagian stage tidak lengkap")
    return dict(stage1=g1, stage2a=g2a, stage2b=g2b, route_A=g1 + g2a, route_B=g1 + g2b, dual_full=g_dual,
                definisi=f"FlopCounterMode, batch 1, {imgsz}x{imgsz}, conv+matmul, 1 MAC = 2 FLOP")


def static_gflops(model, imgsz=640):
    """GFLOPs model statis (Vanilla / P2-Static) dengan penghitung yang sama."""
    m = copy.deepcopy(model).float().cpu().eval()
    return _gflops(m, torch.zeros(1, 3, imgsz, imgsz))


def effective_gflops(g, activation_rate):
    """G_eff = G_Stage1 + r * G_Stage2A + (1 - r) * G_Stage2B, r = aktivasi P2 terukur (0..1)."""
    r = float(activation_rate)
    assert 0.0 <= r <= 1.0, "activation_rate harus fraksi 0..1"
    return g["stage1"] + r * g["stage2a"] + (1.0 - r) * g["stage2b"]


if __name__ == "__main__":
    # Uji mandiri dengan bobot acak (angka arsitektur saja, tidak bergantung checkpoint).
    import json
    from ultralytics.model_yolov8_router import DualBranchDetectionModel

    mdl = DualBranchDetectionModel(cfg="ultralytics/cfg/models/v8/yolov8n-p2-router.yaml", nc=2, verbose=False)
    G = stage_gflops(mdl, 640)
    G["router_eff_r0.5"] = effective_gflops(G, 0.5)
    print(json.dumps(G, indent=1))
