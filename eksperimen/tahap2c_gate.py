"""
tahap2c_gate.py — Gate post-hoc terkunci (Tahap 2c): dari sklearn ke modul PyTorch/ONNX/TensorRT.

Gate (dikunci dari Tahap 2b, dipilih di val di 2 dari 3 seed):
    target  : y = 1[Δsoft_small > 0]   (Δ = A − B jumlah skor deteksi TP pada GT kecil, IoU 0.5, gambar TRAIN)
    model   : StandardScaler + LogisticRegression(C=0.001)
    fitur   : [GAP‖GMP](p3_neck) ‖ [GAP‖GMP](p2_backbone) ‖ [GAP‖GMP](p4_neck) ‖ [GAP‖GMP](p5_sppf) ‖ z_lama   (961)
    keputusan: skor > thr, thr = kuantil (1 − ρ) skor VAL, ρ = 0.5

Scaler + logistic dilipat jadi satu vektor bobot: s = f·w + b, w = coef/scale, b = intercept − Σ coef·mean/scale.
Di engine Stage 1 ini hanya pooling global + 1 Gemm + perbandingan -> output `gate` 0/1 dengan nama & bentuk
yang sama seperti Stage 1 lama, sehingga runtime 3-engine (TRTSystem) tidak berubah.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import tahap2_eval as E
from router_train_utils import find_router, _raw_gate_logit


def _pool(t):
    return torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1)


class PostHocGate(nn.Module):
    def __init__(self, router, w, b, thr):
        super().__init__()
        self.router = router                                   # hanya dipakai untuk z lama (fitur ke-961)
        self.register_buffer("w", torch.as_tensor(np.asarray(w), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))
        self.register_buffer("thr", torch.tensor(float(thr), dtype=torch.float32))

    def features(self, p3, p2, p4, p5):
        z = _raw_gate_logit(self.router, p3, p2).float()
        return torch.cat([_pool(p3), _pool(p2), _pool(p4), _pool(p5), z[:, None]], 1).float()

    def score(self, p3, p2, p4, p5):
        return self.features(p3, p2, p4, p5) @ self.w + self.b

    def forward(self, p3, p2, p4, p5):
        return (self.score(p3, p2, p4, p5) > self.thr).float()


class Stage1PostHoc(nn.Module):
    """Layer 0–15 + gate post-hoc. Output identik bentuk/namanya dengan Stage1 lama: (gate, p3, p2, p4, p5)."""
    def __init__(self, full_model, gate):
        super().__init__()
        ml = list(full_model.model)
        self.pre_router_layers = nn.ModuleList(ml[0:16])
        self.gate = gate

    def trunk(self, x):
        y = []
        for m in self.pre_router_layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return x, y[2], y[12], y[9]

    def forward(self, x):
        p3, p2, p4, p5 = self.trunk(x)
        return self.gate(p3, p2, p4, p5), p3, p2, p4, p5


def fit_locked_gate(feat_train, delta_train, C=0.001):
    """Returns (w, b, info). feat_train sudah termasuk z di kolom terakhir (P.features)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    y = (np.asarray(delta_train) > 0).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(feat_train, y)
    sc, lr = pipe[0], pipe[1]
    scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    w = lr.coef_[0] / scale
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / scale))
    info = dict(C=C, n_train=int(len(y)), pos_rate_train=float(y.mean()), n_features=int(feat_train.shape[1]),
                n_iter=int(np.max(lr.n_iter_)))
    return w, b, pipe, info


@torch.no_grad()
def gate_scores(stage1ph, loader, device):
    """Skor & keputusan gate per gambar (urutan loader) lewat modul yang SAMA dengan yang diekspor."""
    s, d, files = [], [], []
    for b in loader:
        im = b["img"].to(device).float() / 255.0
        p3, p2, p4, p5 = stage1ph.trunk(im)
        sc = stage1ph.gate.score(p3, p2, p4, p5)
        s.append(sc.cpu().numpy()); d.append((sc > stage1ph.gate.thr).cpu().numpy())
        files += [Path(f).name for f in b["im_file"]]
    return np.concatenate(s), np.concatenate(d), files


def export_onnx_posthoc(model, gate, vanilla, p2static, x0, out_dir, tag):
    """ONNX untuk job TensorRT (nama file/IO sama dengan Tahap 2). Returns paths, salinan CPU, statics."""
    m_cpu = E.cpu_copy(model)
    g_cpu = PostHocGate(find_router(m_cpu), gate.w.cpu().numpy(), float(gate.b), float(gate.thr)).eval()
    s1c = Stage1PostHoc(m_cpu, g_cpu).eval()
    _, s2ac, s2bc = E.make_stages(m_cpu)
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1c(x0)
    P = {k: Path(out_dir) / f"{k}_{tag}.onnx" for k in ("stage1", "stage2a", "stage2b")}
    P["vanilla"], P["p2static"] = Path(out_dir) / "vanilla.onnx", Path(out_dir) / "p2static.onnx"
    E.export_onnx(E._Stage1Out(s1c), (x0,), P["stage1"], ["image"], ["gate", "p3_neck", "p2_backbone", "p4_neck", "p5_sppf"])
    E.export_onnx(s2ac, (p3, p2, p4, p5), P["stage2a"], ["p3_neck", "p2_backbone", "p4_neck", "p5_sppf"], ["y_A"])
    E.export_onnx(s2bc, (p3, p4, p5), P["stage2b"], ["p3_neck", "p4_neck", "p5_sppf"], ["y_B"])
    statics = {"vanilla": E._StaticOut(E.cpu_copy(vanilla)).eval(), "p2static": E._StaticOut(E.cpu_copy(p2static)).eval()}
    for k, mod in statics.items():
        E.export_onnx(mod, (x0,), P[k], ["image"], ["y"])
    return P, (s1c, s2ac, s2bc), statics


def verify_onnx_posthoc(P, cpu_stages, statics, loader, n=5):
    """Seperti E.verify_onnx; keputusan gate boleh beda HANYA bila |skor − thr| < 1e-4 (batas numerik)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(P["stage1"]), providers=["CPUExecutionProvider"])
    s1c = cpu_stages[0]
    with torch.no_grad():
        for k, b in enumerate(loader):
            if k >= n:
                break
            x = b["img"].float() / 255.0
            g, p3, p2, p4, p5 = s1c(x)
            o = sess.run(None, {"image": x.numpy()})
            if float(o[0].flatten()[0]) != float(g.flatten()[0]):
                margin = float((s1c.gate.score(p3, p2, p4, p5) - s1c.gate.thr).abs())
                assert margin < 1e-4, f"Gate ONNX != PyTorch pada {b['im_file'][0]} (margin {margin:.2e})"
    return _verify_rest(P, cpu_stages, statics, loader, n)


def _verify_rest(P, cpu_stages, statics, loader, n):
    import onnxruntime as ort
    rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    sess = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in P.items()}
    run = lambda k, feed: sess[k].run(None, {i.name: feed[i.name] for i in sess[k].get_inputs()})
    s1c, s2ac, s2bc = cpu_stages
    worst = {}
    with torch.no_grad():
        for k, b in enumerate(loader):
            if k >= n:
                break
            x = b["img"].float() / 255.0
            _, p3, p2, p4, p5 = s1c(x)
            o1 = run("stage1", {"image": x.numpy()})
            worst["stage1_feats"] = max(worst.get("stage1_feats", 0), *[rel(a, t.numpy()) for a, t in zip(o1[1:], (p3, p2, p4, p5))])
            f = {"p3_neck": p3.numpy(), "p2_backbone": p2.numpy(), "p4_neck": p4.numpy(), "p5_sppf": p5.numpy()}
            worst["stage2a"] = max(worst.get("stage2a", 0), rel(run("stage2a", f)[0], s2ac(p3, p2, p4, p5).numpy()))
            worst["stage2b"] = max(worst.get("stage2b", 0), rel(run("stage2b", f)[0], s2bc(p3, p4, p5).numpy()))
            for nm, mod in statics.items():
                worst[nm] = max(worst.get(nm, 0), rel(run(nm, {"image": x.numpy()})[0], mod(x).numpy()))
    assert all(v < 1e-4 for v in worst.values()), f"Selisih ONNX vs PyTorch > 1e-4: {worst}"
    return worst
