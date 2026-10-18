"""
Ekspor deployment A3 (router_final): SATU graf ONNX berisi Stage 1 + If(Stage 2A | Stage 2B).

    input : image (1, 3, 640, 640) FP32 [0, 1], mode (1,) FP32, thr (1,) FP32
    cond  : mode < 0 ? (score > thr) : (mode > 0.5)          mode −1 = ikut gate, 1 = paksa A, 0 = paksa B
    output: y (1, 4 + nc, 34000) — rute B di-pad nol dari 8400 (skor kelas 0 -> dibuang ambang conf NMS),
            use_a (1,), score (1,)
Threshold adalah INPUT engine dan skor adalah OUTPUT, sehingga B2 berjalan di host tanpa engine kedua.

FP16: trunk & Stage 2 FP16 dengan I/O FP32; pooling + Gemm gate FP32 (keputusan tidak berubah karena pembulatan
bobot gate). TensorRT 10+/11 dengan jaringan STRONGLY_TYPED -> presisi = tipe di ONNX.
Diporting dari eksperimen/tahap4_final.py (build_if_v2, export_final) dan tahap3a2 (_prefixed_graph, _IOCast).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .gate import GateHead
from .model import FEATS_A, FEATS_B, BranchA, BranchB, reset_anchor_cache


class Stage1(nn.Module):
    """image -> (score, p3, p2, p4, p5). Trunk ikut dtype model; fitur & skor FP32."""

    def __init__(self, model, gate):
        super().__init__()
        self.m = model
        self.gate = GateHead(gate["w"], gate["b"])
        self._dt = next(model.parameters()).dtype

    def forward(self, x):
        p = [t.float() for t in self.m.trunk(x.to(self._dt))]
        return (self.gate(*p), *p)


class IOCast(nn.Module):
    """Bobot FP16 di dalam, I/O FP32."""

    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, *xs):
        out = self.m(*[x.half() for x in xs])
        return tuple(o.float() for o in out) if isinstance(out, (tuple, list)) else out.float()


def cpu_copy(m):
    return reset_anchor_cache(copy.deepcopy(m).cpu().float().eval())


def half_copy(m, device):
    return reset_anchor_cache(copy.deepcopy(m).to(device).half().eval())


def export_onnx(module, args, path, in_names, out_names, opset=17):
    with torch.no_grad():
        torch.onnx.export(module.eval(), args, str(path), input_names=in_names, output_names=out_names,
                          opset_version=opset, dynamic_axes=None, dynamo=False)


def _prefixed_graph(model, prefix, keep):
    """Salin graph dengan semua nama tensor/node diberi prefix, kecuali nama di `keep` (input dari scope luar)."""
    import onnx
    g = model.graph
    ren = lambda n: n if (n == "" or n in keep) else prefix + n
    nodes = []
    for nd in g.node:
        new = onnx.NodeProto(); new.CopyFrom(nd)
        new.name = prefix + (nd.name or nd.op_type)
        del new.input[:]; new.input.extend([ren(i) for i in nd.input])
        del new.output[:]; new.output.extend([ren(o) for o in nd.output])
        nodes.append(new)
    inits = []
    for t in g.initializer:
        new = onnx.TensorProto(); new.CopyFrom(t); new.name = ren(t.name); inits.append(new)
    return nodes, inits, [ren(o.name) for o in g.output]


def build_if(p_s1, p_s2a, p_s2b, out_path, n_a, n_b):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m1, ma, mb = onnx.load(str(p_s1)), onnx.load(str(p_s2a)), onnx.load(str(p_s2b))
    g1 = m1.graph
    assert [o.name for o in g1.output] == ["score", *FEATS_A], [o.name for o in g1.output]
    no = int(ma.graph.output[0].type.tensor_type.shape.dim[1].dim_value)
    keep = set(FEATS_A)
    na, ia, (ya,) = _prefixed_graph(ma, "A__", keep)
    nb, ib, (yb,) = _prefixed_graph(mb, "B__", keep)
    pad = numpy_helper.from_array(np.zeros((1, no, n_a - n_b), np.float32), "B__pad_zeros")
    nb.append(helper.make_node("Concat", [yb, "B__pad_zeros"], ["B__y_pad"], axis=2, name="B__pad_concat"))
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, no, n_a])
    then_g = helper.make_graph(na, "rute_A", [], [vi(ya)], initializer=ia)
    else_g = helper.make_graph(nb, "rute_B", [], [vi("B__y_pad")], initializer=ib + [pad])
    nodes = list(g1.node) + [
        helper.make_node("Constant", [], ["c_zero"], value=numpy_helper.from_array(np.zeros(1, np.float32))),
        helper.make_node("Constant", [], ["c_half"], value=numpy_helper.from_array(np.full(1, 0.5, np.float32))),
        helper.make_node("Constant", [], ["c_ax0"], value=numpy_helper.from_array(np.zeros(1, np.int64))),
        helper.make_node("Less", ["mode", "c_zero"], ["is_auto"]),
        helper.make_node("Greater", ["score", "thr"], ["gate_on"]),
        helper.make_node("Cast", ["gate_on"], ["gate_f"], to=TensorProto.FLOAT),
        helper.make_node("Where", ["is_auto", "gate_f", "mode"], ["sel_f"]),
        helper.make_node("Greater", ["sel_f", "c_half"], ["sel"]),
        helper.make_node("Squeeze", ["sel", "c_ax0"], ["cond"]),
        helper.make_node("If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g, name="rute_if"),
        helper.make_node("Cast", ["sel"], ["use_a"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "router_final_a3", list(g1.input) + [helper.make_tensor_value_info("mode", TensorProto.FLOAT, [1]),
                                                     helper.make_tensor_value_info("thr", TensorProto.FLOAT, [1])],
        [vi("y"), helper.make_tensor_value_info("use_a", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])],
        initializer=list(g1.initializer))
    model = helper.make_model(graph, opset_imports=list(m1.opset_import), producer_name="router_final")
    model.ir_version = m1.ir_version
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def export_a3(model, gate, x0, out_dir, precision="fp32", device="cpu", opset=17):
    """
    Returns (paths, mods). FP32 (CPU) selalu diekspor (verifikasi ONNX Runtime); FP16 butuh GPU.
    paths[presisi] = {s1, s2a, s2b, a3}; mods = modul FP32 CPU (acuan verifikasi) + n_a, n_b.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    m = cpu_copy(model)
    s1, s2a, s2b = Stage1(m, gate).eval(), BranchA(m).eval(), BranchB(m).eval()
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1(x0)
        n_a, n_b = int(s2a(p3, p2, p4, p5).shape[-1]), int(s2b(p3, p4, p5).shape[-1])
    P32 = {k: out / f"{k}_fp32.onnx" for k in ("s1", "s2a", "s2b", "a3")}
    export_onnx(s1, (x0,), P32["s1"], ["image"], ["score", *FEATS_A], opset)
    export_onnx(s2a, (p3, p2, p4, p5), P32["s2a"], list(FEATS_A), ["y_A"], opset)
    export_onnx(s2b, (p3, p4, p5), P32["s2b"], list(FEATS_B), ["y_B"], opset)
    build_if(P32["s1"], P32["s2a"], P32["s2b"], P32["a3"], n_a, n_b)
    paths = {"fp32": P32}
    if precision == "fp16":
        m16 = half_copy(model, device)
        s1h, s2ah, s2bh = Stage1(m16, gate).to(device).eval(), BranchA(m16).eval(), BranchB(m16).eval()
        x = x0.to(device)
        with torch.no_grad():
            _, q3, q2, q4, q5 = s1h(x)
            h = [t.half() for t in (q3, q2, q4, q5)]
            s2ah(*h); s2bh(h[0], h[2], h[3])                    # anchors Detect (half) dihitung di luar tracing
        P16 = {k: out / f"{k}_fp16.onnx" for k in ("s1", "s2a", "s2b", "a3")}
        export_onnx(s1h, (x,), P16["s1"], ["image"], ["score", *FEATS_A], opset)
        export_onnx(IOCast(s2ah), (q3, q2, q4, q5), P16["s2a"], list(FEATS_A), ["y_A"], opset)
        export_onnx(IOCast(s2bh), (q3, q4, q5), P16["s2b"], list(FEATS_B), ["y_B"], opset)
        build_if(P16["s1"], P16["s2a"], P16["s2b"], P16["a3"], n_a, n_b)
        paths["fp16"] = P16
    elif precision != "fp32":
        raise ValueError(precision)
    return paths, dict(s1=s1, s2a=s2a, s2b=s2b, n_a=n_a, n_b=n_b)


def verify_a3(path_a3_fp32, mods, images, tol=1e-4):
    """ONNX Runtime (graf FP32) vs PyTorch untuk ketiga mode dan dua arah threshold. Returns selisih terburuk."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path_a3_fp32), providers=["CPUExecutionProvider"])
    rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    worst = {"y": 0.0, "score": 0.0}
    with torch.no_grad():
        for x in images:
            s, p3, p2, p4, p5 = mods["s1"](x)
            s = float(s[0])
            ya = mods["s2a"](p3, p2, p4, p5).numpy()
            yb = mods["s2b"](p3, p4, p5).numpy()
            ybp = np.concatenate([yb, np.zeros((1, yb.shape[1], mods["n_a"] - mods["n_b"]), np.float32)], 2)
            for mode, thr, want in ((1.0, 1e9, True), (0.0, -1e9, False), (-1.0, s - 0.5, True), (-1.0, s + 0.5, False)):
                y, use_a, score = sess.run(None, {"image": x.numpy(), "mode": np.array([mode], np.float32),
                                                  "thr": np.array([thr], np.float32)})
                assert bool(use_a[0] > 0.5) == want, f"rute salah (mode {mode}, thr {thr})"
                worst["y"] = max(worst["y"], rel(y, ya if want else ybp))
                worst["score"] = max(worst["score"], abs(float(score[0]) - s) / (1.0 + abs(s)))
    assert all(v < tol for v in worst.values()), f"ONNX vs PyTorch > {tol}: {worst}"
    return worst
