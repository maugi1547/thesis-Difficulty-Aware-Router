"""
A12 — engine A3 tanpa biaya *padding* keluaran rute B pada batas L1 (rancangan lanjutan; pasca-hasil).

Masalah (profil segmen KITTI s42): rute B − Vanilla = +0,096 ms, dengan salinan keluaran +0,048 ms karena keluaran If
rute B di-*padding* 8.400 -> 34.000 anchor (1×6×34.000, 816 KB) agar kedua cabang If berbentuk sama. Salin 8.400 anchor
dari tata letak (1, 6, N) = irisan tidak bersebelahan (6 potongan), sehingga hasilnya tidak konsisten antar-sesi.

Rancangan A3T: engine yang SAMA (graf If identik) + satu Transpose di akhir -> y_t (1, N, 6). Anchor rute B kini AWALAN
bersebelahan y_t[:, :8.400] (anchor padding nol ada di belakang), sehingga host dapat menyalin awalan saja.
Router A3T "dua tahap" (tanpa menunggu keputusan sebelum salinan pertama):
    thr (host) -> H2D citra + thr -> engine -> salin awalan 8.400 + use_a + skor -> sync
    -> keputusan & push B2 -> HANYA bila rute A: salin sisa 25.600 anchor -> sync
Rute B menghemat salinan 25.600 × 6 float; rute A membayar satu salinan + sync tambahan. Akurasi tidak berubah (nilai
keluaran sama, hanya tata letak), sehingga H5 dihitung dari pass akurasi suite dengan f baru (scripts/h6_latency.py).

Pembanding dalam sesi yang sama (wajib, karena variasi antar-sesi sebesar efek yang dicari): router A3 asli
("Router B2 efisien (e2e)", = suite) -> f kontrol.
"""
from __future__ import annotations

import collections
import time

import numpy as np
import torch

from .budget import SortedWindowQuantile

E2E = ("Vanilla", "P2-Static", "A3 rute A", "A3 rute B", "A3T rute A", "A3T rute B (salin awalan)",
       "A3T rute B (salin penuh)", "Router B2 efisien (e2e)", "Router A3T dua tahap")
ROUTERS = ("Router B2 efisien (e2e)", "Router A3T dua tahap")


def transpose_output(src, dst):
    """ONNX A3 (keluaran y (1, no, N)) -> A3T (keluaran y (1, N, no)) dengan satu Transpose setelah If."""
    import onnx
    from onnx import helper
    m = onnx.load(str(src))
    g = m.graph
    outs = [o for o in g.output if o.name == "y"]
    assert len(outs) == 1, [o.name for o in g.output]
    nd = [n for n in g.node if "y" in n.output]
    assert len(nd) == 1 and nd[0].op_type == "If", [n.op_type for n in nd]
    nd[0].output[list(nd[0].output).index("y")] = "y_pad"
    dims = [d.dim_value for d in outs[0].type.tensor_type.shape.dim]
    assert len(dims) == 3 and dims[0] == 1, dims
    g.node.append(helper.make_node("Transpose", ["y_pad"], ["y"], perm=[0, 2, 1], name="a12_transpose"))
    et = outs[0].type.tensor_type.elem_type
    idx = list(g.output).index(outs[0])
    g.output.remove(outs[0])
    g.output.insert(idx, helper.make_tensor_value_info("y", et, [1, dims[2], dims[1]]))
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return dict(no=dims[1], n=dims[2])


def make_ops(eng, n_a, n_b, dev, pin, s, sync):
    """eng: vanilla, p2static, a3g, a3t. Ops = callable tanpa argumen; host buffer di H (pinned bila CUDA)."""
    H = {"img": pin(eng["vanilla"].buf["image"]), "thr": pin(torch.zeros(1))}
    modes = {mv: torch.full((1,), mv, dtype=torch.float32, device=dev) for mv in (-1.0, 0.0, 1.0)}
    ops = {}
    for k in ("vanilla", "p2static"):
        e = eng[k]; ho = pin(e.buf[e.out[0]]); H[k] = ho

        def f(e=e, ho=ho):
            e.buf["image"].copy_(H["img"], non_blocking=True); e.run(s)
            ho.copy_(e.buf[e.out[0]], non_blocking=True); sync()
        ops[k] = f

    e = eng["a3g"]                                         # = suite make_ops (salin y penuh)
    hy, hu, hs = pin(e.buf["y"]), pin(e.buf["use_a"]), pin(e.buf["score"]); H["a3g"] = (hy, hu, hs)
    for mv in (-1.0, 0.0, 1.0):
        def f(e=e, mv=mv):
            e.bind("mode", modes[mv])
            e.buf["image"].copy_(H["img"], non_blocking=True); e.buf["thr"].copy_(H["thr"], non_blocking=True)
            e.run(s)
            hy.copy_(e.buf["y"], non_blocking=True); hu.copy_(e.buf["use_a"], non_blocking=True)
            hs.copy_(e.buf["score"], non_blocking=True); sync()
        ops[("a3g", mv)] = f

    t = eng["a3t"]
    yt = t.buf["y"]
    assert tuple(yt.shape[1:2]) == (n_a,), yt.shape
    ty_full, tu, ts = pin(yt), pin(t.buf["use_a"]), pin(t.buf["score"])
    ty_pre, ty_rest = pin(yt[:, :n_b]), pin(yt[:, n_b:])
    H["a3t"] = (ty_full, tu, ts, ty_pre, ty_rest)

    def launch(mv):
        t.bind("mode", modes[mv])
        t.buf["image"].copy_(H["img"], non_blocking=True); t.buf["thr"].copy_(H["thr"], non_blocking=True)
        t.run(s)

    def a3t_full(mv):
        def f():
            launch(mv)
            ty_full.copy_(t.buf["y"], non_blocking=True); tu.copy_(t.buf["use_a"], non_blocking=True)
            ts.copy_(t.buf["score"], non_blocking=True); sync()
        return f

    def a3t_prefix_b():
        launch(0.0)
        ty_pre.copy_(t.buf["y"][:, :n_b], non_blocking=True); tu.copy_(t.buf["use_a"], non_blocking=True)
        ts.copy_(t.buf["score"], non_blocking=True); sync()

    def a3t_two_stage():
        """Router: awalan + keputusan dulu; sisa anchor hanya bila rute A. Kembali: keputusan (bool)."""
        launch(-1.0)
        ty_pre.copy_(t.buf["y"][:, :n_b], non_blocking=True); tu.copy_(t.buf["use_a"], non_blocking=True)
        ts.copy_(t.buf["score"], non_blocking=True); sync()
        use = bool(float(tu[0]) > 0.5)
        if use:
            ty_rest.copy_(t.buf["y"][:, n_b:], non_blocking=True); sync()
        return use

    ops[("a3t", 1.0)] = a3t_full(1.0)
    ops[("a3t", "awalan")] = a3t_prefix_b
    ops[("a3t", 0.0)] = a3t_full(0.0)
    ops[("a3t", "dua_tahap")] = a3t_two_stage
    return ops, H


def e2e_pass(ops, H, batches, rho, window, min_hist, thr0, n_warmup, n_reps, ctx, dsync):
    """Sama dengan rfinal.suite.e2e_pass (timer: threshold B2 host -> H2D -> engine -> D2H -> sync), varian E2E."""
    def b2(kind):
        hist = SortedWindowQuantile(window)

        def f():
            thr = np.float32(hist.quantile(1 - rho)) if len(hist.h) >= min_hist else np.float32(thr0)
            H["thr"].fill_(float(thr))
            if kind == "a3g":
                ops[("a3g", -1.0)]()
                use, sc = bool(float(H["a3g"][1][0]) > 0.5), float(H["a3g"][2][0])
            else:
                use = ops[("a3t", "dua_tahap")]()
                sc = float(H["a3t"][2][0])
            hist.push(sc)
            return use
        return f

    def fixed(op):
        def f():
            H["thr"].fill_(float(thr0)); op()
        return f

    def fresh():
        return {"Vanilla": ops["vanilla"], "P2-Static": ops["p2static"],
                "A3 rute A": fixed(ops[("a3g", 1.0)]), "A3 rute B": fixed(ops[("a3g", 0.0)]),
                "A3T rute A": fixed(ops[("a3t", 1.0)]), "A3T rute B (salin awalan)": fixed(ops[("a3t", "awalan")]),
                "A3T rute B (salin penuh)": fixed(ops[("a3t", 0.0)]),
                "Router B2 efisien (e2e)": b2("a3g"), "Router A3T dua tahap": b2("a3t")}

    def timed(fn, im):
        H["img"].copy_(im.float() / 255.0); dsync()
        w0 = time.perf_counter()
        with ctx():
            u = fn()
        return (time.perf_counter() - w0) * 1e3, u

    warm = fresh()
    for im in batches[:n_warmup]:
        for fn in warm.values():
            timed(fn, im)
    lat = {k: [] for k in E2E}
    use = {k: [] for k in ROUTERS}
    for rep in range(n_reps):
        fns = fresh(); order = list(fns)
        for i, im in enumerate(batches):
            for k in (order if (rep + i) % 2 == 0 else order[::-1]):
                ms, u = timed(fns[k], im)
                lat[k].append(ms)
                if rep == 0 and k in use:
                    use[k].append(u)
    return lat, use


def parity(eng, ops, H, imgs, n_b):
    """A3T vs A3 (engine berbeda, FP16): y_t == y^T untuk rute paksa A dan B; padding A3T nol; skor sama."""
    rows = collections.defaultdict(float)
    for im in imgs:
        H["img"].copy_(im.float() / 255.0); H["thr"].fill_(0.0)
        for mv in (1.0, 0.0):
            ops[("a3g", mv)](); ops[("a3t", mv)]()
            y = H["a3g"][0].float(); yt = H["a3t"][0].float().transpose(1, 2)
            d = (y - yt).abs().max().item()
            rows[f"y rute {'A' if mv else 'B'} (maks abs)"] = max(rows[f"y rute {'A' if mv else 'B'} (maks abs)"], d)
            rows[f"y rute {'A' if mv else 'B'} (maks rel)"] = max(rows[f"y rute {'A' if mv else 'B'} (maks rel)"],
                                                                  d / (y.abs().max().item() + 1e-12))
            rows["skor A3 vs A3T (maks abs)"] = max(rows["skor A3 vs A3T (maks abs)"],
                                                    abs(float(H["a3g"][2][0]) - float(H["a3t"][2][0])))
            if not mv:
                rows["padding A3T rute B (maks abs)"] = max(rows["padding A3T rute B (maks abs)"],
                                                            H["a3t"][0][:, n_b:].abs().max().item())
        ops[("a3t", "awalan")]()
        rows["awalan == y_t[:, :n_b] (maks abs)"] = max(rows["awalan == y_t[:, :n_b] (maks abs)"],
                                                         (H["a3t"][3].float() - H["a3t"][0][:, :n_b].float()).abs().max().item())
    return dict(rows)
