"""
tahap4_final.py — Evaluasi final sistem router (protokol dikunci 16 Okt 2026, sebelum dijalankan).

Sistem final (tidak ada pemilihan apa pun di notebook ini):
  detektor dual-branch M0 (beku, last.pt)
  + gate post-hoc tanpa z: [GAP‖GMP](p3, p2, p4, p5) = 960 fitur, StandardScaler + LogReg C = 0.001 (target soft_small
    biner dari train), dilipat menjadi satu Gemm
  + kendali anggaran B2: threshold = kuantil (1 − ρ) skor N = 200 gambar sebelumnya (ρ = 0.5); < 20 riwayat -> kuantil val
  + deployment A3 FP16: SATU engine TensorRT dengan ONNX `If`; threshold = INPUT engine, skor = OUTPUT engine,
    sehingga B2 berjalan di host tanpa engine kedua.

Pembanding:
  Vanilla, P2-Static, Branch A / B paksa, routing acak (aktivasi sama), campuran acak Vanilla/P2-Static pada latensi
  sama (H6), heuristik kepadatan tanpa label di Stage 1 (head P3 Detect_B, B2 yang sama), kaskade bergate
  Vanilla -> P2-Static (eksploratif; latensi dirakit dari komponen terukur).

Akurasi utama = keluaran TensorRT FP16 (sistem yang di-deploy). PyTorch FP32 = pemeriksaan silang.
Semua metrik routing dirakit dari evalImgs COCO per gambar (CocoMix, Tahap 2b), sehingga satu COCOeval per keluaran
cukup untuk ratusan routing. Bootstrap berpasangan per gambar memakai indeks resample yang SAMA untuk semua model
(boot_seed tetap) -> selisih antar-seed KITTI (test sama) bisa dirata-rata per resample di ringkasan gabungan.

Job TensorRT berjalan di proses terpisah (python -m tahap4_final job.json), memakai PyTorch untuk memori/stream.
Backend "ort" (ONNX Runtime CPU, graf FP32) memakai alur yang sama persis; dipakai untuk uji lokal tanpa GPU.
"""
from __future__ import annotations

import collections
import contextlib
import gc
import glob
import io
import json
import math
import multiprocessing as mp
import pickle
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import tahap2_eval as E
import tahap2b_plafon as P
import tahap2c_gate as G
import tahap3a as A3
import tahap3a2 as X

FEATS_A, FEATS_B = X.FEATS_A, X.FEATS_B
SMALL_PX = 32.0          # batas "kecil" COCO (piksel asli)
HEUR_CONF = 0.25         # ambang skor kelas maks anchor P3 (dikunci; sama dengan proksi kepadatan D2b)
HEUR_EPS = 0.01          # bobot pemecah seri: skor = log(1 + n_kecil) + HEUR_EPS · log(1 + n_semua)
METRICS = E.BOOT_METRICS  # ("mAP50-95", "mAP50", "mAP_small")
H6_SENS = "[sensitivitas, pasca-hasil] H6 | router (B2 efisien) − campuran acak V/P2S (latensi sama)"


def _pool(t):
    return torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1)


def pool4(p3, p2, p4, p5):
    return torch.cat([_pool(p3), _pool(p2), _pool(p4), _pool(p5)], 1).float()


# =============================================================================
# MODUL STAGE 1: gate post-hoc (skor) dan heuristik kepadatan P3 (tanpa label)
# =============================================================================
class P3Density(nn.Module):
    """
    Heuristik kepadatan objek kecil TANPA label dan TANPA fit, dihitung di Stage 1 dari head P3 Detect_B
    (layer 15, stride 8): n_semua = jumlah anchor P3 dengan skor kelas maks ≥ HEUR_CONF; n_kecil = yang luas kotak
    hasil decode-nya < small_area (piksel letterbox², diisi host = 32² · gain_h · gain_w = "kecil" COCO di piksel asli).
    skor = log(1 + n_kecil) + HEUR_EPS · log(1 + n_semua). Tanpa NMS: jumlah anchor ≠ jumlah objek, tetapi untuk
    PENGURUTAN yang penting monotonisitasnya (diperiksa terhadap versi deteksi lengkap D2b).
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
        hit = cls.amax(1).reshape(b, h * w) > self.logit_thr                      # sigmoid(c) > conf  <=>  c > logit(conf)
        d = self.dfl(box.reshape(b, 4 * self.reg_max, h * w)).float() * self.stride   # jarak l, t, r, b (piksel letterbox)
        area = (d[:, 0] + d[:, 2]) * (d[:, 1] + d[:, 3])
        small = area < small_area.float().reshape(-1, 1)
        return hit.float().sum(1), (hit & small).float().sum(1)

    def combine(self, n_all, n_small):
        return torch.log(1.0 + n_small) + self.eps * torch.log(1.0 + n_all)

    def forward(self, p3, small_area):
        return self.combine(*self.counts(p3, small_area))


class S1Gate(nn.Module):
    """image -> (score, p3, p2, p4, p5). Trunk ikut dtype model (FP32/FP16); fitur, Gemm gate dan skor FP32."""
    def __init__(self, model, w, b):
        super().__init__()
        self.s = G.Stage1PostHoc(model, nn.Identity())
        self.register_buffer("w", torch.as_tensor(np.asarray(w), dtype=torch.float32))
        self.register_buffer("b", torch.tensor([float(b)], dtype=torch.float32))
        self._dt = next(model.parameters()).dtype

    def forward(self, x):
        p = [t.float() for t in self.s.trunk(x.to(self._dt))]
        return (pool4(*p) @ self.w + self.b, *p)


class S1Heur(nn.Module):
    """(image, small_area) -> (score, p3, p2, p4, p5); skor = P3Density (head P3 Detect_B di Stage 1)."""
    def __init__(self, model):
        super().__init__()
        self.s = G.Stage1PostHoc(model, nn.Identity())
        self.dens = P3Density(list(model.model)[33])
        self._dt = next(model.parameters()).dtype

    def forward(self, x, small_area):
        p = self.s.trunk(x.to(self._dt))
        return (self.dens(p[0], small_area), *[t.float() for t in p])


@torch.no_grad()
def static_with_feats(m, x):
    """Model statis (Vanilla/P2-Static) -> (y, (p3 = layer 15, p2 = 2, p4 = 12, p5 = 9)) untuk gate kaskade."""
    y = []
    for mod in m.model:
        if mod.f != -1:
            x = y[mod.f] if isinstance(mod.f, int) else [x if j == -1 else y[j] for j in mod.f]
        x = mod(x)
        y.append(x)
    out = x[0] if isinstance(x, (tuple, list)) else x
    return out, (y[15], y[2], y[12], y[9])


def small_area_of(ratio_pad):
    (gh, gw), _ = ratio_pad
    return SMALL_PX ** 2 * float(gh) * float(gw)


# =============================================================================
# CACHE PyTorch (train / val / test)
# =============================================================================
@torch.no_grad()
def cache_split(model, loader, device, conf, iou, max_det, dets=True, heur=True, statics=None, max_images=None,
                log_every=1000, log=print):
    """Satu forward per gambar (letterbox persegi, batch 1, urutan loader). Returns dict list/array per gambar."""
    trunk = G.Stage1PostHoc(model, nn.Identity()).eval()
    _, s2a, s2b = E.make_stages(model)
    dens = P3Density(list(model.model)[33]).eval() if heur else None
    van, p2s = statics if statics else (None, None)
    keys = ("file", "image_id", "feat", "detA", "detB", "heur", "heur_n_all", "heur_n_small", "small_area",
            "featV", "detV", "detP")
    out = {k: [] for k in keys}
    t0 = time.perf_counter()
    for i, b in enumerate(loader):
        if max_images and i >= max_images:
            break
        im = b["img"].to(device).float() / 255.0
        rp, osh = b["ratio_pad"][0], b["ori_shape"][0]
        sa = small_area_of(rp)
        p3, p2, p4, p5 = trunk.trunk(im)
        out["feat"].append(pool4(p3, p2, p4, p5)[0].cpu().numpy())
        if dets:
            out["detA"].append(P._to_dets(s2a(p3, p2, p4, p5), rp, osh, conf, iou, max_det))
            out["detB"].append(P._to_dets(s2b(p3, p4, p5), rp, osh, conf, iou, max_det))
        if dens is not None:
            na, ns = dens.counts(p3, torch.tensor([sa], device=device))
            out["heur"].append(float(dens.combine(na, ns)[0])); out["heur_n_all"].append(float(na[0]))
            out["heur_n_small"].append(float(ns[0]))
        out["small_area"].append(sa)
        if van is not None:
            yV, fV = static_with_feats(van, im)
            out["featV"].append(pool4(*fV)[0].cpu().numpy())
            out["detV"].append(P._to_dets(yV, rp, osh, conf, iou, max_det))
            yP = p2s(im)
            out["detP"].append(P._to_dets(yP[0] if isinstance(yP, (tuple, list)) else yP, rp, osh, conf, iou, max_det))
        stem = Path(b["im_file"][0]).stem
        out["file"].append(Path(b["im_file"][0]).name)
        out["image_id"].append(int(stem) if stem.isnumeric() else stem)
        if log_every and (i + 1) % log_every == 0:
            log(f"  {i + 1} gambar | {time.perf_counter() - t0:.0f} s")
    for k in ("feat", "featV"):
        out[k] = np.stack(out[k]) if out[k] else None
    for k in ("heur", "heur_n_all", "heur_n_small", "small_area"):
        out[k] = np.asarray(out[k], float)
    return out


def jdict_list(dets, image_ids):
    js = []
    for d, iid in zip(dets, image_ids):
        for x, y, w, h, s, c in d:
            js.append({"image_id": iid, "category_id": int(c), "bbox": [float(x), float(y), float(w), float(h)], "score": float(s)})
    return js


def coco_of(gt, dets, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        return E.coco_eval(gt, jdict_list(dets, image_ids))


def router_of(gt, dets_a, dets_b, image_ids):
    """P.Router: use_a True -> dets_a. Returns (router, stats_a, stats_b)."""
    sa, ea = coco_of(gt, dets_a, image_ids)
    sb, eb = coco_of(gt, dets_b, image_ids)
    return P.Router(ea, eb, image_ids), sa, sb


def fit_gate(X_, delta, C):
    X_ = np.asarray(X_, np.float64)                          # fit & cek lipatan di float64 (fitur cache float32)
    pipe = A3.fit_logreg(X_, delta, C)
    w, b = A3.folded_weights(pipe)
    ref = pipe.decision_function(X_[:256])
    assert np.allclose(X_[:256] @ w + b, ref, atol=1e-6 * max(1.0, np.abs(ref).max())), "lipatan scaler + logreg salah"
    y = (np.asarray(delta) > 0)
    return dict(w=np.asarray(w, np.float64), b=float(b), info=dict(C=C, n_train=int(len(y)), pos_rate=float(y.mean()),
                                                                   n_fitur=int(X_.shape[1])))


# =============================================================================
# EKSPOR ONNX: A3 v2 (threshold = input, skor = output)
# =============================================================================
def build_if_v2(p_s1, p_s2a, p_s2b, out_path, n_a, n_b):
    """
    Satu graf: (input Stage 1 [image, (small_area)], mode, thr) -> Stage 1 -> If(cond) {Stage 2A} else {Stage 2B + pad 0}.
      cond = mode < 0 ? (score > thr) : (mode > 0.5)       (mode −1 ikut skor, 1 paksa A, 0 paksa B)
    Keluaran: y (1, 4+nc, n_a), use_a (1,), score (1,). Padding nol (skor kelas 0) dibuang ambang conf NMS.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m1, ma, mb = onnx.load(str(p_s1)), onnx.load(str(p_s2a)), onnx.load(str(p_s2b))
    g1 = m1.graph
    assert [o.name for o in g1.output] == ["score", *FEATS_A], [o.name for o in g1.output]
    no = int(ma.graph.output[0].type.tensor_type.shape.dim[1].dim_value)
    keep = set(FEATS_A)
    na, ia, (ya,) = X._prefixed_graph(ma, "A__", keep)
    nb, ib, (yb,) = X._prefixed_graph(mb, "B__", keep)
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
        helper.make_node("Where", ["is_auto", "gate_f", "mode"], ["sel_f"]),        # Where float (pola 3a2 yang lolos TRT)
        helper.make_node("Greater", ["sel_f", "c_half"], ["sel"]),
        helper.make_node("Squeeze", ["sel", "c_ax0"], ["cond"]),
        helper.make_node("If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g, name="rute_if"),
        helper.make_node("Cast", ["sel"], ["use_a"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "router_a3_v2",
        list(g1.input) + [helper.make_tensor_value_info("mode", TensorProto.FLOAT, [1]),
                          helper.make_tensor_value_info("thr", TensorProto.FLOAT, [1])],
        [vi("y"), helper.make_tensor_value_info("use_a", TensorProto.FLOAT, [1]),
         helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])],
        initializer=list(g1.initializer))
    model = helper.make_model(graph, opset_imports=list(m1.opset_import), producer_name="tahap4_final")
    model.ir_version = m1.ir_version
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def export_final(model, van, p2s, gate, x0, out_dir, device, precisions=("fp16",)):
    """
    FP32 (CPU) selalu diekspor (verifikasi ONNX Runtime + backend ORT). FP16 (GPU) bila diminta: trunk, Stage 2 dan
    baseline FP16 dengan I/O FP32; Gemm gate & skor heuristik FP32 (pola tahap3a2.export_fp16 yang sudah lolos di
    TensorRT 11). Returns ({presisi: {nama: path}}, modul FP32 CPU untuk verifikasi).
    """
    out = Path(out_dir)
    sa0 = torch.tensor([SMALL_PX ** 2 * 0.25])
    m = E.cpu_copy(model)
    s1g, s1h = S1Gate(m, gate["w"], gate["b"]).eval(), S1Heur(m).eval()
    _, s2a, s2b = E.make_stages(m)
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1g(x0)
        n_a, n_b = int(s2a(p3, p2, p4, p5).shape[-1]), int(s2b(p3, p4, p5).shape[-1])
    statics = {"vanilla": E._StaticOut(E.cpu_copy(van)).eval(), "p2static": E._StaticOut(E.cpu_copy(p2s)).eval()}
    P32 = {k: out / f"{k}_fp32.onnx" for k in ("s1g", "s1h", "s2a", "s2b", "vanilla", "p2static", "a3g", "a3h")}
    E.export_onnx(s1g, (x0,), P32["s1g"], ["image"], ["score", *FEATS_A])
    E.export_onnx(s1h, (x0, sa0), P32["s1h"], ["image", "small_area"], ["score", *FEATS_A])
    E.export_onnx(s2a, (p3, p2, p4, p5), P32["s2a"], list(FEATS_A), ["y_A"])
    E.export_onnx(s2b, (p3, p4, p5), P32["s2b"], list(FEATS_B), ["y_B"])
    for k, mod in statics.items():
        E.export_onnx(mod, (x0,), P32[k], ["image"], ["y"])
    build_if_v2(P32["s1g"], P32["s2a"], P32["s2b"], P32["a3g"], n_a, n_b)
    build_if_v2(P32["s1h"], P32["s2a"], P32["s2b"], P32["a3h"], n_a, n_b)
    paths = {"fp32": P32}
    mods = dict(s1g=s1g, s1h=s1h, s2a=s2a, s2b=s2b, **statics, n_a=n_a, n_b=n_b)
    if "fp16" in precisions:
        m16 = X._half_copy(model, device)
        s1g16, s1h16 = S1Gate(m16, gate["w"], gate["b"]).to(device).eval(), S1Heur(m16).to(device).eval()
        _, s2a16, s2b16 = E.make_stages(m16)
        x, sa = x0.to(device), sa0.to(device)
        with torch.no_grad():
            _, q3, q2, q4, q5 = s1g16(x); s1h16(x, sa)
            h = [t.half() for t in (q3, q2, q4, q5)]
            s2a16(*h); s2b16(h[0], h[2], h[3])                      # anchors Detect (half) dihitung di luar tracing
        P16 = {k: out / f"{k}_fp16.onnx" for k in ("s1g", "s1h", "s2a", "s2b", "vanilla", "p2static", "a3g", "a3h")}
        E.export_onnx(s1g16, (x,), P16["s1g"], ["image"], ["score", *FEATS_A])
        E.export_onnx(s1h16, (x, sa), P16["s1h"], ["image", "small_area"], ["score", *FEATS_A])
        E.export_onnx(X._IOCast(s2a16), (q3, q2, q4, q5), P16["s2a"], list(FEATS_A), ["y_A"])
        E.export_onnx(X._IOCast(s2b16), (q3, q4, q5), P16["s2b"], list(FEATS_B), ["y_B"])
        for k, src in (("vanilla", van), ("p2static", p2s)):
            mod = X._IOCast(E._StaticOut(X._half_copy(src, device))).eval()
            with torch.no_grad():
                mod(x)
            E.export_onnx(mod, (x,), P16[k], ["image"], ["y"])
        build_if_v2(P16["s1g"], P16["s2a"], P16["s2b"], P16["a3g"], n_a, n_b)
        build_if_v2(P16["s1h"], P16["s2a"], P16["s2b"], P16["a3h"], n_a, n_b)
        paths["fp16"] = P16
        del m16, s1g16, s1h16, s2a16, s2b16
    return paths, mods


def verify_final(P32, mods, loader, n=4, tol=1e-4):
    """ORT (graf FP32) vs PyTorch: skor gate, fitur, Stage 2, statis, A3 v2 untuk semua mode & arah threshold."""
    import onnxruntime as ort
    sess = {k: ort.InferenceSession(str(P32[k]), providers=["CPUExecutionProvider"])
            for k in ("s1g", "s1h", "vanilla", "p2static", "a3g", "a3h")}
    rel = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
    worst, n_a, n_b = {}, mods["n_a"], mods["n_b"]
    up = lambda k, v: worst.__setitem__(k, max(worst.get(k, 0.0), float(v)))
    with torch.no_grad():
        for k, b in enumerate(loader):
            if k >= n:
                break
            x = b["img"].float() / 255.0
            sa = np.array([small_area_of(b["ratio_pad"][0])], np.float32)
            sg, p3, p2, p4, p5 = mods["s1g"](x)
            sh = float(mods["s1h"](x, torch.from_numpy(sa))[0][0])
            o = sess["s1g"].run(None, {"image": x.numpy()})
            up("s1g_skor", abs(float(o[0][0]) - float(sg[0])) / (1.0 + abs(float(sg[0]))))
            up("s1g_fitur", max(rel(a, t.numpy()) for a, t in zip(o[1:], (p3, p2, p4, p5))))
            oh = sess["s1h"].run(None, {"image": x.numpy(), "small_area": sa})
            up("s1h_skor (diskret, informatif)", abs(float(oh[0][0]) - sh))
            ya, yb = mods["s2a"](p3, p2, p4, p5).numpy(), mods["s2b"](p3, p4, p5).numpy()
            ybp = np.concatenate([yb, np.zeros((1, yb.shape[1], n_a - n_b), np.float32)], 2)
            for nm, extra, s_ref in (("a3g", {}, float(o[0][0])), ("a3h", {"small_area": sa}, float(oh[0][0]))):
                for mode, thr, want in ((1.0, 1e9, True), (0.0, -1e9, False), (-1.0, s_ref - 0.5, True), (-1.0, s_ref + 0.5, False)):
                    y, use_a, score = sess[nm].run(None, {"image": x.numpy(), "mode": np.array([mode], np.float32),
                                                          "thr": np.array([thr], np.float32), **extra})
                    assert bool(use_a[0] > 0.5) == want, f"{nm}: rute salah (mode {mode}, thr {thr})"
                    up(f"{nm}_y", rel(y, ya if want else ybp))
                    up(f"{nm}_skor", abs(float(score[0]) - s_ref))
            for nm in ("vanilla", "p2static"):
                up(nm, rel(sess[nm].run(None, {"image": x.numpy()})[0], mods[nm](x).numpy()))
    bad = {k: v for k, v in worst.items() if "informatif" not in k and v > tol}
    assert not bad, f"Selisih ONNX vs PyTorch > {tol}: {bad}"
    assert worst["s1h_skor (diskret, informatif)"] < 1.0, f"skor heuristik ONNX jauh dari PyTorch: {worst}"
    return worst


# =============================================================================
# JOB INFERENSI (TensorRT FP16 / ORT): keluaran per gambar test + latensi per build
# =============================================================================
class _OrtEng:
    """Antarmuka sama dengan engine TensorRT di job (buf, inp, out, bind, run) — untuk uji lokal di CPU."""
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = [i.name for i in self.sess.get_inputs()]
        self.out = [o.name for o in self.sess.get_outputs()]
        self.buf = {i.name: torch.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=torch.float32)
                    for i in self.sess.get_inputs()}
        for nme, r in zip(self.out, self.sess.run(None, {k: self.buf[k].numpy() for k in self.inp})):
            self.buf[nme] = torch.from_numpy(np.array(r, np.float32))

    def bind(self, nme, t):
        self.buf[nme] = t

    def run(self, s=None):
        for nme, r in zip(self.out, self.sess.run(None, {k: self.buf[k].numpy() for k in self.inp})):
            self.buf[nme].copy_(torch.from_numpy(np.asarray(r, np.float32)))


def b2_thresholds(scores, rho, window, min_hist, thr0):
    """Threshold B2 untuk tiap posisi aliran (hanya skor MASA LALU). float32 agar sama dengan input engine."""
    s = np.asarray(scores, np.float64); out = np.empty(len(s), np.float32)
    for t in range(len(s)):
        hist = s[max(0, t - window):t]
        out[t] = np.float32(np.quantile(hist, 1 - rho)) if len(hist) >= min_hist else np.float32(thr0)
    return out


class SortedWindowQuantile:
    """Kuantil jendela bergulir dengan daftar terurut (bisect): nilai IDENTIK dengan np.quantile (interpolasi linear)
    pada jendela yang sama, tanpa membuat array baru per gambar."""
    def __init__(self, window):
        import bisect
        self._b, self.window, self.h, self.s = bisect, window, collections.deque(), []

    def push(self, x):
        if len(self.h) == self.window:
            old = self.h.popleft(); del self.s[self._b.bisect_left(self.s, old)]
        self.h.append(x); self._b.insort(self.s, x)

    def quantile(self, q):
        n = len(self.s); pos = q * (n - 1); lo = int(pos); fr = pos - lo
        return self.s[lo] + (self.s[min(lo + 1, n - 1)] - self.s[lo]) * fr


def b2_host_overhead(window, rho, n=4000):
    """ms per gambar: (np.quantile atas deque = implementasi job terkunci) vs jendela terurut (nilai identik)."""
    xs = np.random.default_rng(0).standard_normal(n + window)
    h = collections.deque(xs[:window], maxlen=window)
    t0 = time.perf_counter()
    for i in range(n):
        np.quantile(np.asarray(h, np.float64), 1 - rho); h.append(xs[window + i])
    t_np = (time.perf_counter() - t0) / n * 1e3
    sw = SortedWindowQuantile(window)
    for x in xs[:window]:
        sw.push(float(x))
    h = collections.deque(xs[:window], maxlen=window)
    for i in range(200):                                       # cek kesamaan nilai
        assert abs(sw.quantile(1 - rho) - np.quantile(np.asarray(h, np.float64), 1 - rho)) < 1e-12
        sw.push(float(xs[window + i])); h.append(xs[window + i])
    t0 = time.perf_counter()
    for i in range(n):
        sw.quantile(1 - rho); sw.push(float(xs[i]))
    return t_np, (time.perf_counter() - t0) / n * 1e3


def b2_decide(scores, rho, window, min_hist, thr0):
    thr = b2_thresholds(scores, rho, window, min_hist, thr0)
    return np.asarray(scores, np.float32) > thr, thr


def job_main(job_path):
    job = json.loads(Path(job_path).read_text())
    T0 = time.perf_counter()
    log = lambda *a: print(f"[job {time.perf_counter() - T0:6.0f}s]", *a, flush=True)
    E.COCO_MAX_DETS = list(job["coco_max_dets"])
    use_trt = job["backend"] == "trt"
    res = {"backend": job["backend"], "build_s": {}, "latency_builds": {}, "parity": {}, "prof_activation": {}}
    if use_trt:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        dev = torch.device("cuda:0"); torch.cuda.init()
        T2T = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
               trt.int64: torch.int64, trt.bool: torch.bool}

        def build(onnx_path, engine_path):
            builder = trt.Builder(logger)
            flags = 0
            if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
                flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)   # presisi = tipe di ONNX
            net = builder.create_network(flags); parser = trt.OnnxParser(net, logger)
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if job["precision"] == "fp16" and not flags and hasattr(trt.BuilderFlag, "FP16"):
                cfg.set_flag(trt.BuilderFlag.FP16)
            t = time.perf_counter(); eb = builder.build_serialized_network(net, cfg)
            assert eb is not None, f"gagal build {onnx_path}"
            Path(engine_path).write_bytes(eb)
            return engine_path, time.perf_counter() - t

        class Eng:
            def __init__(self, path):
                self.rt = trt.Runtime(logger)
                self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
                self.ctx = self.engine.create_execution_context()
                self.inp, self.out, self.buf = [], [], {}
                for i in range(self.engine.num_io_tensors):
                    nme = self.engine.get_tensor_name(i)
                    (self.inp if self.engine.get_tensor_mode(nme) == trt.TensorIOMode.INPUT else self.out).append(nme)
                for nme in self.inp + self.out:
                    shape = tuple(self.engine.get_tensor_shape(nme))
                    if nme in self.inp:
                        self.ctx.set_input_shape(nme, shape)
                    self.buf[nme] = torch.zeros(shape, dtype=T2T[self.engine.get_tensor_dtype(nme)], device=dev)
                    self.ctx.set_tensor_address(nme, self.buf[nme].data_ptr())

            def bind(self, nme, t):
                self.buf[nme] = t; self.ctx.set_tensor_address(nme, t.data_ptr())

            def run(self, s):
                assert self.ctx.execute_async_v3(s.cuda_stream), "execute_async_v3 gagal"

        s = torch.cuda.Stream()
        ctx = lambda: torch.cuda.stream(s)
        sync, dsync = s.synchronize, torch.cuda.synchronize
        pin = lambda t: torch.zeros(t.shape, dtype=t.dtype).pin_memory()
        res.update(trt_version=trt.__version__, gpu=torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu"); s = None
        build = lambda o, e: (o, 0.0)
        Eng = _OrtEng
        ctx = contextlib.nullcontext
        sync = dsync = (lambda: None)
        pin = lambda t: torch.zeros(t.shape, dtype=t.dtype)
        res.update(trt_version=None, gpu="cpu (ONNX Runtime, uji lokal)")

    from ultralytics.data.utils import check_det_dataset
    DATA = check_det_dataset(job["data_yaml"])
    loader = E.make_loader(DATA, job["split"], 1, job["imgsz"], workers=job.get("workers", 2), pin_memory=False)
    rho, W, MINH = job["rho"], job["b2_window"], job["b2_min_hist"]
    thr_g0, thr_h0 = np.float32(job["thr_val_gate"]), np.float32(job["thr_val_heur"])
    names = list(job["onnx"])
    Path(job["engine_dir"]).mkdir(parents=True, exist_ok=True)

    def load_engines(bi):
        eng = {}
        for k in names:
            p = job["onnx"][k]
            try:
                path, secs = build(p, str(Path(job["engine_dir"]) / f"{Path(p).stem}_b{bi}.engine"))
                e_ = Eng(path)
            except Exception as ex:                       # heuristik bersifat pembanding: gagal build tidak menghentikan job
                if k != "a3h":
                    raise
                res.setdefault("build_error", {})[f"b{bi}|{k}"] = f"{type(ex).__name__}: {str(ex)[:500]}"
                log("A3 heuristik GAGAL dibangun:", res["build_error"][f"b{bi}|{k}"]); continue
            res["build_s"][f"b{bi}|{k}"] = secs; eng[k] = e_; log("engine", f"build {bi}", k, f"{secs:.0f} s")
        return eng

    def make_ops(eng):
        H = {"img": pin(eng["vanilla"].buf["image"]), "sa": pin(torch.zeros(1)),
             "thr": {k: pin(torch.zeros(1)) for k in ("a3g", "a3h")}}
        modes = {mv: torch.full((1,), mv, dtype=torch.float32, device=dev) for mv in (-1.0, 0.0, 1.0)}
        ops = {}
        for k in ("vanilla", "p2static"):
            e = eng[k]; ho = pin(e.buf[e.out[0]]); H[k] = ho

            def f(e=e, ho=ho):
                e.buf["image"].copy_(H["img"], non_blocking=True); e.run(s)
                ho.copy_(e.buf[e.out[0]], non_blocking=True); sync()
            ops[k] = f
        for k in [k_ for k_ in ("a3g", "a3h") if k_ in eng]:
            e = eng[k]; hy, hu, hs = pin(e.buf["y"]), pin(e.buf["use_a"]), pin(e.buf["score"]); H[k] = (hy, hu, hs)
            has_sa = "small_area" in e.inp
            for mv in (-1.0, 0.0, 1.0):
                def f(e=e, mv=mv, hy=hy, hu=hu, hs=hs, has_sa=has_sa, ht=H["thr"][k]):
                    e.bind("mode", modes[mv])
                    e.buf["image"].copy_(H["img"], non_blocking=True); e.buf["thr"].copy_(ht, non_blocking=True)
                    if has_sa:
                        e.buf["small_area"].copy_(H["sa"], non_blocking=True)
                    e.run(s)
                    hy.copy_(e.buf["y"], non_blocking=True); hu.copy_(e.buf["use_a"], non_blocking=True)
                    hs.copy_(e.buf["score"], non_blocking=True); sync()
                ops[(k, mv)] = f
        d_img = torch.zeros_like(eng["vanilla"].buf["image"])

        def fh():
            d_img.copy_(H["img"], non_blocking=True); sync()
        ops["h2d"] = fh                                              # untuk kaskade: H2D gambar tidak dihitung dua kali
        e = eng["s1g"]; hs1 = pin(e.buf["score"]); H["s1g"] = hs1

        def f1(e=e):
            e.buf["image"].copy_(H["img"], non_blocking=True); e.run(s); hs1.copy_(e.buf["score"], non_blocking=True); sync()
        ops["s1g"] = f1
        return ops, H

    def call(op):
        dsync()
        with ctx():
            op()

    nms = lambda y, rp, osh: P._to_dets(y, rp, osh, job["conf"], job["iou"], job["max_det"])

    def acc_pass(eng, ops, H):
        D = {k: [] for k in ("V", "P2S", "A", "B")}
        rec = collections.defaultdict(list); hist = collections.deque(maxlen=W)
        par = {"y_router_vs_paksa_maks": 0.0, "skor_antar_mode_maks": 0.0, "use_a_vs_host_beda": 0}
        for i, b in enumerate(loader):
            H["img"].copy_(b["img"].float() / 255.0)
            rp, osh = b["ratio_pad"][0], b["ori_shape"][0]
            H["sa"].fill_(small_area_of(rp))
            call(ops["vanilla"]); D["V"].append(nms(eng["vanilla"].buf[eng["vanilla"].out[0]], rp, osh))
            call(ops["p2static"]); D["P2S"].append(nms(eng["p2static"].buf[eng["p2static"].out[0]], rp, osh))
            H["thr"]["a3g"].fill_(float(thr_g0))
            call(ops[("a3g", 1.0)]); yA = eng["a3g"].buf["y"].clone(); sA = float(H["a3g"][2][0])
            call(ops[("a3g", 0.0)]); yB = eng["a3g"].buf["y"].clone(); sB = float(H["a3g"][2][0])
            D["A"].append(nms(yA, rp, osh)); D["B"].append(nms(yB, rp, osh))
            thr = np.float32(np.quantile(np.asarray(hist, np.float64), 1 - rho)) if len(hist) >= MINH else thr_g0
            H["thr"]["a3g"].fill_(float(thr))
            call(ops[("a3g", -1.0)])                                   # sistem yang di-deploy: B2 + A3 (thr = input)
            s_auto, use = float(H["a3g"][2][0]), bool(float(H["a3g"][1][0]) > 0.5)
            par["y_router_vs_paksa_maks"] = max(par["y_router_vs_paksa_maks"], float((eng["a3g"].buf["y"] - (yA if use else yB)).abs().max()))
            par["skor_antar_mode_maks"] = max(par["skor_antar_mode_maks"], abs(s_auto - sA), abs(s_auto - sB))
            par["use_a_vs_host_beda"] += int(use != bool(np.float32(s_auto) > thr))
            hist.append(s_auto)
            if "a3h" in eng:
                H["thr"]["a3h"].fill_(float(thr_h0))
                call(ops[("a3h", 0.0)])                                # skor heuristik (rute B: murah); keputusan B2 offline
                rec["score_h"].append(float(H["a3h"][2][0]))
            rec["file"].append(Path(b["im_file"][0]).name); rec["score_g"].append(s_auto); rec["thr_g"].append(float(thr))
            rec["use_g"].append(use)
            if (i + 1) % 250 == 0:
                log(f"akurasi: {i + 1} gambar | aktivasi sejauh ini {np.mean(rec['use_g']):.3f}")
        par["n"] = len(rec["file"])
        return D, dict(rec), par

    def lat_pass(ops, H, imgs, bi):
        H["thr"]["a3g"].fill_(float(thr_g0)); H["thr"]["a3h"].fill_(float(thr_h0))
        V = {"Vanilla": ops["vanilla"], "P2-Static": ops["p2static"], "Stage 1 + gate (s1g)": ops["s1g"], "H2D gambar": ops["h2d"],
             "A3 gate rute A": ops[("a3g", 1.0)], "A3 gate rute B": ops[("a3g", 0.0)],
             "A3 gate router (thr val)": ops[("a3g", -1.0)]}
        if ("a3h", 1.0) in ops:
            V.update({"A3 heuristik rute A": ops[("a3h", 1.0)], "A3 heuristik rute B": ops[("a3h", 0.0)],
                      "A3 heuristik router (thr val)": ops[("a3h", -1.0)]})

        def run_fn(fn, im, sa):
            H["img"].copy_(im.float() / 255.0); H["sa"].fill_(sa); dsync()          # konversi di luar timer
            w0 = time.perf_counter()
            with ctx():
                fn()
            return (time.perf_counter() - w0) * 1e3
        for im, sa in imgs[:job["n_warmup"]]:
            for fn in V.values():
                run_fn(fn, im, sa)
        acts = {k: [] for k in ("A3 gate router (thr val)", "A3 heuristik router (thr val)") if k in V}
        rec = {k: [] for k in V}; order = list(V)
        for rep in range(job["n_reps"]):
            for im, sa in imgs:
                for k in (order if rep % 2 == 0 else order[::-1]):
                    rec[k].append(run_fn(V[k], im, sa))
                    if k in acts and rep == 0:
                        acts[k].append(float(H["a3g" if "gate" in k else "a3h"][1][0]) > 0.5)
        for k, v in rec.items():
            a = np.asarray(v)
            res["latency_builds"][f"b{bi}|{k}"] = dict(mean=a.mean(), std=a.std(), p50=np.percentile(a, 50),
                                                         p95=np.percentile(a, 95), n=int(a.size))
        res["prof_activation"][f"b{bi}"] = {k: float(np.mean(v)) for k, v in acts.items()}

    imgs = []
    for k, b in enumerate(loader):
        if k >= job["n_prof_images"]:
            break
        imgs.append((b["img"].clone(), small_area_of(b["ratio_pad"][0])))            # uint8 (hemat RAM); float di luar timer
    n_builds = int(job["n_builds"]) if use_trt else 1
    for bi in range(n_builds):
        eng = load_engines(bi); ops, H = make_ops(eng)
        if bi == 0:
            D, rec, par = acc_pass(eng, ops, H)
            res["parity"] = par; res["files"] = rec["file"]
            with open(job["dets_out"], "wb") as f:
                pickle.dump({"dets": D, "rec": rec}, f)
            Path(job["out_json"]).write_text(json.dumps(res, default=float, indent=1))
            log("akurasi selesai | paritas", par)
        lat_pass(ops, H, imgs, bi)
        Path(job["out_json"]).write_text(json.dumps(res, default=float, indent=1))
        log(f"latensi build {bi} selesai")
        del eng, ops, H; gc.collect()
        if use_trt:
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    res["b2_host_ms"], res["b2_host_ms_efisien"] = b2_host_overhead(W, rho)
    res["engine_bytes"] = {Path(f).name: Path(f).stat().st_size for f in sorted(Path(job["engine_dir"]).glob("*_b0.engine"))}
    Path(job["out_json"]).write_text(json.dumps(res, default=float, indent=1))
    log("selesai")


def run_subprocess(cmd, cwd, timeout_s, tail_path, log=print):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    killed = []
    timer = threading.Timer(timeout_s, lambda: (killed.append(1), proc.kill())); timer.start()
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True); tail = (tail + [line])[-80:]
    rc = proc.wait(); timer.cancel()
    Path(tail_path).write_text("".join(tail))
    assert not killed, f"subprocess TIMEOUT {timeout_s} s; lihat {tail_path}"
    assert rc == 0, f"subprocess gagal (returncode {rc}); lihat {tail_path}"


# =============================================================================
# BOOTSTRAP BERPASANGAN (indeks resample sama untuk semua routing; paralel lewat fork)
# =============================================================================
class UnionBoot:
    """
    Bootstrap cepat untuk routing per gambar antara dua sumber deteksi (A/B) — pengganti CocoMix.metrics di bootstrap.
    Deteksi kedua sumber diurutkan SEKALI per (kategori, area) dengan kunci (−skor, gambar, sumber, posisi) = urutan
    COCOeval.accumulate untuk routing apa pun. Untuk resample dengan multiplisitas c_i, setiap deteksi aktif menjadi satu
    titik berbobot c_i: presisi terinterpolasi di akhir kelompok salinan identik sama dengan ekspansi penuh (TP: presisi
    naik di dalam kelompok; FP: tidak pernah menjadi maksimum), sehingga argsort/concatenate per resample tidak perlu.
    Hasil pada himpunan penuh == COCOeval persis; pada resample berulang hanya urutan seri skor antar-gambar yang bisa
    berbeda dari CocoBootstrap (urutan indeks resample), efeknya ~1e-6.
    """
    AREAS = ("all", "small")

    def __init__(self, bA, bB):
        self.I, self.T, self.R, self.K, self.iou50 = bA.I, bA.T, np.asarray(bA.R), bA.K, bA.iou50
        assert bB.I == self.I and bB.K == self.K
        self.cells = {}
        for k in range(self.K):
            for area in self.AREAS:
                sc, tp, fp, img, src, pos = [], [], [], [], [], []
                npig = np.zeros(self.I, np.int64)
                for s_i, b in enumerate((bA, bB)):
                    for i, c in enumerate(b.cells[(k, area)]):
                        if c is None:
                            continue
                        s_, dtm, dtig, n_ = c
                        npig[i] = n_                                   # GT sama untuk kedua sumber
                        if len(s_):
                            sc.append(s_); tp.append(dtm & ~dtig); fp.append(~dtm & ~dtig)
                            img.append(np.full(len(s_), i, np.int64)); src.append(np.full(len(s_), s_i, np.int8))
                            pos.append(np.arange(len(s_)))
                if sc:
                    sc, img, src, pos = map(np.concatenate, (sc, img, src, pos))
                    tp, fp = np.concatenate(tp, 1), np.concatenate(fp, 1)
                    o = np.lexsort((pos, src, img, -sc))
                    o = o[(tp[:, o] | fp[:, o]).any(0)]                # diabaikan di semua IoU -> tanpa pengaruh (eksak)
                    self.cells[(k, area)] = (tp[:, o], fp[:, o], img[o], src[o], npig)
                else:
                    self.cells[(k, area)] = (np.zeros((self.T, 0), bool), np.zeros((self.T, 0), bool),
                                             np.zeros(0, np.int64), np.zeros(0, np.int8), npig)

    def metrics(self, use_a_sorted, idx):
        c = np.bincount(idx, minlength=self.I)
        P_ = {"all": [], "small": []}
        for (k, area), (tp, fp, img, src, npig) in self.cells.items():
            npg = int((c * npig).sum())
            if npg == 0:
                continue
            q = np.zeros((self.T, len(self.R)))
            if len(img):
                w = c[img] * ((src == 0) == use_a_sorted[img])
                sel = np.flatnonzero(w)
                if len(sel):
                    ww = w[sel].astype(np.int32)
                    tpc = np.cumsum(tp[:, sel] * ww, axis=1, dtype=np.int32).astype(float)   # bilangan bulat: eksak
                    fpc = np.cumsum(fp[:, sel] * ww, axis=1, dtype=np.int32).astype(float)
                    rc, pr = tpc / npg, tpc / (fpc + tpc + np.spacing(1))
                    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                    nd = len(sel)
                    for t in range(self.T):
                        ii = np.searchsorted(rc[t], self.R, side="left"); ok = ii < nd
                        q[t, ok] = pr[t, ii[ok]]
            P_[area].append(q)
        m = {}
        if P_["all"]:
            Pa = np.stack(P_["all"]); m["mAP50-95"] = float(Pa.mean()); m["mAP50"] = float(Pa[:, self.iou50].mean())
        else:
            m["mAP50-95"] = m["mAP50"] = np.nan
        m["mAP_small"] = float(np.stack(P_["small"]).mean()) if P_["small"] else np.nan
        return m


_BCTX = {}


def _boot_rows(rows):
    mixes, idx = _BCTX["mixes"], _BCTX["idx"]
    return [{k: U.metrics(u, idx[r]) for k, (U, u) in mixes.items()} for r in rows]


def boot_engine(mixes, n_boot, seed, n_jobs=1):
    """
    mixes: {nama: (UnionBoot, use_a dalam urutan imgIds COCO)}.
    Returns (titik {nama: metrik} pada himpunan penuh, sampel {nama: {metrik: array n_boot}}).
    """
    U0 = next(iter(mixes.values()))[0]
    I = U0.I
    assert all(U.I == I for U, _ in mixes.values())
    idx = np.random.default_rng(seed).integers(0, I, (n_boot, I))
    t1 = time.perf_counter()
    point = {k: U.metrics(u, np.arange(I)) for k, (U, u) in mixes.items()}
    t1 = time.perf_counter() - t1                                     # ≈ waktu satu resample untuk semua routing
    _BCTX.update(mixes=mixes, idx=idx)
    chunks = [c for c in np.array_split(np.arange(n_boot), max(1, 4 * n_jobs)) if len(c)]
    parts = None
    try:
        if n_jobs > 1:
            timeout = max(300.0, 4.0 * t1 * n_boot / n_jobs)            # jaga-jaga deadlock fork -> fallback serial
            pool = mp.get_context("fork").Pool(n_jobs)
            try:
                parts = pool.map_async(_boot_rows, chunks).get(timeout=timeout)
                pool.close()
            except mp.TimeoutError:
                print(f"⚠ bootstrap paralel melewati {timeout:.0f} s -> dihentikan, diulang serial", flush=True)
            finally:
                pool.terminate(); pool.join()
        if parts is None:
            parts = [_boot_rows(c) for c in chunks]
    finally:
        _BCTX.clear()
    flat = [d for p_ in parts for d in p_]
    samples = {k: {m: np.array([d[k][m] for d in flat], float) for m in METRICS} for k in mixes}
    return point, samples


def _pval(s):
    return float(min(1.0, max(2 * min((s <= 0).mean(), (s >= 0).mean()), 1.0 / len(s))))


def boot_tests(point, samples, tests):
    """tests: [(label, nama_a, [nama_b...])] -> DataFrame selisih a − rata-rata(b) + CI persentil 95%."""
    rows = []
    for label, a, bs in tests:
        if a not in point or not bs or any(b not in point for b in bs):
            continue
        for m in METRICS:
            pt = point[a][m] - np.nanmean([point[b][m] for b in bs])
            sm = samples[a][m] - np.nanmean(np.stack([samples[b][m] for b in bs]), 0)
            sm = sm[np.isfinite(sm)]
            lo, hi = np.percentile(sm, [2.5, 97.5])
            rows.append(dict(uji=label, metric=m, diff=pt, ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm), n_boot=len(sm)))
    return pd.DataFrame(rows)


def partial_spearman_p(x, y, controls):
    """ρ Spearman parsial (rank kontrol diregresikan keluar dari rank x dan y) + p dua sisi (uji t, df = n − 2 − k)."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    R = lambda v: pd.Series(np.asarray(v, float)).rank().values
    Xc = np.c_[np.ones(len(x)), *[R(c) for c in controls]]
    res = lambda v: R(v) - Xc @ np.linalg.lstsq(Xc, R(v), rcond=None)[0]
    rx, ry = res(x), res(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(rx, ry)[0, 1]); df = len(x) - 2 - len(controls)
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df)), len(x)


# =============================================================================
# TAHAP A (PyTorch): cache, fit gate (train), threshold (val), cache test, ekspor + verifikasi ONNX
# =============================================================================
def find_ckpt(archive, ckpt_name, log=print):
    arch = Path(archive)
    if not arch.exists():                                   # nama pemilik dataset bisa beda (agungmaugi / arungmaugi)
        alt = sorted(glob.glob(f"/kaggle/input/datasets/*/{arch.name}")) + sorted(glob.glob(f"/kaggle/input/{arch.name}"))
        assert len(alt) == 1, f"arsip {arch} tidak ada; kandidat dengan nama sama: {alt}"
        log(f"arsip dipindah ke {alt[0]}"); arch = Path(alt[0])
    hits = sorted(glob.glob(str(arch / "**" / "weights" / ckpt_name), recursive=True))
    assert len(hits) == 1, f"{ckpt_name} ditemukan {len(hits)}x di {arch}: {hits}"
    return hits[0]


def check_static(path, nc, log=print):
    p = Path(path)
    assert p.exists(), (f"checkpoint baseline tidak ada: {p}. Kandidat best.pt di /kaggle/input: "
                        f"{sorted(glob.glob('/kaggle/input/**/best.pt', recursive=True))[:20]}")
    return str(p)


def stage_a(spec, cfg, out, device, log=print):
    from ultralytics.data.utils import check_det_dataset
    t0 = time.perf_counter()
    E.COCO_MAX_DETS = list(spec["coco_max_dets"])
    DATA = check_det_dataset(spec["data_yaml"]); NAMES = DATA["names"]
    L = {s: E.make_loader(DATA, s, 1, cfg["imgsz"]) for s in ("train", "val", "test")}
    N = {s: len(ld.dataset) for s, ld in L.items()}
    if cfg.get("n_train"):
        N["train"] = min(N["train"], cfg["n_train"])
    ckpt = find_ckpt(spec["archive"], cfg["ckpt_name"], log)
    model, minfo = E.load_dualbranch(ckpt, cfg["model_yaml"], DATA["nc"], NAMES, device)
    assert minfo["train_args_seed"] in (None, spec["seed"]), f"seed checkpoint {minfo['train_args_seed']} != {spec['seed']}"
    E.check_stage_equivalence(model, E.make_stages(model), L["test"], device)
    van = E.load_static(check_static(spec["vanilla_ckpt"], DATA["nc"]), device)
    p2s = E.load_static(check_static(spec["p2static_ckpt"], DATA["nc"]), device)
    for nm, m_ in (("Vanilla", van), ("P2-Static", p2s)):
        assert int(m_.model[-1].nc) == int(DATA["nc"]), f"{nm}: nc {m_.model[-1].nc} != dataset {DATA['nc']} (baseline dataset lain?)"
    statics = (van, p2s)
    kw = dict(conf=cfg["conf"], iou=cfg["iou"], max_det=spec["max_det"], log=log)
    log(f"train: cache M0{' + Vanilla + P2-Static' if cfg['cascade'] else ''} ({N['train']} gambar) ...")
    Ctr = cache_split(model, L["train"], device, dets=True, heur=False, statics=statics if cfg["cascade"] else None,
                      max_images=N["train"], **kw)
    gt_tr = P.subset_coco_gt(L["train"].dataset, NAMES, out / "gt_train.json", N["train"])
    R_tr, _, _ = router_of(gt_tr, Ctr["detA"], Ctr["detB"], Ctr["image_id"])
    TG_tr = R_tr.targets(); del R_tr; gc.collect()
    gate = fit_gate(Ctr["feat"], TG_tr["soft_small"], cfg["gate_C"])
    label_cost = dict(n_gambar_train=int(N["train"]), n_kotak_gt_train=int(TG_tr["n_gt_all"].sum()),
                      n_kotak_gt_kecil_train=int(TG_tr["n_gt_small"].sum()),
                      frac_gambar_berobjek_kecil=float((TG_tr["n_gt_small"] > 0).mean()))
    log(f"gate: {gate['info']} | biaya label: {label_cost}")
    casc = None
    if cfg["cascade"]:
        R_c, _, _ = router_of(gt_tr, Ctr["detP"], Ctr["detV"], Ctr["image_id"])
        TG_c = R_c.targets(); del R_c; gc.collect()
        casc = fit_gate(Ctr["featV"], TG_c["soft_small"], cfg["gate_C"])
        log(f"gate kaskade (fitur trunk Vanilla, target Δ P2-Static − Vanilla): {casc['info']}")
    del Ctr, gt_tr; gc.collect()

    log(f"val: skor gate & heuristik ({N['val']} gambar, tanpa NMS) ...")
    Cva = cache_split(model, L["val"], device, dets=False, heur=True, statics=None, **kw)
    sg_va = Cva["feat"] @ gate["w"] + gate["b"]
    thr_g = float(np.quantile(sg_va, 1 - cfg["rho"])); thr_h = float(np.quantile(Cva["heur"], 1 - cfg["rho"]))
    log(f"thr val: gate {thr_g:+.4f} | heuristik {thr_h:.4f} | nilai heuristik unik di val {len(np.unique(Cva['heur']))}/{N['val']}")

    log(f"test: cache M0 + Vanilla + P2-Static ({N['test']} gambar) ...")
    Cte = cache_split(model, L["test"], device, dets=True, heur=True, statics=statics, **kw)
    gt_te_path = out / "gt_test.json"; E.build_coco_gt(L["test"].dataset, NAMES, gt_te_path)
    test = {k: Cte[k] for k in ("file", "image_id", "detA", "detB", "detV", "detP", "heur", "heur_n_all", "heur_n_small", "small_area")}
    test["score_g32"] = Cte["feat"] @ gate["w"] + gate["b"]
    test["score_c32"] = Cte["featV"] @ casc["w"] + casc["b"] if casc else None

    log("GFLOPs + ekspor ONNX (FP32 untuk verifikasi, FP16 untuk TensorRT) ...")
    x0 = next(iter(L["test"]))["img"].float() / 255.0
    mc = E.cpu_copy(model); trunk = G.Stage1PostHoc(mc, nn.Identity()).eval(); _, s2a, s2b = E.make_stages(mc)
    with torch.no_grad():
        p3, p2, p4, p5 = trunk.trunk(x0)
    dens = P3Density(list(mc.model)[33]).eval(); sa0 = torch.tensor([SMALL_PX ** 2 * 0.25])
    gw_, gb_ = torch.as_tensor(gate["w"], dtype=torch.float32), float(gate["b"])
    fl = {"trunk 0–15": A3.gflops(trunk.trunk, (x0,)),
          "gate post-hoc (pooling + Gemm)": A3.gflops(lambda *a: pool4(*a) @ gw_ + gb_, (p3, p2, p4, p5)),
          "heuristik P3 (head P3 Detect_B + hitung)": A3.gflops(dens, (p3, sa0)),
          "Stage 2A": A3.gflops(s2a, (p3, p2, p4, p5)), "Stage 2B": A3.gflops(s2b, (p3, p4, p5)),
          "Vanilla": A3.gflops(E._StaticOut(E.cpu_copy(van)).eval(), (x0,)),
          "P2-Static": A3.gflops(E._StaticOut(E.cpu_copy(p2s)).eval(), (x0,))}
    log("GFLOPs:", {k: round(v, 3) for k, v in fl.items()})
    del mc, trunk, s2a, s2b, dens
    precs = ("fp16",) if (cfg["precision"] == "fp16" and (torch.cuda.is_available() or cfg.get("allow_cpu_fp16"))) else ()
    paths, mods = export_final(model, van, p2s, gate, x0, out, device, precs)
    ver = verify_final(paths["fp32"], mods, L["test"], n=cfg["n_onnx_check"])
    log("verifikasi ONNX Runtime (selisih maks):", {k: f"{v:.1e}" for k, v in ver.items()})
    A = dict(spec=spec, minfo=minfo, gate=gate, casc=casc, thr_val_gate=thr_g, thr_val_heur=thr_h,
             val=dict(score_g32=sg_va, score_h32=Cva["heur"], file=Cva["file"]), test=test, label_cost=label_cost,
             gflops=fl, onnx={p: {k: str(v) for k, v in d.items()} for p, d in paths.items()}, onnx_verif=ver,
             n_a=mods["n_a"], n_b=mods["n_b"], gt_test=str(gt_te_path), stage_a_min=(time.perf_counter() - t0) / 60)
    del model, van, p2s, mods, Cva, Cte; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return A


# =============================================================================
# TAHAP C: analisis (semua dari keluaran job + cache test)
# =============================================================================
def latency_summary(res):
    rows = []
    for k, v in res["latency_builds"].items():
        b, name = k.split("|", 1)
        rows.append(dict(build=b, varian=name, mean=v["mean"], p50=v["p50"], p95=v["p95"], std=v["std"]))
    L = pd.DataFrame(rows)
    T = L.groupby("varian")["mean"].agg(["mean", "std", "min", "max", "count"]).rename(columns={"count": "n_build"})
    T["p50 (rata-rata build)"] = L.groupby("varian")["p50"].mean()
    T["p95 (rata-rata build)"] = L.groupby("varian")["p95"].mean()
    return L, T


def analyze(A, res, J, cfg, out, log=print):
    from pycocotools.coco import COCO
    spec, rho, n_draws, seed = A["spec"], cfg["rho"], cfg["n_draws"], int(spec_seed(A))
    E.COCO_MAX_DETS = list(spec["coco_max_dets"])
    te, rec, D = A["test"], J["rec"], J["dets"]
    assert rec["file"] == te["file"], "urutan gambar job != cache test"
    ids = te["image_id"]; n = len(ids)
    gt_path = A["gt_test"] if Path(A["gt_test"]).exists() else Path(out) / "gt_test.json"   # sesi lain: dari reuse_dir
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))

    # ---- COCO per keluaran (FP16 deployment) + router CocoMix
    R_AB, st_A, st_B = router_of(gt, D["A"], D["B"], ids)
    R_VP, st_P, st_V = router_of(gt, D["P2S"], D["V"], ids)
    TG = R_AB.targets()

    # ---- latensi
    LB, LT = latency_summary(res)
    T = LT["mean"].to_dict(); b2ms = float(res.get("b2_host_ms", 0.0))
    TA, TB, TV, TP = T["A3 gate rute A"], T["A3 gate rute B"], T["Vanilla"], T["P2-Static"]
    THA, THB = T.get("A3 heuristik rute A", np.nan), T.get("A3 heuristik rute B", np.nan)
    TS1, TH2D = T["Stage 1 + gate (s1g)"], T.get("H2D gambar", 0.0)

    # ---- keputusan
    heur_src = "TensorRT" if rec.get("score_h") else "PyTorch FP32 (engine A3 heuristik gagal dibangun)"
    sg = np.asarray(rec["score_g"], float)
    sh = np.asarray(rec["score_h"], float) if rec.get("score_h") else np.asarray(te["heur"], float)
    d_dep = np.asarray(rec["use_g"], bool)
    d_off, _ = b2_decide(sg, rho, cfg["b2_window"], cfg["b2_min_hist"], A["thr_val_gate"])
    d_heur, _ = b2_decide(sh, rho, cfg["b2_window"], cfg["b2_min_hist"], A["thr_val_heur"])
    d_top_g, d_top_h = P.top_rho(sg, rho), P.top_rho(sh, rho)
    act, act_h = float(d_dep.mean()), float(d_heur.mean())
    T_router = act * TA + (1 - act) * TB + b2ms
    T_heur = act_h * THA + (1 - act_h) * THB + b2ms
    f_mix = float(np.clip((T_router - TV) / (TP - TV), 0, 1))
    f_mix_h = float(np.clip((T_heur - TV) / (TP - TV), 0, 1)) if np.isfinite(T_heur) else np.nan
    TcA, TcB = TS1 + TP - TH2D, TV                   # kaskade: rute A = trunk Vanilla + gate + P2-Static utuh (gambar sudah di GPU)
    rho_c = float(np.clip((T_router - TcB) / (TcA - TcB), 0, 1))
    # sensitivitas (pasca-hasil s42): B2 dengan jendela terurut -> keputusan IDENTIK, hanya overhead host berbeda
    if "b2_host_ms_efisien" in res:
        b2e, b2e_src = float(res["b2_host_ms_efisien"]), "job"
    else:                                            # job lama: ukur ulang kedua implementasi di proses ini, skala ke angka job
        t_np, t_eff = b2_host_overhead(cfg["b2_window"], rho)
        b2e, b2e_src = b2ms * t_eff / t_np, f"diukur ulang di analisis (np {t_np * 1e3:.1f} µs, efisien {t_eff * 1e3:.1f} µs), diskalakan"
    T_router_e = act * TA + (1 - act) * TB + b2e
    f_mix_e = float(np.clip((T_router_e - TV) / (TP - TV), 0, 1))
    rho_c_e = float(np.clip((T_router_e - TcB) / (TcA - TcB), 0, 1))
    lat = dict(T_router=T_router, T_heur=T_heur, T_V=TV, T_P2S=TP, T_A3_A=TA, T_A3_B=TB, T_A3h_A=THA, T_A3h_B=THB,
               T_s1g=TS1, T_h2d=TH2D, b2_host_ms=b2ms, f_mix=f_mix, f_mix_heur=f_mix_h, rho_kaskade=rho_c,
               T_kaskade_A=TcA, T_kaskade_B=TcB, b2_host_ms_efisien=b2e, b2_efisien_sumber=b2e_src,
               T_router_b2_efisien=T_router_e, f_mix_b2_efisien=f_mix_e, rho_kaskade_b2_efisien=rho_c_e,
               H5_rute_A=TA - TP, H5_rute_B=TB - TV, heur_tambahan_rute_A=THA - TA, heur_tambahan_rute_B=THB - TB)

    # ---- campuran untuk bootstrap
    k_dep, k_h = int(d_dep.sum()), int(d_heur.sum())
    U_AB, U_VP = UnionBoot(R_AB.bA, R_AB.bB), UnionBoot(R_VP.bA, R_VP.bB)
    ab = lambda m_: (U_AB, R_AB.sorted_mask(m_)); vp = lambda m_: (U_VP, R_VP.sorted_mask(m_))
    mixes = {"A": ab(np.ones(n, bool)), "B": ab(np.zeros(n, bool)), "V": vp(np.zeros(n, bool)), "P2S": vp(np.ones(n, bool)),
             "router": ab(d_dep), "router_top": ab(d_top_g), "heur": ab(d_heur), "heur_top": ab(d_top_h)}
    for j, m_ in enumerate(P.random_masks(n, k_dep / n, n_draws, seed + 11)):
        mixes[f"acak_router_{j}"] = ab(m_)
    n2 = cfg.get("n_draws_sekunder", 3)                       # pembanding sekunder: undian lebih sedikit (waktu)
    for j, m_ in enumerate(P.random_masks(n, rho, n2, seed + 13)):
        mixes[f"acak_top_{j}"] = ab(m_)
    for j, m_ in enumerate(P.random_masks(n, k_h / n, n2, seed + 17)):
        mixes[f"acak_heur_{j}"] = ab(m_)
    for j, m_ in enumerate(P.random_masks(n, f_mix, n_draws, seed + 19)):
        mixes[f"campuran_router_{j}"] = vp(m_)
    if np.isfinite(f_mix_h):
        for j, m_ in enumerate(P.random_masks(n, f_mix_h, n2, seed + 23)):
            mixes[f"campuran_heur_{j}"] = vp(m_)
    for j, m_ in enumerate(P.random_masks(n, f_mix_e, n_draws, seed + 19)):
        mixes[f"campuran_b2ef_{j}"] = vp(m_)
    if te.get("score_c32") is not None:
        mixes["kaskade"] = vp(P.top_rho(te["score_c32"], rho_c))
        mixes["kaskade_b2ef"] = vp(P.top_rho(te["score_c32"], rho_c_e))
    grp = lambda p_: [k for k in mixes if k.startswith(p_)]
    tests = [("H2/H7 | router (B2) − acak (aktivasi sama)", "router", grp("acak_router_")),
             ("router (top-ρ) − acak (ρ)", "router_top", grp("acak_top_")),
             ("H6 | router (B2) − campuran acak Vanilla/P2-Static (latensi sama)", "router", grp("campuran_router_")),
             ("router (B2) − Branch A (P2 selalu)", "router", ["A"]),
             ("router (B2) − Branch B (P2 tidak pernah)", "router", ["B"]),
             ("Branch A − P2-Static", "A", ["P2S"]), ("Branch B − Vanilla", "B", ["V"]),
             ("gate − heuristik kepadatan (top-ρ)", "router_top", ["heur_top"]),
             ("gate (B2) − heuristik kepadatan (B2)", "router", ["heur"]),
             ("heuristik (B2) − acak (aktivasi sama)", "heur", grp("acak_heur_")),
             ("heuristik (B2) − campuran Vanilla/P2-Static (latensi heuristik)", "heur", grp("campuran_heur_")),
             ("router (B2) − kaskade bergate Vanilla→P2-Static (latensi sama)", "router", ["kaskade"]),
             (H6_SENS, "router", grp("campuran_b2ef_")),
             ("[sensitivitas, pasca-hasil] router (B2 efisien) − kaskade (latensi sama)", "router", ["kaskade_b2ef"])]
    tb = time.perf_counter()
    log(f"bootstrap: {len(mixes)} routing × {cfg['n_boot']} resample ({cfg['boot_jobs']} proses) ...")
    point, samples = boot_engine(mixes, cfg["n_boot"], cfg["boot_seed"], cfg["boot_jobs"])
    BS = boot_tests(point, samples, tests)
    log(f"bootstrap selesai ({time.perf_counter() - tb:.0f} s)")
    np.savez_compressed(out / "boot_samples.npz", **{f"{k}|{m}": v[m].astype(np.float32) for k, v in samples.items() for m in METRICS})
    (out / "boot_point.json").write_text(json.dumps(point, indent=1, default=float))
    (out / "boot_tests.json").write_text(json.dumps(tests, indent=1))

    # ---- sensitivitas latensi per build (titik; rata-rata undian campuran pada himpunan penuh)
    PB = []
    for b_, g_ in LB.groupby("build"):
        t_ = g_.set_index("varian")["mean"]
        for lab, b2x in (("B2 terkunci (np.quantile)", b2ms), ("B2 efisien", b2e)):
            Tr = act * t_["A3 gate rute A"] + (1 - act) * t_["A3 gate rute B"] + b2x
            f_ = float(np.clip((Tr - t_["Vanilla"]) / (t_["P2-Static"] - t_["Vanilla"]), 0, 1))
            mm = [R_VP.metrics(m_) for m_ in P.random_masks(n, f_, n_draws, seed + 19)]
            PB.append(dict(build=b_, b2=lab, T_router=Tr, f_campuran=f_,
                           **{f"router − campuran {k}": point["router"][k] - np.mean([x[k] for x in mm]) for k in METRICS}))
    PB = pd.DataFrame(PB)

    # ---- COCO lengkap (12 metrik) untuk sistem utama
    def routed(mask):
        return [a if u else b for a, b, u in zip(D["A"], D["B"], mask)]
    full = {"Vanilla": st_V, "P2-Static": st_P, "Branch A (paksa)": st_A, "Branch B (paksa)": st_B,
            "Router (B2, deploy)": coco_of(gt, routed(d_dep), ids)[0],
            "Heuristik kepadatan (B2)": coco_of(gt, routed(d_heur), ids)[0]}
    for nm, key in (("router", "Router (B2, deploy)"), ("heur", "Heuristik kepadatan (B2)"), ("A", "Branch A (paksa)"),
                    ("V", "Vanilla"), ("P2S", "P2-Static")):
        for k in METRICS:                                  # CocoMix (dasar bootstrap) == COCOeval langsung pada jdict ter-route
            a_, b_ = point[nm][k], full[key][k]
            b_ = np.nan if b_ < 0 else b_
            assert (np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) < 1e-9, f"CocoMix {nm} {k} {a_} != COCOeval {b_}"
    FULL = pd.DataFrame(full).T
    FULL["latensi_ms"] = [TV, TP, TA, TB, T_router, T_heur]
    FULL["aktivasi_P2"] = [0.0, 1.0, 1.0, 0.0, act, act_h]

    # ---- H3 & perilaku skor
    has = TG["n_gt_small"] > 0; anyg = TG["n_gt_all"] > 0
    ratio = np.where(anyg, TG["n_gt_small"] / np.maximum(TG["n_gt_all"], 1), np.nan)
    H3 = []
    for nm, sc in (("gate", sg), ("heuristik", sh)):
        r, p, nn_ = partial_spearman_p(sc[anyg], ratio[anyg], [TG["n_gt_all"][anyg]])
        H3.append(dict(skor=nm, uji="H3: ρ parsial(skor, proporsi GT kecil | jumlah GT)", rho=r, p=p, n=nn_))
        r, p, nn_ = partial_spearman_p(sc[has], TG["soft_small"][has], [TG["n_gt_small"][has], TG["n_gt_all"][has]])
        H3.append(dict(skor=nm, uji="ρ parsial(skor, Δ_small | GT kecil & semua), gambar berobjek kecil", rho=r, p=p, n=nn_))
    from scipy import stats as st_
    nsm_det = np.array([np.log1p(((d[:, 4] >= HEUR_CONF) & (d[:, 2] * d[:, 3] < SMALL_PX ** 2)).sum()) if len(d) else 0.0
                        for d in te["detB"]])
    SAN = dict(spearman_heur_trt_vs_pytorch=float(st_.spearmanr(sh, te["heur"])[0]),
               spearman_heur_vs_kotak_kecil_deteksi_D2b=float(st_.spearmanr(te["heur"], nsm_det)[0]),
               spearman_gate_vs_heur=float(st_.spearmanr(sg, sh)[0]),
               spearman_gate_trt_vs_pytorch=float(st_.spearmanr(sg, te["score_g32"])[0]),
               setuju_keputusan_router_job_vs_offline=float((d_dep == d_off).mean()), sumber_skor_heuristik=heur_src,
               nilai_heuristik_unik_test=int(len(np.unique(sh))))

    # ---- B2 pada urutan acak (ketahanan; titik saja)
    rng = np.random.default_rng(seed + 29)
    orders = {"asli (nama file)": np.arange(n), **{f"acak #{j + 1}": rng.permutation(n) for j in range(cfg["b2_orders"])}}
    BO = []
    for on, order in orders.items():
        for nm, sc, thr0 in (("gate", sg, A["thr_val_gate"]), ("heuristik", sh, A["thr_val_heur"])):
            dec = np.zeros(n, bool)
            dec[order] = b2_decide(sc[order], rho, cfg["b2_window"], cfg["b2_min_hist"], thr0)[0]
            m_ = R_AB.metrics(dec)
            base = [R_AB.metrics(r_) for r_ in P.random_masks(n, dec.mean(), n_draws, seed + 31)]
            BO.append(dict(urutan=on, skor=nm, aktivasi=float(dec.mean()), abs_dev=abs(float(dec.mean()) - rho),
                           **{f"gain_{k}": m_[k] - np.mean([b_[k] for b_ in base]) for k in METRICS}))
    BO = pd.DataFrame(BO)

    # ---- pemeriksaan silang PyTorch FP32 (titik saja)
    X32 = {}
    if cfg["fp32_check"]:
        R32, _, _ = router_of(gt, te["detA"], te["detB"], ids)
        RVP32, _, _ = router_of(gt, te["detP"], te["detV"], ids)
        d32, _ = b2_decide(te["score_g32"], rho, cfg["b2_window"], cfg["b2_min_hist"], A["thr_val_gate"])
        a32 = float(d32.mean()); f32 = float(np.clip((a32 * TA + (1 - a32) * TB + b2ms - TV) / (TP - TV), 0, 1))
        m_r = R32.metrics(d32)
        rnd = [R32.metrics(r_) for r_ in P.random_masks(n, a32, n_draws, seed + 11)]
        mix = [RVP32.metrics(r_) for r_ in P.random_masks(n, f32, n_draws, seed + 19)]
        mA, mB = R32.metrics(np.ones(n, bool)), R32.metrics(np.zeros(n, bool))
        mV, mP = RVP32.metrics(np.zeros(n, bool)), RVP32.metrics(np.ones(n, bool))
        for k in METRICS:
            X32[k] = {"router − acak": m_r[k] - np.mean([r_[k] for r_ in rnd]),
                      "router − campuran V/P2S": m_r[k] - np.mean([r_[k] for r_ in mix]),
                      "Branch A − P2-Static": mA[k] - mP[k], "Branch B − Vanilla": mB[k] - mV[k],
                      "router": m_r[k], "Vanilla": mV[k], "P2-Static": mP[k]}
        X32 = pd.DataFrame(X32)
        SAN["aktivasi_fp32"] = a32; SAN["setuju_keputusan_fp16_vs_fp32"] = float((d32 == d_dep).mean())

    # ---- kriteria per model
    bs = BS.set_index(["uji", "metric"])
    get = lambda u, m="mAP_small": bs.loc[(u, m)] if (u, m) in bs.index else None
    h2 = get(tests[0][0]); h6 = get(tests[2][0])
    stat = lambda r: "unggul (CI > 0)" if r["ci_lo"] > 0 else ("kalah (CI < 0)" if r["ci_hi"] < 0 else "tidak kalah (CI memuat 0)")
    KR = dict(
        H1_mAP50=bool(st_V["mAP50"] < full["Router (B2, deploy)"]["mAP50"] < st_P["mAP50"]),
        H1_latensi=bool(TV < T_router < TP),
        H2_router_vs_acak=bool(h2["ci_lo"] > 0),
        H5_rute_A=bool(TA - TP <= cfg["h5_ms"]), H5_rute_B=bool(TB - TV <= cfg["h5_ms"]),
        H6_status=stat(h6),
        H7_aktivasi=bool(abs(act - rho) <= cfg["h7_dev"]), H7_gain=bool(h2["ci_lo"] > 0),
        H6_sensitivitas_B2_efisien=stat(get(H6_SENS)) if get(H6_SENS) is not None else None)
    RES = dict(model=spec["name"], aktivasi_test=act, aktivasi_heur=act_h, latensi=lat, kriteria=KR, sanity=SAN,
               parity=res["parity"], label_cost=A["label_cost"], gflops=A["gflops"], gate_info=A["gate"]["info"],
               thr_val_gate=A["thr_val_gate"], thr_val_heur=A["thr_val_heur"], trt=res.get("trt_version"), gpu=res.get("gpu"),
               backend=res["backend"], n_test=n, prof_activation=res.get("prof_activation"), build_error=res.get("build_error"),
               engine_bytes=res.get("engine_bytes"),
               h3={f"{r['skor']} | {r['uji']}": dict(rho=r["rho"], p=r["p"], n=r["n"]) for r in H3})
    # ---- simpan
    BS.to_csv(out / "bootstrap_test.csv", index=False); FULL.to_csv(out / "coco_lengkap.csv")
    LT.to_csv(out / "latensi.csv"); LB.to_csv(out / "latensi_per_build.csv", index=False)
    pd.DataFrame(H3).to_csv(out / "h3_perilaku_skor.csv", index=False); BO.to_csv(out / "b2_urutan.csv", index=False)
    PB.to_csv(out / "h6_per_build.csv", index=False)
    if len(X32):
        X32.to_csv(out / "cek_fp32.csv")
    pd.DataFrame({"file": te["file"], "skor_gate": sg, "thr_b2": rec["thr_g"], "pakai_P2": d_dep, "skor_heuristik": sh,
                  "pakai_P2_heuristik": d_heur, "skor_gate_fp32": te["score_g32"], "n_gt_kecil": TG["n_gt_small"],
                  "n_gt": TG["n_gt_all"], "delta_soft_small": TG["soft_small"]}).to_csv(out / "per_gambar_test.csv", index=False)
    (out / "hasil.json").write_text(json.dumps(RES, indent=1, default=float))
    return RES, dict(BS=BS, FULL=FULL, LT=LT, H3=pd.DataFrame(H3), BO=BO, X32=X32, PB=PB)


def spec_seed(A):
    return A["spec"]["seed"]


def print_model(RES, T, log=print):
    pd.set_option("display.width", 220)
    log("\n=== " + RES["model"] + " ===")
    log("COCO (TensorRT " + str(RES["backend"]) + "):\n" + T["FULL"][["mAP50-95", "mAP50", "mAP_small", "mAP_medium", "mAP_large",
                                                                   "AR_small", "latensi_ms", "aktivasi_P2"]].round(4).to_string())
    log("Bootstrap berpasangan (test):\n" + T["BS"][["uji", "metric", "diff", "ci_lo", "ci_hi", "p_two_sided"]].round(4).to_string(index=False))
    log("Latensi (ms; rata-rata antar-build):\n" + T["LT"].round(3).to_string())
    log("Turunan latensi:", {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in RES["latensi"].items()})
    log("H3 / perilaku skor:\n" + T["H3"].round(4).to_string(index=False))
    log("B2 per urutan:\n" + T["BO"].round(4).to_string(index=False))
    log("H6 per build (titik; B2 terkunci vs efisien):\n" + T["PB"].round(4).to_string(index=False))
    if isinstance(T["X32"], pd.DataFrame) and len(T["X32"]):
        log("Cek silang PyTorch FP32 (titik):\n" + T["X32"].round(4).to_string())
    log("Sanity:", RES["sanity"]); log("Paritas job:", RES["parity"]); log("Kriteria:", RES["kriteria"])


# =============================================================================
# ORKESTRASI SATU MODEL
# =============================================================================
INTERMEDIATE = ("stageA.pkl", "job_res.json", "job_dets.pkl", "gt_test.json")


def run_model(spec, cfg, out_root, repo, log=print):
    out = Path(out_root) / spec["name"]; out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available() or cfg["backend"] == "ort", "GPU tidak aktif: cek Accelerator di pengaturan notebook"
    if cfg.get("reuse_dir"):
        cand = sorted(Path(cfg["reuse_dir"]).glob(f"**/{spec['name']}/stageA.pkl"))
        src = cand[0].parent if cand else Path(cfg["reuse_dir"]) / spec["name"]
        for f in INTERMEDIATE:
            if (src / f).exists() and not (out / f).exists():
                (out / f).write_bytes((src / f).read_bytes()); log(f"pakai ulang {src / f}")
    A_path = out / "stageA.pkl"
    if A_path.exists():
        A = pickle.loads(A_path.read_bytes()); log("tahap A dimuat dari", A_path)
        if not all(Path(p).exists() for p in A["onnx"].get(cfg["precision"], {}).values()):
            assert (out / "job_res.json").exists(), "ONNX tidak ada (dipakai ulang dari sesi lain?) dan job belum jalan: hapus stageA.pkl"
    else:
        A = stage_a(spec, cfg, out, device, log)
        A_path.write_bytes(pickle.dumps(A)); log(f"tahap A selesai ({A['stage_a_min']:.1f} mnt)")
    if not ((out / "job_res.json").exists() and (out / "job_dets.pkl").exists()):
        prec = cfg["precision"]
        assert prec in A["onnx"], f"ONNX {prec} tidak tersedia (GPU tidak aktif saat ekspor?)"
        O = A["onnx"][prec]
        job = dict(backend=cfg["backend"], precision=prec, onnx={k: O[k] for k in ("vanilla", "p2static", "a3g", "a3h", "s1g")},
                   data_yaml=spec["data_yaml"], split="test", imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"],
                   max_det=spec["max_det"], coco_max_dets=spec["coco_max_dets"], rho=cfg["rho"], b2_window=cfg["b2_window"],
                   b2_min_hist=cfg["b2_min_hist"], thr_val_gate=A["thr_val_gate"], thr_val_heur=A["thr_val_heur"],
                   n_prof_images=cfg["n_prof_images"], n_warmup=cfg["n_warmup"], n_reps=cfg["n_reps"],
                   n_builds=spec.get("n_builds", cfg["n_builds"]), engine_dir=str(out / "engines"), workers=cfg.get("job_workers", 2),
                   out_json=str(out / "job_res.json"), dets_out=str(out / "job_dets.pkl"))
        (out / "job.json").write_text(json.dumps(job, indent=1))
        log(f"job inferensi ({cfg['backend']}, {prec}; timeout {cfg['trt_timeout_s']} s) ...")
        run_subprocess([sys.executable, "-u", "-m", "tahap4_final", str(out / "job.json")], cwd=str(repo),
                       timeout_s=cfg["trt_timeout_s"], tail_path=out / "job_log_tail.txt", log=log)
    res = json.loads((out / "job_res.json").read_text())
    J = pickle.loads((out / "job_dets.pkl").read_bytes())
    log("analisis + bootstrap ...")
    RES, T = analyze(A, res, J, cfg, out, log)
    print_model(RES, T, log)
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return RES, T


def zip_model(out_root, name, dest_dir, intermediate=False, log=print):
    d = Path(out_root) / name
    arc = Path(dest_dir) / f"tahap4_{name}.zip"
    skip_suffix = (".onnx", ".engine", ".pkl")
    with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file() and not f.name.startswith("gt_") and f.suffix not in skip_suffix:
                z.write(f, Path(name) / f.relative_to(d))
    log(f"📦 {arc} ({arc.stat().st_size / 1e6:.2f} MB)")
    if intermediate:
        arc2 = Path(dest_dir) / f"tahap4_{name}_intermediate.zip"
        with zipfile.ZipFile(arc2, "w", zipfile.ZIP_DEFLATED) as z:
            for f in INTERMEDIATE:
                if (d / f).exists():
                    z.write(d / f, Path(name) / f)
        log(f"📦 {arc2} ({arc2.stat().st_size / 1e6:.1f} MB) — untuk menjalankan ulang analisis tanpa GPU (reuse_dir)")


# =============================================================================
# RINGKASAN GABUNGAN (semua model)
# =============================================================================
MAIN_TESTS = ("H2/H7 | router (B2) − acak (aktivasi sama)",
              "H6 | router (B2) − campuran acak Vanilla/P2-Static (latensi sama)", H6_SENS,
              "gate (B2) − heuristik kepadatan (B2)", "gate − heuristik kepadatan (top-ρ)",
              "router (B2) − kaskade bergate Vanilla→P2-Static (latensi sama)",
              "router (B2) − Branch A (P2 selalu)", "Branch A − P2-Static", "Branch B − Vanilla")


def pooled_tests(dirs, tests_wanted=MAIN_TESTS):
    """Selisih rata-rata antar-model per resample (indeks resample sama -> sah bila test sama, mis. KITTI 3 seed)."""
    S, Pt, Tdef = {}, {}, None
    for d in dirs:
        z = np.load(d / "boot_samples.npz"); S[d.name] = {k: z[k] for k in z.files}
        Pt[d.name] = json.loads((d / "boot_point.json").read_text())
        Tdef = Tdef or json.loads((d / "boot_tests.json").read_text())
    rows = []
    for label, a, bs in Tdef:
        if label not in tests_wanted:
            continue
        ok = [m for m in S if all(f"{x}|mAP_small" in S[m] for x in [a, *bs])]
        if len(ok) != len(S):
            continue
        for met in METRICS:
            diffs = [S[m][f"{a}|{met}"] - np.nanmean(np.stack([S[m][f"{b}|{met}"] for b in bs]), 0) for m in ok]
            pts = [Pt[m][a][met] - np.nanmean([Pt[m][b][met] for b in bs]) for m in ok]
            sm = np.mean(np.stack(diffs), 0); sm = sm[np.isfinite(sm)]
            lo, hi = np.percentile(sm, [2.5, 97.5])
            rows.append(dict(uji=label, metric=met, diff_rata2=float(np.mean(pts)), ci_lo=lo, ci_hi=hi, p_two_sided=_pval(sm),
                             n_model=len(ok)))
    return pd.DataFrame(rows)


def merge(out_root, all_models, merge_glob=None, log=print):
    out_root = Path(out_root)
    if merge_glob:
        hits = sorted(glob.glob(merge_glob, recursive=True)); log("gabung:", hits)
        for hp in map(Path, hits):
            if hp.is_file() and hp.suffix == ".zip":
                with zipfile.ZipFile(hp) as z:
                    z.extractall(out_root)
            elif hp.is_dir():                                   # dataset Kaggle: zip sudah diekstrak otomatis
                import shutil
                for s in all_models:
                    cand = [d.parent for d in hp.glob(f"**/{s['name']}/DONE")]
                    if not cand:
                        continue
                    best = sorted(cand, key=lambda d: ((d / "h6_per_build.csv").exists(), (d / "DONE").stat().st_mtime))[-1]
                    if len(cand) > 1:
                        log(f"{s['name']}: {len(cand)} salinan, dipakai {best}")
                    shutil.copytree(best, out_root / s["name"], dirs_exist_ok=True)
    done = [s for s in all_models if (out_root / s["name"] / "DONE").exists()]
    log("model selesai:", [s["name"] for s in done], f"({len(done)}/{len(all_models)})")
    assert done, "Belum ada model yang selesai."
    R = {s["name"]: json.loads((out_root / s["name"] / "hasil.json").read_text()) for s in done}
    BS = pd.concat([pd.read_csv(out_root / s["name"] / "bootstrap_test.csv").assign(model=s["name"]) for s in done])
    FULL = pd.concat([pd.read_csv(out_root / s["name"] / "coco_lengkap.csv", index_col=0).assign(model=s["name"]) for s in done])
    rows = []
    for nm, r in R.items():
        bm = BS[(BS.model == nm) & (BS.metric == "mAP_small")].set_index("uji")
        row = dict(model=nm, aktivasi=r["aktivasi_test"], T_router=r["latensi"]["T_router"], T_V=r["latensi"]["T_V"],
                   T_P2S=r["latensi"]["T_P2S"], f_campuran=r["latensi"]["f_mix"], **{k: v for k, v in r["kriteria"].items()})
        for u in MAIN_TESTS:
            if u in bm.index:
                row[f"{u.split('|')[-1].strip()} [mAP_small]"] = f"{bm.loc[u, 'diff']:+.4f} [{bm.loc[u, 'ci_lo']:+.4f}, {bm.loc[u, 'ci_hi']:+.4f}]"
        rows.append(row)
    TAB = pd.DataFrame(rows).set_index("model")
    H3 = pd.DataFrame([dict(model=nm, kunci=k, **v) for nm, r in R.items() for k, v in r["h3"].items()])
    h3m = H3.kunci == "gate | H3: ρ parsial(skor, proporsi GT kecil | jumlah GT)"
    H3.loc[h3m, "p_holm"] = E.holm(H3.loc[h3m, "p"].fillna(1.0).values)
    H3.loc[h3m, "H3_lolos"] = (H3.loc[h3m, "rho"] > 0) & (H3.loc[h3m, "p_holm"] < 0.05)
    kitti = [out_root / s["name"] for s in done if s["name"].startswith("kitti")]
    POOL = pooled_tests(kitti) if len(kitti) > 1 else pd.DataFrame()
    LAT = pd.concat([pd.read_csv(out_root / s["name"] / "latensi_per_build.csv").assign(model=s["name"]) for s in done])
    LATP = LAT.groupby("varian")["mean"].agg(["mean", "std", "count"]).rename(columns={"count": "n_build_total"})
    return dict(R=R, TAB=TAB, BS=BS, FULL=FULL, H3=H3, POOL=POOL, LATP=LATP, done=[s["name"] for s in done])


def pareto_plot(out_root, R, FULL, path):
    import matplotlib.pyplot as plt
    names = list(R)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.4), squeeze=False)
    for ax, nm in zip(axes[0], names):
        r, F = R[nm], FULL[FULL.model == nm]
        L = r["latensi"]
        bp = json.loads((Path(out_root) / nm / "boot_point.json").read_text())
        ax.plot([L["T_V"], L["T_P2S"]], [F.loc["Vanilla", "mAP_small"], F.loc["P2-Static", "mAP_small"]], "k:", label="garis Vanilla↔P2-Static")
        ax.plot(L["T_V"], F.loc["Vanilla", "mAP_small"], "ks", label="Vanilla"); ax.plot(L["T_P2S"], F.loc["P2-Static", "mAP_small"], "k^", label="P2-Static")
        cm = [v["mAP_small"] for k, v in bp.items() if k.startswith("campuran_router_")]
        if cm:
            ax.plot(L["T_router"], np.mean(cm), "o", mfc="none", ms=9, label="campuran acak V/P2S")
        if "kaskade" in bp:
            ax.plot(L["T_router"], bp["kaskade"]["mAP_small"], "D", mfc="none", ms=8, label="kaskade V→P2S")
        ax.plot(L["T_heur"], F.loc["Heuristik kepadatan (B2)", "mAP_small"], "P", ms=9, label="heuristik (B2)")
        ax.plot(L["T_router"], F.loc["Router (B2, deploy)", "mAP_small"], "*", ms=14, label="router (B2, A3 FP16)")
        ax.plot([L["T_A3_B"], L["T_A3_A"]], [F.loc["Branch B (paksa)", "mAP_small"], F.loc["Branch A (paksa)", "mAP_small"]], "x--", alpha=.5, label="rute paksa B/A")
        ax.set(title=nm, xlabel="latensi TensorRT FP16 (ms)", ylabel="mAP_small (COCO)"); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=130)
    return fig


if __name__ == "__main__":
    job_main(sys.argv[1])
