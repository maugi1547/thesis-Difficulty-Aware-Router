"""
Profil latensi per segmen (diagnostik, di luar kriteria terkunci): dari mana overhead rute B dan rute A berasal?

Pertanyaan (Bab IV Subbab 4.8.2): rute B engine A3 0,10–0,13 ms lebih lambat dari Vanilla dan keputusan host menambah
±0,05 ms. Dugaan: salinan keluaran rute B yang di-pad ke 34.000 anchor (816 KB vs 202 KB Vanilla) dan/atau operator If.

Setiap pemanggilan diukur dalam tiga segmen GPU (CUDA event di stream yang sama) + waktu dinding:
  h2d  = salin citra FP32 (dan thr) pinned -> GPU     exec = engine (atau rantai dua engine)
  d2h  = salin keluaran -> host pinned                 host = dinding − (h2d + exec + d2h) = launch, sync, Python
Persiapan citra (uint8 -> float) di luar timer, sama dengan e2e_pass suite. B2 (host) diukur terpisah.

Varian (engine FP16 sama dengan suite: export_suite + build STRONGLY_TYPED):
  Vanilla, P2-Static, A3 rute A (mode 1), A3 rute B (mode 0)            = varian lat_pass suite
  Rantai s1g→s2a, Rantai s1g→s2b                                         = dua engine tanpa If, tanpa padding
  A3 rute B, salin 8.400                                                 = salin hanya anchor rute B yang asli
  Stage 1 + gate (s1g)                                                   = trunk + gate saja
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import FEATS_A, FEATS_B

SEGMENTS = ("h2d", "exec", "d2h", "host", "gpu_total", "wall")


class TrtEng:
    """Engine TensorRT dengan buffer tensor CUDA PyTorch (sama dengan Eng di job_main suite)."""

    def __init__(self, path, dev):
        import tensorrt as trt
        t2t = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
               trt.int64: torch.int64, trt.bool: torch.bool}
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.inp, self.out, self.buf = [], [], {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (self.inp if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.out).append(n)
        for n in self.inp + self.out:
            shape = tuple(self.engine.get_tensor_shape(n))
            if n in self.inp:
                self.ctx.set_input_shape(n, shape)
            self.buf[n] = torch.zeros(shape, dtype=t2t[self.engine.get_tensor_dtype(n)], device=dev)
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())

    def bind(self, n, t):
        self.buf[n] = t
        self.ctx.set_tensor_address(n, t.data_ptr())

    def run(self, s):
        assert self.ctx.execute_async_v3(s.cuda_stream), "execute_async_v3 gagal"


def gpu_clock_snapshot():
    """Clock SM/memori, suhu, daya (nvidia-smi); string kosong bila tidak tersedia."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,"
                               "power.draw,pstate", "--format=csv"], capture_output=True, text=True, timeout=20).stdout
    except Exception as ex:                                                   # noqa: BLE001
        return f"nvidia-smi tidak tersedia: {ex}"


def make_ops(eng, n_b, dev, pin):
    """eng: dict nama -> engine (vanilla, p2static, a3g, s1g, s2a, s2b). Returns (ops, H) — op(mark) memanggil mark()
    empat kali: awal, setelah h2d, setelah exec, setelah d2h."""
    H = {"img": pin(eng["vanilla"].buf["image"]), "thr": pin(torch.zeros(1))}
    ops = {}

    for k, label in (("vanilla", "Vanilla"), ("p2static", "P2-Static")):
        e = eng[k]
        ho = pin(e.buf[e.out[0]])

        def f(mark, s, e=e, ho=ho):
            mark(); e.buf["image"].copy_(H["img"], non_blocking=True)
            mark(); e.run(s)
            mark(); ho.copy_(e.buf[e.out[0]], non_blocking=True)
            mark()
        ops[label] = f

    e = eng["a3g"]
    hy, hu, hs = pin(e.buf["y"]), pin(e.buf["use_a"]), pin(e.buf["score"])
    hy8 = pin(e.buf["y"][:, :, :n_b])
    modes = {mv: torch.full((1,), mv, dtype=torch.float32, device=dev) for mv in (0.0, 1.0)}

    def a3(mv, cut=False):
        def f(mark, s):
            e.bind("mode", modes[mv])
            mark(); e.buf["image"].copy_(H["img"], non_blocking=True); e.buf["thr"].copy_(H["thr"], non_blocking=True)
            mark(); e.run(s)
            mark()
            if cut:
                hy8.copy_(e.buf["y"][:, :, :n_b], non_blocking=True)
            else:
                hy.copy_(e.buf["y"], non_blocking=True)
            hu.copy_(e.buf["use_a"], non_blocking=True); hs.copy_(e.buf["score"], non_blocking=True)
            mark()
        return f
    ops["A3 rute A"] = a3(1.0)
    ops["A3 rute B"] = a3(0.0)
    ops["A3 rute B, salin 8.400"] = a3(0.0, cut=True)

    s1 = eng["s1g"]
    for k in ("s2a", "s2b"):                                       # input Stage 2 = buffer keluaran Stage 1 (tanpa salin)
        for n in (FEATS_A if k == "s2a" else FEATS_B):
            eng[k].bind(n, s1.buf[n])
    hs1 = pin(s1.buf["score"])
    for k, label in (("s2a", "Rantai s1g→s2a"), ("s2b", "Rantai s1g→s2b")):
        e2 = eng[k]
        hy2 = pin(e2.buf[e2.out[0]])

        def f(mark, s, e2=e2, hy2=hy2):
            mark(); s1.buf["image"].copy_(H["img"], non_blocking=True)
            mark(); s1.run(s); e2.run(s)
            mark(); hy2.copy_(e2.buf[e2.out[0]], non_blocking=True); hs1.copy_(s1.buf["score"], non_blocking=True)
            mark()
        ops[label] = f

    def f1(mark, s):
        mark(); s1.buf["image"].copy_(H["img"], non_blocking=True)
        mark(); s1.run(s)
        mark(); hs1.copy_(s1.buf["score"], non_blocking=True)
        mark()
    ops["Stage 1 + gate (s1g)"] = f1
    return ops, H


class Timer:
    """CUDA: 4 event di stream op (segmen GPU) + perf_counter (dinding). CPU: perf_counter untuk semuanya."""

    def __init__(self, cuda, stream):
        self.cuda, self.s = cuda, stream
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda else None

    def __call__(self, op):
        marks = []
        if self.cuda:
            def mark():
                self.ev[len(marks)].record(self.s); marks.append(None)
            torch.cuda.synchronize()
            w0 = time.perf_counter()
            with torch.cuda.stream(self.s):
                op(mark, self.s)
            self.s.synchronize()
            wall = (time.perf_counter() - w0) * 1e3
            seg = [self.ev[i].elapsed_time(self.ev[i + 1]) for i in range(3)]
            gpu = self.ev[0].elapsed_time(self.ev[3])
        else:
            def mark():
                marks.append(time.perf_counter())
            w0 = time.perf_counter()
            op(mark, self.s)
            wall = (time.perf_counter() - w0) * 1e3
            seg = [(marks[i + 1] - marks[i]) * 1e3 for i in range(3)]
            gpu = (marks[3] - marks[0]) * 1e3
        assert len(marks) == 4
        return dict(h2d=seg[0], exec=seg[1], d2h=seg[2], gpu_total=gpu, wall=wall, host=wall - gpu)


def profile_pass(ops, H, imgs, timer, n_warmup=30, n_reps=3):
    """imgs: list tensor uint8 (1,3,640,640). Urutan varian dibalik tiap gambar (interleaved), seperti e2e_pass."""
    names = list(ops)

    def one(k, im):
        H["img"].copy_(im.float() / 255.0)                         # konversi di luar timer
        return timer(ops[k])
    for im in imgs[:n_warmup]:
        for k in names:
            one(k, im)
    rec = {k: {s: [] for s in SEGMENTS} for k in names}
    for rep in range(n_reps):
        for i, im in enumerate(imgs):
            for k in (names if (rep + i) % 2 == 0 else names[::-1]):
                r = one(k, im)
                for s in SEGMENTS:
                    rec[k][s].append(r[s])
    return rec


def summarize(recs):
    """recs: {build: rec} -> DataFrame build, varian, segmen, mean, std, p50, p95, n."""
    rows = []
    for b, rec in recs.items():
        for k, d in rec.items():
            for s, v in d.items():
                a = np.asarray(v)
                rows.append(dict(build=b, varian=k, segmen=s, mean=a.mean(), std=a.std(), p50=np.percentile(a, 50),
                                 p95=np.percentile(a, 95), n=int(a.size)))
    return pd.DataFrame(rows)


DIFFS = [
    ("rute B − Vanilla", "A3 rute B", "Vanilla"),
    ("rute A − P2-Static", "A3 rute A", "P2-Static"),
    ("A3 rute B − rantai s1g→s2b (If + padding di engine)", "A3 rute B", "Rantai s1g→s2b"),
    ("A3 rute A − rantai s1g→s2a (If di engine)", "A3 rute A", "Rantai s1g→s2a"),
    ("rantai s1g→s2b − Vanilla (engine dipecah + fitur FP32)", "Rantai s1g→s2b", "Vanilla"),
    ("rute B salin semua − salin 8.400 (padding di salinan)", "A3 rute B", "A3 rute B, salin 8.400"),
]


def derive(seg_df):
    """Selisih rata-rata per segmen untuk pasangan diagnostik, per build dan rata-rata build."""
    m = seg_df.pivot_table(index=["build", "varian"], columns="segmen", values="mean")
    rows = []
    for label, a, b in DIFFS:
        per = []
        for bi in sorted(seg_df.build.unique()):
            if (bi, a) in m.index and (bi, b) in m.index:
                per.append({s: m.loc[(bi, a), s] - m.loc[(bi, b), s] for s in SEGMENTS})
        if per:
            rows.append(dict(selisih=label, **{s: float(np.mean([p[s] for p in per])) for s in SEGMENTS},
                             **{f"wall_build{i}": p["wall"] for i, p in enumerate(per)}))
    return pd.DataFrame(rows)
