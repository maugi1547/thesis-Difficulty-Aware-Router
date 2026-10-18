"""
tahap3a.py — Tahap 3a (tanpa training): D1 ablasi fitur gate, B kendali anggaran aktivasi, A1 profil latensi + GFLOPs.

D1  Fitur gate post-hoc (target & model TERKUNCI: soft_small biner, StandardScaler + LogReg C=0.001):
    varian blok fitur dari [GAP‖GMP](p3) ‖ [GAP‖GMP](p2) ‖ [GAP‖GMP](p4) ‖ [GAP‖GMP](p5) ‖ z  (961 kolom).
    Dinilai pada urutan skor tepat ρ (top-ρ) agar perbandingan antar-varian tidak terganggu pergeseran threshold.
B   Strategi threshold untuk anggaran aktivasi ρ pada aliran gambar test:
    B1 kuantil val tetap | B2 kuantil bergulir N gambar terakhir | B3 kuantil 10% awal aliran (tanpa label).
    Dievaluasi di 90% sisa aliran; urutan asli (nama file) + beberapa urutan acak.
A1  Profil TensorRT per segmen (CUDA event, satu stream) + GFLOPs per modul (FlopCounterMode).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import tahap2_eval as E
import tahap2b_plafon as P
import tahap2c_gate as G
from router_train_utils import find_router

# indeks kolom fitur (urutan P.cache_dets: p3, p2, p4, p5 masing-masing [GAP | GMP], lalu z)
def feature_blocks(c_p3, c_p2, c_p4, c_p5):
    idx, o = {}, 0
    for nm, c in (("p3", c_p3), ("p2", c_p2), ("p4", c_p4), ("p5", c_p5)):
        idx[f"{nm}_gap"] = np.arange(o, o + c); idx[f"{nm}_gmp"] = np.arange(o + c, o + 2 * c); o += 2 * c
    idx["z"] = np.array([o])
    return idx, o + 1


def variant_columns(blocks):
    cat = lambda *ks: np.concatenate([blocks[k] for k in ks])
    allf = [k for k in blocks if k != "z"]
    return {
        "lengkap (961)": cat(*allf, "z"),
        "tanpa z": cat(*allf),
        "hanya z": cat("z"),
        "p3+p2": cat("p3_gap", "p3_gmp", "p2_gap", "p2_gmp"),
        "hanya p3": cat("p3_gap", "p3_gmp"),
        "hanya p2": cat("p2_gap", "p2_gmp"),
        "hanya p4": cat("p4_gap", "p4_gmp"),
        "hanya p5": cat("p5_gap", "p5_gmp"),
        "hanya GAP": cat(*[k for k in allf if k.endswith("gap")]),
        "hanya GMP": cat(*[k for k in allf if k.endswith("gmp")]),
    }


def fit_logreg(X, delta, C=0.001):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    y = (np.asarray(delta) > 0).astype(int)
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def d1_feature_ablation(Ctr, Tgt_tr, Cva, Rva, Cte, Rte, rho, C=0.001, n_random=20, seed=0, n_boot=1000):
    """Returns (tabel, skor per varian {split: array}, model per varian)."""
    blocks, D = feature_blocks(*_channel_split(Cte))
    cols = variant_columns(blocks)
    Xtr, Xva, Xte = P.features(Ctr), P.features(Cva), P.features(Cte)
    assert Xtr.shape[1] == D, f"jumlah kolom fitur {Xtr.shape[1]} != {D}"
    rnd = {}
    for nm, R, n in (("val", Rva, len(Cva["z"])), ("test", Rte, len(Cte["z"]))):
        rnd[nm] = [R.metrics(m) for m in P.random_masks(n, rho, n_random, seed + 3)]
    base = {s: {k: float(np.mean([r[k] for r in rnd[s]])) for k in E.BOOT_METRICS} for s in rnd}
    rows, scores, models = [], {}, {}
    for vname, cidx in cols.items():
        pipe = fit_logreg(Xtr[:, cidx], Tgt_tr["soft_small"], C)
        sv, st = pipe.decision_function(Xva[:, cidx]), pipe.decision_function(Xte[:, cidx])
        scores[vname] = {"val": sv, "test": st}; models[vname] = (pipe, cidx)
        mv, mt = Rva.metrics(P.top_rho(sv, rho)), Rte.metrics(P.top_rho(st, rho))
        rows.append(dict(varian=vname, n_fitur=len(cidx),
                         **{f"val_gain_{k}": mv[k] - base["val"][k] for k in E.BOOT_METRICS},
                         **{f"test_gain_{k}": mt[k] - base["test"][k] for k in E.BOOT_METRICS}))
    tab = pd.DataFrame(rows)
    # uji berpasangan di test: lengkap vs tanpa z; tanpa z vs acak
    m_full = Rte.mix(P.top_rho(scores["lengkap (961)"]["test"], rho))
    m_noz = Rte.mix(P.top_rho(scores["tanpa z"]["test"], rho))
    m_rnd = [Rte.mix(m) for m in P.random_masks(len(Cte["z"]), rho, 5, seed + 7)]
    bs = pd.concat([E.paired_bootstrap(m_full, [m_noz], n_boot, seed).assign(uji="lengkap − tanpa z (test, top-ρ)"),
                    E.paired_bootstrap(m_noz, m_rnd, n_boot, seed).assign(uji="tanpa z − acak (test, top-ρ)")])
    return tab, bs, scores, models


def _channel_split(C):
    """Jumlah kanal p3, p2, p4, p5 dari panjang fitur (YOLOv8n-P2: 64, 32, 128, 256). Diverifikasi via 2*sum+1."""
    D = P.features(C).shape[1]                                          # termasuk kolom z
    for ch in ((64, 32, 128, 256), (128, 64, 256, 512), (192, 96, 384, 576)):
        if 2 * sum(ch) + 1 == D:
            return ch
    raise ValueError(f"Tidak bisa menentukan kanal dari {D} fitur")


# =============================================================================
# B — kendali anggaran aktivasi
# =============================================================================
def threshold_stream(scores_stream, rho, calib_n, strategy, thr_val=None, window=None):
    """Keputusan use_a untuk setiap posisi aliran. Hanya memakai skor MASA LALU (tanpa label)."""
    s = np.asarray(scores_stream, float); n = len(s)
    if strategy == "B1 kuantil val":
        return s > thr_val
    if strategy == "B3 kalibrasi 10% awal":
        return s > np.quantile(s[:calib_n], 1 - rho)
    if strategy.startswith("B2"):
        out = np.zeros(n, bool)
        for t in range(n):
            lo = max(0, t - window)
            hist = s[lo:t]
            thr = np.quantile(hist, 1 - rho) if len(hist) >= 20 else thr_val
            out[t] = s[t] > thr
        return out
    raise ValueError(strategy)


def b_budget(Rte, scores_test, s_val, rho, windows=(50, 200, 500), n_orders=5, calib_frac=0.10, n_random=20, seed=0):
    """Returns DataFrame per (urutan, strategi): aktivasi terealisasi di 90% evaluasi, metrik, gain vs acak (aktivasi sama)."""
    n = len(scores_test); thr_val = float(np.quantile(s_val, 1 - rho))
    rng = np.random.default_rng(seed)
    orders = {"asli (nama file)": np.arange(n), **{f"acak #{j + 1}": rng.permutation(n) for j in range(n_orders)}}
    strategies = ["B1 kuantil val", "B3 kalibrasi 10% awal"] + [f"B2 bergulir N={w}" for w in windows]
    rows = []
    for oname, order in orders.items():
        calib_n = int(round(calib_frac * n))
        eval_idx = order[calib_n:]                                   # indeks cache yang dievaluasi
        idx_sorted = Rte.perm[eval_idx]                              # posisi COCO
        for strat in strategies:
            w = int(strat.split("=")[1]) if strat.startswith("B2") else None
            dec_stream = threshold_stream(scores_test[order], rho, calib_n, strat, thr_val, w)
            use_a = np.zeros(n, bool); use_a[order] = dec_stream
            act = float(use_a[eval_idx].mean())
            m = Rte.metrics(use_a, idx=idx_sorted)
            k = int(use_a[eval_idx].sum()); ne = len(eval_idx)
            rnd = []
            for d in P.random_masks(ne, k / ne, n_random, seed + 11):
                ua = np.zeros(n, bool); ua[eval_idx[d]] = True
                rnd.append(Rte.metrics(ua, idx=idx_sorted))
            base = {kk: float(np.mean([r[kk] for r in rnd])) for kk in E.BOOT_METRICS}
            rows.append(dict(urutan=oname, strategi=strat, aktivasi=act, abs_dev=abs(act - rho),
                             **{kk: m[kk] for kk in E.BOOT_METRICS},
                             **{f"gain_{kk}": m[kk] - base[kk] for kk in E.BOOT_METRICS}))
    return pd.DataFrame(rows)


# =============================================================================
# A1 — GFLOPs & modul ekspor untuk profil
# =============================================================================
class PostHocGateNoZ(nn.Module):
    """Gate post-hoc TANPA z lama: hanya pooling global + Gemm (router lama tidak dihitung)."""
    def __init__(self, w, b, thr):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(np.asarray(w), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))
        self.register_buffer("thr", torch.tensor(float(thr), dtype=torch.float32))

    def score(self, p3, p2, p4, p5):
        f = torch.cat([G._pool(p3), G._pool(p2), G._pool(p4), G._pool(p5)], 1).float()
        return f @ self.w + self.b

    def forward(self, p3, p2, p4, p5):
        return (self.score(p3, p2, p4, p5) > self.thr).float()


class TrunkOnly(nn.Module):
    """Layer 0–15 tanpa gate: keluaran p3, p2, p4, p5 (untuk mengisolasi biaya gate)."""
    def __init__(self, full_model):
        super().__init__()
        self.s = G.Stage1PostHoc(full_model, nn.Identity())

    def forward(self, x):
        return self.s.trunk(x)


def folded_weights(pipe):
    sc, lr = pipe[0], pipe[1]
    scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    w = lr.coef_[0] / scale
    return w, float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / scale))


def gflops(module, args):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(*args)
    return fc.get_total_flops() / 1e9


def export_profile_onnx(model, vanilla, p2static, gate_full, gate_noz, x0, out_dir):
    """ONNX untuk profil + GFLOPs per modul (CPU, input 1x3x640x640)."""
    m = E.cpu_copy(model)
    rt = find_router(m)
    g_full = G.PostHocGate(rt, gate_full["w"], gate_full["b"], gate_full["thr"]).eval()
    g_noz = PostHocGateNoZ(gate_noz["w"], gate_noz["b"], gate_noz["thr"]).eval()
    s1_full, s1_noz, trunk = G.Stage1PostHoc(m, g_full).eval(), G.Stage1PostHoc(m, g_noz).eval(), TrunkOnly(m).eval()
    _, s2a, s2b = E.make_stages(m)
    van, p2s = E._StaticOut(E.cpu_copy(vanilla)).eval(), E._StaticOut(E.cpu_copy(p2static)).eval()
    with torch.no_grad():
        p3, p2, p4, p5 = trunk(x0)
    out = Path(out_dir)
    P_ = {k: out / f"prof_{k}.onnx" for k in ("trunk", "s1_full", "s1_noz", "s2a", "s2b", "vanilla", "p2static")}
    feats = ["p3_neck", "p2_backbone", "p4_neck", "p5_sppf"]
    E.export_onnx(trunk, (x0,), P_["trunk"], ["image"], feats)
    E.export_onnx(E._Stage1Out(s1_full), (x0,), P_["s1_full"], ["image"], ["gate"] + feats)
    E.export_onnx(E._Stage1Out(s1_noz), (x0,), P_["s1_noz"], ["image"], ["gate"] + feats)
    E.export_onnx(s2a, (p3, p2, p4, p5), P_["s2a"], feats, ["y_A"])
    E.export_onnx(s2b, (p3, p4, p5), P_["s2b"], ["p3_neck", "p4_neck", "p5_sppf"], ["y_B"])
    E.export_onnx(van, (x0,), P_["vanilla"], ["image"], ["y"])
    E.export_onnx(p2s, (x0,), P_["p2static"], ["image"], ["y"])
    fl = {"trunk 0–15": gflops(trunk, (x0,)),
          "gate lama (z)": gflops(lambda a, b: G._raw_gate_logit(rt, a, b), (p3, p2)),
          "gate post-hoc tanpa z (pool+Gemm)": gflops(g_noz.score, (p3, p2, p4, p5)),
          "Stage 2A (P2 + head A)": gflops(s2a, (p3, p2, p4, p5)),
          "Stage 2B (head B)": gflops(s2b, (p3, p4, p5)),
          "Vanilla utuh": gflops(van, (x0,)), "P2-Static utuh": gflops(p2s, (x0,))}
    # cek numerik ONNX stage1 tanpa z vs PyTorch (gate & fitur)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(P_["s1_noz"]), providers=["CPUExecutionProvider"])
    o = sess.run(None, {"image": x0.numpy()})
    with torch.no_grad():
        ref = s1_noz(x0)
    margin = float((g_noz.score(p3, p2, p4, p5) - g_noz.thr).abs())
    assert float(o[0].flatten()[0]) == float(ref[0].flatten()[0]) or margin < 1e-4, "gate ONNX tanpa z != PyTorch"
    return P_, fl


# =============================================================================
# A1 — profil TensorRT per segmen (dijalankan di proses terpisah: python -m tahap3a job.json)
# =============================================================================
def profile_job_main(job_path):
    import sys
    job = json.loads(Path(job_path).read_text())
    log = lambda *a: print("[profil]", *a, flush=True)
    NS = E.init_trt()
    if NS is None:
        Path(job["out_json"]).write_text(json.dumps({"trt_available": False})); sys.exit(3)
    trt, cuda = NS.trt, NS.cuda
    assert not torch.cuda.is_initialized()

    class Eng:
        def __init__(self, path):
            with open(path, "rb") as f, trt.Runtime(NS.logger) as rt_:
                self.engine = rt_.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.inp, self.out, self.d, self.h = [], [], {}, {}
            for i in range(self.engine.num_io_tensors):
                nme = self.engine.get_tensor_name(i)
                (self.inp if self.engine.get_tensor_mode(nme) == trt.TensorIOMode.INPUT else self.out).append(nme)
            for nme in self.inp + self.out:
                shape, dt = tuple(self.engine.get_tensor_shape(nme)), trt.nptype(self.engine.get_tensor_dtype(nme))
                if nme in self.inp:
                    self.ctx.set_input_shape(nme, shape)
                self.h[nme] = cuda.pagelocked_empty(shape, dt)
                self.d[nme] = cuda.mem_alloc(self.h[nme].nbytes)
                self.ctx.set_tensor_address(nme, int(self.d[nme]))

        def bind(self, nme, ptr):
            self.ctx.set_tensor_address(nme, int(ptr))

        def run(self, stream):
            self.ctx.execute_async_v3(stream_handle=stream.handle)

    eng = {}
    for k, p in job["onnx"].items():
        ep = str(Path(p).with_suffix(".engine")); NS.build_engine(p, ep); eng[k] = Eng(ep); log("engine", k)
    stream = cuda.Stream()
    arr = np.load(job["images_npy"])                                   # uint8 (N,3,H,W); konversi di luar timer
    imgs = [np.ascontiguousarray(a[None].astype(np.float32) / 255.0) for a in arr]
    ev = [cuda.Event() for _ in range(8)]

    def t(a, b):
        return ev[b].time_since(ev[a])

    def static(name, im):
        e = eng[name]; w0 = time.perf_counter()
        ev[0].record(stream); cuda.memcpy_htod_async(e.d["image"], im, stream)
        ev[1].record(stream); e.run(stream)
        ev[2].record(stream); cuda.memcpy_dtoh_async(e.h[e.out[0]], e.d[e.out[0]], stream)
        ev[3].record(stream); stream.synchronize()
        return dict(h2d=t(0, 1), exec=t(1, 2), d2h=t(2, 3), gpu_total=t(0, 3), wall=(time.perf_counter() - w0) * 1e3)

    feats_a, feats_b = ("p3_neck", "p2_backbone", "p4_neck", "p5_sppf"), ("p3_neck", "p4_neck", "p5_sppf")

    def routed(s1name, branch, gated):
        s1 = eng[s1name]; s2 = eng["s2a" if branch == "A" else "s2b"]
        for f in (feats_a if branch == "A" else feats_b):
            s2.bind(f, s1.d[f])

        def fn(im):
            w0 = time.perf_counter()
            ev[0].record(stream); cuda.memcpy_htod_async(s1.d["image"], im, stream)
            ev[1].record(stream); s1.run(stream)
            ev[2].record(stream)
            if gated:
                cuda.memcpy_dtoh_async(s1.h["gate"], s1.d["gate"], stream)
            ev[3].record(stream)
            if gated:
                stream.synchronize(); _ = float(s1.h["gate"].flatten()[0])   # titik sinkronisasi host (seperti runtime)
            w_sync = time.perf_counter()
            ev[4].record(stream); s2.run(stream)
            ev[5].record(stream); cuda.memcpy_dtoh_async(s2.h[s2.out[0]], s2.d[s2.out[0]], stream)
            ev[6].record(stream); stream.synchronize()
            return dict(h2d=t(0, 1), stage1=t(1, 2), gate_d2h=t(2, 3), stage2=t(4, 5), d2h=t(5, 6),
                        host_gap=t(3, 4), gpu_total=t(0, 6), wall=(time.perf_counter() - w0) * 1e3,
                        wall_after_sync=(time.perf_counter() - w_sync) * 1e3)
        return fn

    exps = {"Vanilla": lambda im: static("vanilla", im), "P2-Static": lambda im: static("p2static", im)}
    for s1n, lab, gated in (("trunk", "trunk tanpa gate", False), ("s1_full", "gate lengkap (z)", True),
                            ("s1_noz", "gate tanpa z", True)):
        for br in "AB":
            exps[f"rute {br} | {lab}"] = routed(s1n, br, gated)
    for im in imgs[:job["n_warmup"]]:
        for f in exps.values():
            f(im)
    rec = {k: [] for k in exps}
    names = list(exps)
    for rep in range(job["n_reps"]):
        for im in imgs:
            for k in (names if rep % 2 == 0 else names[::-1]):
                rec[k].append(exps[k](im))
    res = {k: pd.DataFrame(v).agg(["mean", "std", "median"]).to_dict() for k, v in rec.items()}
    Path(job["out_json"]).write_text(json.dumps({"trt_available": True, "trt_version": trt.__version__,
                                                 "n": len(imgs) * job["n_reps"], "profil": res}, default=float))
    log("selesai")


if __name__ == "__main__":
    import sys
    profile_job_main(sys.argv[1])
