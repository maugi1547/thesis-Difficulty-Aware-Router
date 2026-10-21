"""
Evaluasi final terkunci + latensi ujung-ke-ujung (router_final) — port Tahap 4/4B ke lingkungan Ultralytics stok.

    python scripts/suite.py run   --config configs/kitti.yaml --seed 42 --weights RUN/weights/last.pt \
        --gate RUN/gate/gate.json --vanilla VAN/last.pt --p2static P2S/last.pt --out /kaggle/working/suite
    python scripts/suite.py merge --out /kaggle/working/suite_gabung --glob "/kaggle/input/**/suite_*" \
        --models kitti_seed42 kitti_seed7 kitti_seed2026 visdrone_seed42

Isi = eksperimen/tahap4b.py (yang menghasilkan angka Tahap 4/4B), dengan bagian yang menyentuh model diganti modul
router_final (trunk / BranchA / BranchB / Stage1 / build_if) dan Ultralytics stok. Analisis statistik (analyze,
bootstrap, kriteria H1–H7, merge) disalin apa adanya, dengan satu tambahan yang dikunci SEBELUM run ulang:
  H6 utama = router (B2 efisien) − campuran acak V/P2S, fraksi campuran dari latensi UJUNG-KE-UJUNG
  (label H6_UTAMA). H6 Tahap 4 (komposit, B2 np.quantile) & 4B tetap dihitung untuk kesinambungan.
Kesetaraan dengan Tahap 4 diuji pada intermediate Kaggle (tests/test_suite_replay.py) dan pada bobot M0 yang
dipetakan lewat tools/convert_m0.py (tools/compare_suite_runs.py).

Pipeline per model: stage_a (PyTorch: cache train/val/test, cek gate.json, ekspor ONNX + verifikasi ORT)
-> job (proses terpisah: TensorRT FP16 atau ONNX Runtime; akurasi per gambar + latensi per build + ujung-ke-ujung)
-> analyze (CocoMix + bootstrap berpasangan + kriteria).
"""
from __future__ import annotations

import collections
import contextlib
import gc
import glob
import io
import json
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

from .baselines import HEUR_CONF, S1Heur, StaticOut, density_head, load_static, static_with_feats
from .budget import SortedWindowQuantile, b2_decide, b2_host_overhead
from .cocomix import (METRICS, FastUnionBoot, Router, UnionBoot, _pval, boot_engine, boot_tests, cluster_index, holm,
                      partial_spearman_p, random_masks, top_rho)
from .data import SMALL_PX, build_coco_gt, coco_eval, image_id_of, jdict, load_data, make_loader, small_area_of, to_dets
from .export import IOCast, Stage1, build_if, cpu_copy, export_onnx, half_copy
from .gate import fit_gate as _fit_gate, load_gate
from .model import FEATS_A, FEATS_B, BranchA, BranchB, Trunk, load_detector, pool4

H6_SENS = "[sensitivitas, pasca-hasil] H6 | router (B2 efisien) − campuran acak V/P2S (latensi sama)"
H6_E2E = "[4B, pasca-hasil] H6 | router (B2) − campuran acak V/P2S (latensi ujung-ke-ujung)"
H6_UTAMA = "H6 utama | router (B2 efisien) − campuran acak V/P2S (latensi ujung-ke-ujung)"
_STATE = {"max_dets": (1, 10, 100)}


# =============================================================================
# COCO helper
# =============================================================================
def jdict_list(dets, image_ids):
    return jdict(dets, image_ids)


def coco_of(gt, dets, image_ids):
    return coco_eval(gt, jdict(dets, image_ids), _STATE["max_dets"])


def router_of(gt, dets_a, dets_b, image_ids):
    """Router: use_a True -> dets_a. Returns (router, stats_a, stats_b)."""
    sa, ea = coco_of(gt, dets_a, image_ids)
    sb, eb = coco_of(gt, dets_b, image_ids)
    return Router(ea, eb, image_ids), sa, sb


def subset_coco_gt(dataset, names, path, n):
    import copy
    ds = copy.copy(dataset); ds.labels = dataset.labels[:n]
    return build_coco_gt(ds, names, path)


def gflops(module, args):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(*args)
    return fc.get_total_flops() / 1e9


# =============================================================================
# CACHE PyTorch (train / val / test)
# =============================================================================
@torch.no_grad()
def cache_split(model, loader, device, conf, iou, max_det, dets=True, heur=True, statics=None, max_images=None,
                log_every=1000, log=print):
    """Satu forward per gambar (letterbox persegi, batch 1, urutan loader). Returns dict list/array per gambar."""
    s2a, s2b = BranchA(model).eval(), BranchB(model).eval()
    dens = density_head(model).eval() if heur else None
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
        p3, p2, p4, p5 = model.trunk(im)
        out["feat"].append(pool4(p3, p2, p4, p5)[0].cpu().numpy())
        if dets:
            out["detA"].append(to_dets(s2a(p3, p2, p4, p5), rp, osh, conf, iou, max_det))
            out["detB"].append(to_dets(s2b(p3, p4, p5), rp, osh, conf, iou, max_det))
        if dens is not None:
            na, ns = dens.counts(p3, torch.tensor([sa], device=device))
            out["heur"].append(float(dens.combine(na, ns)[0])); out["heur_n_all"].append(float(na[0]))
            out["heur_n_small"].append(float(ns[0]))
        out["small_area"].append(sa)
        if van is not None:
            yV, fV = static_with_feats(van, im)
            out["featV"].append(pool4(*fV)[0].cpu().numpy())
            out["detV"].append(to_dets(yV, rp, osh, conf, iou, max_det))
            yP = p2s(im)
            out["detP"].append(to_dets(yP[0] if isinstance(yP, (tuple, list)) else yP, rp, osh, conf, iou, max_det))
        out["file"].append(Path(b["im_file"][0]).name)
        out["image_id"].append(image_id_of(b["im_file"][0]))
        if log_every and (i + 1) % log_every == 0:
            log(f"  {i + 1} gambar | {time.perf_counter() - t0:.0f} s")
    for k in ("feat", "featV"):
        out[k] = np.stack(out[k]) if out[k] else None
    for k in ("heur", "heur_n_all", "heur_n_small", "small_area"):
        out[k] = np.asarray(out[k], float)
    return out


def fit_gate(X_, delta, C):
    g = _fit_gate(X_, delta, C=C)
    y = (np.asarray(delta) > 0)
    return dict(w=np.asarray(g["w"], np.float64), b=float(g["b"]),
                info=dict(C=C, n_train=int(len(y)), pos_rate=float(y.mean()), n_fitur=int(np.asarray(X_).shape[1])))


def check_stage_equivalence(model, loader, device, n=2, tol=1e-5):
    """trunk + BranchA/B (stage deployment) == forward_dual model utuh."""
    s2a, s2b = BranchA(model).eval(), BranchB(model).eval()
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= n:
                break
            im = batch["img"].to(device).float() / 255.0
            p3, p2, p4, p5 = model.trunk(im)
            oa, ob = model.forward_dual(im)
            yA_ref, yB_ref = (o[0] if isinstance(o, (tuple, list)) else o for o in (oa, ob))
            dA = ((s2a(p3, p2, p4, p5) - yA_ref).abs().max() / yA_ref.abs().max()).item()
            dB = ((s2b(p3, p4, p5) - yB_ref).abs().max() / yB_ref.abs().max()).item()
            assert dA < tol and dB < tol, f"Stage tidak setara dgn model utuh: {dA:.2e} / {dB:.2e}"
    return True


# =============================================================================
# EKSPOR ONNX (gate A3, heuristik A3, Vanilla, P2-Static, Stage 1 gate)
# =============================================================================
def export_suite(model, van, p2s, gate, x0, out_dir, device, precisions=("fp16",), opset=17):
    """FP32 (CPU) selalu (verifikasi ORT + backend ORT); FP16 (GPU) bila diminta. Returns (paths, modul FP32 CPU)."""
    out = Path(out_dir)
    sa0 = torch.tensor([SMALL_PX ** 2 * 0.25])
    m = cpu_copy(model)
    s1g, s1h = Stage1(m, gate).eval(), S1Heur(m).eval()
    s2a, s2b = BranchA(m).eval(), BranchB(m).eval()
    with torch.no_grad():
        _, p3, p2, p4, p5 = s1g(x0)
        n_a, n_b = int(s2a(p3, p2, p4, p5).shape[-1]), int(s2b(p3, p4, p5).shape[-1])
    statics = {"vanilla": StaticOut(cpu_copy(van)).eval(), "p2static": StaticOut(cpu_copy(p2s)).eval()}
    names = ("s1g", "s1h", "s2a", "s2b", "vanilla", "p2static", "a3g", "a3h")
    P32 = {k: out / f"{k}_fp32.onnx" for k in names}
    export_onnx(s1g, (x0,), P32["s1g"], ["image"], ["score", *FEATS_A], opset)
    export_onnx(s1h, (x0, sa0), P32["s1h"], ["image", "small_area"], ["score", *FEATS_A], opset)
    export_onnx(s2a, (p3, p2, p4, p5), P32["s2a"], list(FEATS_A), ["y_A"], opset)
    export_onnx(s2b, (p3, p4, p5), P32["s2b"], list(FEATS_B), ["y_B"], opset)
    for k, mod in statics.items():
        export_onnx(mod, (x0,), P32[k], ["image"], ["y"], opset)
    build_if(P32["s1g"], P32["s2a"], P32["s2b"], P32["a3g"], n_a, n_b)
    build_if(P32["s1h"], P32["s2a"], P32["s2b"], P32["a3h"], n_a, n_b)
    paths = {"fp32": P32}
    mods = dict(s1g=s1g, s1h=s1h, s2a=s2a, s2b=s2b, **statics, n_a=n_a, n_b=n_b)
    if "fp16" in precisions:
        m16 = half_copy(model, device)
        s1g16, s1h16 = Stage1(m16, gate).to(device).eval(), S1Heur(m16).to(device).eval()
        s2a16, s2b16 = BranchA(m16).eval(), BranchB(m16).eval()
        x, sa = x0.to(device), sa0.to(device)
        with torch.no_grad():
            _, q3, q2, q4, q5 = s1g16(x); s1h16(x, sa)
            h = [t.half() for t in (q3, q2, q4, q5)]
            s2a16(*h); s2b16(h[0], h[2], h[3])
        P16 = {k: out / f"{k}_fp16.onnx" for k in names}
        export_onnx(s1g16, (x,), P16["s1g"], ["image"], ["score", *FEATS_A], opset)
        export_onnx(s1h16, (x, sa), P16["s1h"], ["image", "small_area"], ["score", *FEATS_A], opset)
        export_onnx(IOCast(s2a16), (q3, q2, q4, q5), P16["s2a"], list(FEATS_A), ["y_A"], opset)
        export_onnx(IOCast(s2b16), (q3, q4, q5), P16["s2b"], list(FEATS_B), ["y_B"], opset)
        for k, src in (("vanilla", van), ("p2static", p2s)):
            mod = IOCast(StaticOut(half_copy(src, device))).eval()
            with torch.no_grad():
                mod(x)
            export_onnx(mod, (x,), P16[k], ["image"], ["y"], opset)
        build_if(P16["s1g"], P16["s2a"], P16["s2b"], P16["a3g"], n_a, n_b)
        build_if(P16["s1h"], P16["s2a"], P16["s2b"], P16["a3h"], n_a, n_b)
        paths["fp16"] = P16
        del m16, s1g16, s1h16, s2a16, s2b16
    return paths, mods


def verify_suite(P32, mods, loader, n=4, tol=1e-4):
    """ORT (graf FP32) vs PyTorch: skor gate, fitur, Stage 2, statis, A3 untuk semua mode & arah threshold."""
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


# =============================================================================
# TAHAP A (PyTorch): cache, cek gate.json, threshold val, cache test, GFLOPs, ekspor + verifikasi ONNX
# =============================================================================
def stage_a(spec, cfg, out, device, log=print):
    """
    spec: name, data_yaml, seed, weights, gate (gate.json dari scripts/fit_gate.py), vanilla_ckpt, p2static_ckpt,
          max_det, coco_max_dets, expected_counts.
    Gate DIAMBIL dari gate.json (satu sumber kebenaran); gate di-fit ulang di sini dari cache train hanya sebagai
    pemeriksaan silang (harus sama), lalu thr0 val dihitung ulang dan harus sama dengan gate.json.
    """
    t0 = time.perf_counter()
    _STATE["max_dets"] = tuple(spec["coco_max_dets"])
    DATA = load_data(spec["data_yaml"], spec.get("expected_counts")); NAMES = DATA["names"]
    L = {s: make_loader(DATA, s, 1, cfg["imgsz"], cfg.get("workers", 2)) for s in ("train", "val", "test")}
    N = {s: len(ld.dataset) for s, ld in L.items()}
    if cfg.get("n_train"):
        N["train"] = min(N["train"], cfg["n_train"])
    model, minfo = load_detector(spec["weights"], device)
    seed_ck = (minfo.get("train_args") or {}).get("seed")
    assert seed_ck in (None, spec["seed"]), f"seed checkpoint {seed_ck} != {spec['seed']}"
    check_stage_equivalence(model, L["test"], device)
    van = load_static(spec["vanilla_ckpt"], device, DATA["nc"])
    p2s = load_static(spec["p2static_ckpt"], device, DATA["nc"])
    statics = (van, p2s)
    kw = dict(conf=cfg["conf"], iou=cfg["iou"], max_det=spec["max_det"], log=log)
    log(f"train: cache detektor{' + Vanilla + P2-Static' if cfg['cascade'] else ''} ({N['train']} gambar) ...")
    Ctr = cache_split(model, L["train"], device, dets=True, heur=False, statics=statics if cfg["cascade"] else None,
                      max_images=N["train"], **kw)
    gt_tr = subset_coco_gt(L["train"].dataset, NAMES, out / "gt_train.json", N["train"])
    R_tr, _, _ = router_of(gt_tr, Ctr["detA"], Ctr["detB"], Ctr["image_id"])
    TG_tr = R_tr.targets(); del R_tr; gc.collect()

    gj = load_gate(spec["gate"])
    refit = fit_gate(Ctr["feat"], TG_tr["soft_small"], cfg["gate_C"])
    w_rel = float(np.abs(refit["w"] - gj["w"]).max() / (np.abs(gj["w"]).max() + 1e-12))
    gate_check = dict(w_rel_maks=w_rel, b_abs=abs(refit["b"] - float(gj["b"])), gate_json=str(spec["gate"]))
    gate = dict(w=np.asarray(gj["w"], np.float64), b=float(gj["b"]), info=dict(refit["info"], **gate_check))
    log(f"gate.json vs fit ulang dari cache: {gate_check}")
    assert w_rel < 1e-4, f"gate.json berbeda dari fit ulang (w rel {w_rel:.2e}): checkpoint/data/C berbeda?"
    label_cost = dict(n_gambar_train=int(N["train"]), n_kotak_gt_train=int(TG_tr["n_gt_all"].sum()),
                      n_kotak_gt_kecil_train=int(TG_tr["n_gt_small"].sum()),
                      frac_gambar_berobjek_kecil=float((TG_tr["n_gt_small"] > 0).mean()))
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
    if "thr0" in gj:
        gate["info"]["thr0_gate_json"] = float(gj["thr0"])
        assert abs(thr_g - float(gj["thr0"])) <= 1e-5 * (1 + abs(thr_g)), f"thr0 val {thr_g} != gate.json {gj['thr0']}"
    log(f"thr val: gate {thr_g:+.4f} | heuristik {thr_h:.4f} | nilai heuristik unik di val {len(np.unique(Cva['heur']))}/{N['val']}")

    log(f"test: cache detektor + Vanilla + P2-Static ({N['test']} gambar) ...")
    Cte = cache_split(model, L["test"], device, dets=True, heur=True, statics=statics, **kw)
    gt_te_path = out / "gt_test.json"; build_coco_gt(L["test"].dataset, NAMES, gt_te_path)
    test = {k: Cte[k] for k in ("file", "image_id", "detA", "detB", "detV", "detP", "heur", "heur_n_all", "heur_n_small", "small_area")}
    test["score_g32"] = Cte["feat"] @ gate["w"] + gate["b"]
    test["score_c32"] = Cte["featV"] @ casc["w"] + casc["b"] if casc else None

    log("GFLOPs + ekspor ONNX (FP32 untuk verifikasi, FP16 untuk TensorRT) ...")
    x0 = next(iter(L["test"]))["img"].float() / 255.0
    mc = cpu_copy(model)
    with torch.no_grad():
        p3, p2, p4, p5 = mc.trunk(x0)
    dens = density_head(mc).eval(); sa0 = torch.tensor([SMALL_PX ** 2 * 0.25])
    gw_, gb_ = torch.as_tensor(gate["w"], dtype=torch.float32), float(gate["b"])
    fl = {"trunk 0–15": gflops(Trunk(mc), (x0,)),
          "gate post-hoc (pooling + Gemm)": gflops(lambda *a: pool4(*a) @ gw_ + gb_, (p3, p2, p4, p5)),
          "heuristik P3 (head P3 Detect_B + hitung)": gflops(dens, (p3, sa0)),
          "Stage 2A": gflops(BranchA(mc), (p3, p2, p4, p5)), "Stage 2B": gflops(BranchB(mc), (p3, p4, p5)),
          "Vanilla": gflops(StaticOut(cpu_copy(van)).eval(), (x0,)),
          "P2-Static": gflops(StaticOut(cpu_copy(p2s)).eval(), (x0,))}
    log("GFLOPs:", {k: round(v, 3) for k, v in fl.items()})
    del mc, dens
    precs = ("fp16",) if (cfg["precision"] == "fp16" and (torch.cuda.is_available() or cfg.get("allow_cpu_fp16"))) else ()
    paths, mods = export_suite(model, van, p2s, gate, x0, out, device, precs, cfg.get("opset", 17))
    ver = verify_suite(paths["fp32"], mods, L["test"], n=cfg["n_onnx_check"])
    log("verifikasi ONNX Runtime (selisih maks):", {k: f"{v:.1e}" for k, v in ver.items()})
    A = dict(spec=spec, minfo={k: (str(v) if k == "train_args" else v) for k, v in minfo.items()}, gate=gate, casc=casc,
             thr_val_gate=thr_g, thr_val_heur=thr_h,
             val=dict(score_g32=sg_va, score_h32=Cva["heur"], file=Cva["file"]), test=test, label_cost=label_cost,
             gflops=fl, onnx={p: {k: str(v) for k, v in d.items()} for p, d in paths.items()}, onnx_verif=ver,
             n_a=mods["n_a"], n_b=mods["n_b"], gt_test=str(gt_te_path), stage_a_min=(time.perf_counter() - t0) / 60)
    del model, van, p2s, mods, Cva, Cte; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return A


# =============================================================================
# JOB INFERENSI (TensorRT / ORT) — disalin dari eksperimen/tahap4b.py
# =============================================================================
E2E_VARIANTS = ("Vanilla", "P2-Static", "A3 gate rute A", "A3 gate rute B",
                "Router B2 terkunci (e2e)", "Router B2 efisien (e2e)")


def e2e_pass(ops, H, batches, rho, window, min_hist, thr0, n_warmup, n_reps, ctx, dsync):
    """[TAHAP 4B] Latensi ujung-ke-ujung (ms/gambar) pada aliran test berurutan.

    batches: list (img_uint8, small_area) dalam urutan aliran (= urutan acc_pass).
    Timer mencakup: hitung threshold B2 di host -> salin gambar & threshold ke GPU -> engine A3 -> salin y, use_a,
    skor ke host. Persiapan gambar (uint8 -> float, ke buffer pinned) di luar timer, sama dengan lat_pass Tahap 4.
    Riwayat B2 dimulai ulang di setiap pengulangan, sehingga keputusan setiap pengulangan identik dengan acc_pass.
    Kembali: (lat: varian -> list ms, use: varian -> list keputusan pengulangan pertama).
    """
    def b2_router(efisien):
        hist = SortedWindowQuantile(window) if efisien else collections.deque(maxlen=window)

        def f():
            n_h = len(hist.h) if efisien else len(hist)
            if n_h >= min_hist:
                thr = np.float32(hist.quantile(1 - rho) if efisien else np.quantile(np.asarray(hist, np.float64), 1 - rho))
            else:
                thr = np.float32(thr0)
            H["thr"]["a3g"].fill_(float(thr))
            ops[("a3g", -1.0)]()
            s_ = float(H["a3g"][2][0])
            if efisien:
                hist.push(s_)
            else:
                hist.append(s_)
            return bool(float(H["a3g"][1][0]) > 0.5)
        return f

    def fixed(op):
        def f():
            H["thr"]["a3g"].fill_(float(thr0))
            op()
            return None
        return f

    def fns_fresh():
        return {"Vanilla": ops["vanilla"], "P2-Static": ops["p2static"],
                "A3 gate rute A": fixed(ops[("a3g", 1.0)]), "A3 gate rute B": fixed(ops[("a3g", 0.0)]),
                "Router B2 terkunci (e2e)": b2_router(False), "Router B2 efisien (e2e)": b2_router(True)}

    def timed(fn, im, sa):
        H["img"].copy_(im.float() / 255.0); H["sa"].fill_(sa); dsync()       # konversi di luar timer
        w0 = time.perf_counter()
        with ctx():
            u = fn()
        return (time.perf_counter() - w0) * 1e3, u

    warm = fns_fresh()
    for im, sa in batches[:n_warmup]:
        for fn in warm.values():
            timed(fn, im, sa)
    lat = {k: [] for k in E2E_VARIANTS}
    use = {k: [] for k in E2E_VARIANTS if k.startswith("Router")}
    for rep in range(n_reps):
        fns = fns_fresh()
        order = list(fns)
        for i, (im, sa) in enumerate(batches):
            for k in (order if (rep + i) % 2 == 0 else order[::-1]):
                ms, u = timed(fns[k], im, sa)
                lat[k].append(ms)
                if rep == 0 and k in use:
                    use[k].append(u)
    return lat, use



def job_main(job_path):
    job = json.loads(Path(job_path).read_text())
    T0 = time.perf_counter()
    log = lambda *a: print(f"[job {time.perf_counter() - T0:6.0f}s]", *a, flush=True)
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

    DATA = load_data(job["data_yaml"], job.get("expected_counts"))
    loader = make_loader(DATA, job["split"], 1, job["imgsz"], workers=job.get("workers", 2))
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

    nms = lambda y, rp, osh: to_dets(y, rp, osh, job["conf"], job["iou"], job["max_det"])

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
        stream = [(b["img"].clone(), small_area_of(b["ratio_pad"][0])) for b in loader]   # [4B] seluruh test, urutan aliran
        L_e, U_e = e2e_pass(ops, H, stream, rho, W, MINH, thr_g0, job["n_warmup"], job["n_reps"], ctx, dsync)
        del stream
        for k, v in L_e.items():
            a = np.asarray(v)
            res.setdefault("latency_e2e_builds", {})[f"b{bi}|{k}"] = dict(mean=a.mean(), std=a.std(), p50=np.percentile(a, 50),
                                                                         p95=np.percentile(a, 95), n=int(a.size))
        res.setdefault("e2e_activation", {}).update({f"b{bi}|{k}": float(np.mean(v)) for k, v in U_e.items()})
        if bi == 0:
            ref = list(map(bool, rec["use_g"]))
            res["e2e_parity"] = {k: dict(n=len(v), beda_vs_akurasi=int(sum(a_ != b_ for a_, b_ in zip(v, ref))))
                                 for k, v in U_e.items()}
        Path(job["out_json"]).write_text(json.dumps(res, default=float, indent=1))
        log(f"[4B] latensi ujung-ke-ujung build {bi} selesai | paritas {res.get('e2e_parity')}")
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
    _STATE["max_dets"] = tuple(spec["coco_max_dets"])
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
    d_top_g, d_top_h = top_rho(sg, rho), top_rho(sh, rho)
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

    # ---- [4B] latensi ujung-ke-ujung (pass terpisah, sesi yang sama)
    LE = None
    if res.get("latency_e2e_builds"):
        LE = pd.DataFrame([dict(build=k.split("|", 1)[0], varian=k.split("|", 1)[1], **v)
                           for k, v in res["latency_e2e_builds"].items()])
        te_ = LE.groupby("varian")["mean"].mean().to_dict()
        TrE, TVE, TPE = te_["Router B2 terkunci (e2e)"], te_["Vanilla"], te_["P2-Static"]
        TAE, TBE = te_["A3 gate rute A"], te_["A3 gate rute B"]
        act_e = float(np.mean([v for k, v in res["e2e_activation"].items() if k.endswith("terkunci (e2e)")]))
        lat.update(T_router_e2e=TrE, T_router_e2e_efisien=te_["Router B2 efisien (e2e)"], T_V_e2e=TVE, T_P2S_e2e=TPE,
                   T_A_e2e=TAE, T_B_e2e=TBE, aktivasi_e2e=act_e,
                   T_router_komposit_sepass=act_e * TAE + (1 - act_e) * TBE + b2ms,
                   f_mix_e2e=float(np.clip((TrE - TVE) / (TPE - TVE), 0, 1)),
                   f_mix_e2e_efisien=float(np.clip((te_["Router B2 efisien (e2e)"] - TVE) / (TPE - TVE), 0, 1)),
                   e2e_parity=res.get("e2e_parity"))
        lat["selisih_e2e_minus_komposit_sepass"] = TrE - lat["T_router_komposit_sepass"]

    # ---- campuran untuk bootstrap
    k_dep, k_h = int(d_dep.sum()), int(d_heur.sum())
    U_AB, U_VP = UnionBoot(R_AB.bA, R_AB.bB), UnionBoot(R_VP.bA, R_VP.bB)
    ab = lambda m_: (U_AB, R_AB.sorted_mask(m_)); vp = lambda m_: (U_VP, R_VP.sorted_mask(m_))
    mixes = {"A": ab(np.ones(n, bool)), "B": ab(np.zeros(n, bool)), "V": vp(np.zeros(n, bool)), "P2S": vp(np.ones(n, bool)),
             "router": ab(d_dep), "router_top": ab(d_top_g), "heur": ab(d_heur), "heur_top": ab(d_top_h)}
    for j, m_ in enumerate(random_masks(n, k_dep / n, n_draws, seed + 11)):
        mixes[f"acak_router_{j}"] = ab(m_)
    n2 = cfg.get("n_draws_sekunder", 3)                       # pembanding sekunder: undian lebih sedikit (waktu)
    for j, m_ in enumerate(random_masks(n, rho, n2, seed + 13)):
        mixes[f"acak_top_{j}"] = ab(m_)
    for j, m_ in enumerate(random_masks(n, k_h / n, n2, seed + 17)):
        mixes[f"acak_heur_{j}"] = ab(m_)
    for j, m_ in enumerate(random_masks(n, f_mix, n_draws, seed + 19)):
        mixes[f"campuran_router_{j}"] = vp(m_)
    if np.isfinite(f_mix_h):
        for j, m_ in enumerate(random_masks(n, f_mix_h, n2, seed + 23)):
            mixes[f"campuran_heur_{j}"] = vp(m_)
    for j, m_ in enumerate(random_masks(n, f_mix_e, n_draws, seed + 19)):
        mixes[f"campuran_b2ef_{j}"] = vp(m_)
    if "f_mix_e2e" in lat:                                   # [4B]
        for j, m_ in enumerate(random_masks(n, lat["f_mix_e2e"], n_draws, seed + 19)):
            mixes[f"campuran_e2e_{j}"] = vp(m_)
        for j, m_ in enumerate(random_masks(n, lat["f_mix_e2e_efisien"], n_draws, seed + 19)):   # H6 utama
            mixes[f"campuran_utama_{j}"] = vp(m_)
    if te.get("score_c32") is not None:
        mixes["kaskade"] = vp(top_rho(te["score_c32"], rho_c))
        mixes["kaskade_b2ef"] = vp(top_rho(te["score_c32"], rho_c_e))
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
    if "f_mix_e2e" in lat:                                   # [4B]
        tests.append((H6_E2E, "router", grp("campuran_e2e_")))
        tests.append((H6_UTAMA, "router", grp("campuran_utama_")))
    tb = time.perf_counter()
    log(f"bootstrap: {len(mixes)} routing × {cfg['n_boot']} resample ({cfg['boot_jobs']} proses) ...")
    point, samples = boot_engine(mixes, cfg["n_boot"], cfg["boot_seed"], cfg["boot_jobs"])
    BS = boot_tests(point, samples, tests)
    log(f"bootstrap selesai ({time.perf_counter() - tb:.0f} s)")
    np.savez_compressed(out / "boot_samples.npz", **{f"{k}|{m}": v[m].astype(np.float32) for k, v in samples.items() for m in METRICS})
    (out / "boot_point.json").write_text(json.dumps(point, indent=1, default=float))
    (out / "boot_tests.json").write_text(json.dumps(tests, indent=1))

    # ---- bootstrap KLASTER (VisDrone: ID video sumber = token pertama nama file; dikunci sebelum run ulang)
    if cfg.get("cluster_by") == "video_prefix":
        tk = time.perf_counter()
        pos_ = {iid: j for j, iid in enumerate(sorted(ids))}
        grp_ = {}
        for f_, iid in zip(te["file"], ids):
            grp_.setdefault(f_.split("_")[0], []).append(pos_[iid])
        G_ = [np.asarray(v) for v in grp_.values()]; sizes = np.array([len(g) for g in G_])
        fast_, fmix = {}, {}
        for k_, (U_, u_) in mixes.items():
            fast_.setdefault(id(U_), FastUnionBoot(U_)); fmix[k_] = (fast_[id(U_)], u_)
        _, S_cl = boot_engine(fmix, None, None, cfg["boot_jobs"], idx=cluster_index(G_, cfg["n_boot"], cfg["cluster_seed"]))
        BS_cl = boot_tests(point, S_cl, tests)
        big = int(np.argmax(sizes)); keep = np.concatenate([g for j_, g in enumerate(G_) if j_ != big])
        pt_nb = {k_: U_.metrics(u_, keep) for k_, (U_, u_) in fmix.items()}
        jk = [dict(uji=lb, metric=m_, diff_tanpa_klaster_terbesar=pt_nb[a_][m_] - np.nanmean([pt_nb[b_][m_] for b_ in bs_]))
              for lb, a_, bs_ in tests if a_ in pt_nb and bs_ and all(b_ in pt_nb for b_ in bs_) for m_ in METRICS]
        CL = BS[["uji", "metric", "diff", "ci_lo", "ci_hi"]].rename(columns={"ci_lo": "lo_gambar", "ci_hi": "hi_gambar"}) \
            .merge(BS_cl[["uji", "metric", "ci_lo", "ci_hi", "p_two_sided"]].rename(
                columns={"ci_lo": "lo_klaster", "ci_hi": "hi_klaster", "p_two_sided": "p_klaster"}), on=["uji", "metric"]) \
            .merge(pd.DataFrame(jk), on=["uji", "metric"])
        CL["rasio_lebar"] = (CL.hi_klaster - CL.lo_klaster) / (CL.hi_gambar - CL.lo_gambar)
        CL["status_klaster"] = np.where(CL.lo_klaster > 0, "CI > 0", np.where(CL.hi_klaster < 0, "CI < 0", "memuat 0"))
        CL.to_csv(out / "bootstrap_klaster.csv", index=False)
        (out / "klaster_info.json").write_text(json.dumps(dict(n_klaster=len(G_), ukuran=sizes.tolist(), B=cfg["n_boot"],
                                                               seed=cfg["cluster_seed"]), indent=1))
        log(f"bootstrap klaster: {len(G_)} klaster (terbesar {sizes.max()}) | {time.perf_counter() - tk:.0f} s")

    # ---- sensitivitas latensi per build (titik; rata-rata undian campuran pada himpunan penuh)
    PB = []
    for b_, g_ in LB.groupby("build"):
        t_ = g_.set_index("varian")["mean"]
        for lab, b2x in (("B2 terkunci (np.quantile)", b2ms), ("B2 efisien", b2e)):
            Tr = act * t_["A3 gate rute A"] + (1 - act) * t_["A3 gate rute B"] + b2x
            f_ = float(np.clip((Tr - t_["Vanilla"]) / (t_["P2-Static"] - t_["Vanilla"]), 0, 1))
            mm = [R_VP.metrics(m_) for m_ in random_masks(n, f_, n_draws, seed + 19)]
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
            base = [R_AB.metrics(r_) for r_ in random_masks(n, dec.mean(), n_draws, seed + 31)]
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
        rnd = [R32.metrics(r_) for r_ in random_masks(n, a32, n_draws, seed + 11)]
        mix = [RVP32.metrics(r_) for r_ in random_masks(n, f32, n_draws, seed + 19)]
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
    if "f_mix_e2e" in lat:                                   # [4B]
        KR.update(H1_latensi_e2e=bool(lat["T_V_e2e"] < lat["T_router_e2e"] < lat["T_P2S_e2e"]),
                  H5_rute_A_e2e=bool(lat["T_A_e2e"] - lat["T_P2S_e2e"] <= cfg["h5_ms"]),
                  H5_rute_B_e2e=bool(lat["T_B_e2e"] - lat["T_V_e2e"] <= cfg["h5_ms"]),
                  H6_e2e_status=stat(get(H6_E2E)) if get(H6_E2E) is not None else None,
                  H1_latensi_e2e_efisien=bool(lat["T_V_e2e"] < lat["T_router_e2e_efisien"] < lat["T_P2S_e2e"]),
                  H6_utama_status=stat(get(H6_UTAMA)) if get(H6_UTAMA) is not None else None)
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
    if LE is not None:                                        # [4B]
        LE.to_csv(out / "latensi_e2e_per_build.csv", index=False)
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
                   out_json=str(out / "job_res.json"), dets_out=str(out / "job_dets.pkl"),
                   expected_counts=spec.get("expected_counts"))
        (out / "job.json").write_text(json.dumps(job, indent=1))
        log(f"job inferensi ({cfg['backend']}, {prec}; timeout {cfg['trt_timeout_s']} s) ...")
        run_subprocess([sys.executable, "-u", "-m", "rfinal.suite", str(out / "job.json")], cwd=str(repo),
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
    arc = Path(dest_dir) / f"suite_{name}.zip"
    skip_suffix = (".onnx", ".engine", ".pkl")
    with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file() and not f.name.startswith("gt_") and f.suffix not in skip_suffix:
                z.write(f, Path(name) / f.relative_to(d))
    log(f"📦 {arc} ({arc.stat().st_size / 1e6:.2f} MB)")
    if intermediate:
        arc2 = Path(dest_dir) / f"suite_{name}_intermediate.zip"
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
              "router (B2) − Branch A (P2 selalu)", "Branch A − P2-Static", "Branch B − Vanilla", H6_E2E, H6_UTAMA)


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
                   T_P2S=r["latensi"]["T_P2S"], f_campuran=r["latensi"]["f_mix"],
                   **{k: r["latensi"].get(k) for k in ("T_router_e2e", "T_router_e2e_efisien", "f_mix_e2e_efisien",
                                                       "T_V_e2e", "T_P2S_e2e", "aktivasi_e2e",
                                                       "T_router_komposit_sepass", "selisih_e2e_minus_komposit_sepass",
                                                       "f_mix_e2e")},
                   **{k: v for k, v in r["kriteria"].items()})
        for u in MAIN_TESTS:
            if u in bm.index:
                row[f"{u.split('|')[-1].strip()} [mAP_small]"] = f"{bm.loc[u, 'diff']:+.4f} [{bm.loc[u, 'ci_lo']:+.4f}, {bm.loc[u, 'ci_hi']:+.4f}]"
        rows.append(row)
    TAB = pd.DataFrame(rows).set_index("model")
    H3 = pd.DataFrame([dict(model=nm, kunci=k, **v) for nm, r in R.items() for k, v in r["h3"].items()])
    h3m = H3.kunci == "gate | H3: ρ parsial(skor, proporsi GT kecil | jumlah GT)"
    H3.loc[h3m, "p_holm"] = holm(H3.loc[h3m, "p"].fillna(1.0).values)
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
