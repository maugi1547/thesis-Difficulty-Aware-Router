"""
Profil latensi per segmen (diagnostik Subbab 4.8.2, di luar kriteria terkunci H1–H7).

    python scripts/profile_latency.py --config configs/kitti.yaml --weights RUN/weights/last.pt \
        --gate RUN/gate/gate.json --vanilla VAN.pt --p2static P2S.pt --out /kaggle/working/profil_kitti_seed42

Mengekspor ONNX FP16 dengan export_suite (sama dengan suite.py), membangun engine TensorRT (vanilla, p2static, a3g,
s1g, s2a, s2b) sebanyak --n-builds kali, lalu mengukur tiap varian dalam segmen h2d / exec / d2h / host (rfinal/profile.py).
Keluaran: segmen.csv, selisih.csv, paritas_rantai.csv, ringkasan.txt, clock_sebelum.txt, clock_sesudah.txt,
provenance.json. Tanpa GPU: backend ONNX Runtime FP32 di CPU (uji alur saja, angka tidak bermakna).

Latensi L2 (rfinal/latency_l2.py; default aktif, --no-l2 untuk mematikan) memakai engine yang sama di sesi yang sama:
seluruh aliran test, citra uint8 asli -> letterbox GPU -> engine -> NMS GPU -> kotak di host, untuk conf --l2-conf
(0,25 = deployment, satu label; 0,001 = protokol evaluasi, multi-label). Keluaran tambahan: latensi_l2.csv, l2.json (T per varian, f_L2,
aktivasi, kesepakatan keputusan L2 vs L1), keputusan_l2.csv (keputusan router per gambar). H6 pada f_L2 dihitung
offline: scripts/h6_latency.py (membandingkan keputusan_l2.csv dengan keputusan pass akurasi suite).
"""
import argparse
import gc
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch

from rfinal.baselines import load_static
from rfinal.budget import b2_host_overhead
from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, require_stock_ultralytics, write_json
from rfinal.gate import load_gate
from rfinal.model import load_detector
from rfinal.latency_l2 import GpuLetterbox, l1_decisions, l2_pass, make_l2_ops
from rfinal.profile import SEGMENTS, Timer, TrtEng, derive, gpu_clock_snapshot, make_ops, profile_pass, summarize
from rfinal.suite import _OrtEng, export_suite

ENGINES = ("vanilla", "p2static", "a3g", "s1g", "s2a", "s2b")


def chain_parity(eng, ops, H, imgs, n_b, s):
    """Keluaran rantai s1g→s2x vs A3 rute paksa (engine berbeda, FP16): selisih maksimum absolut dan relatif."""
    rows = []
    noop = lambda: None
    for im in imgs:
        H["img"].copy_(im.float() / 255.0)
        out = {}
        for k in ("A3 rute A", "A3 rute B", "Rantai s1g→s2a", "Rantai s1g→s2b"):
            if s is not None:
                with torch.cuda.stream(s):
                    ops[k](noop, s)
                s.synchronize()
            else:
                ops[k](noop, s)
            src = eng["a3g"].buf["y"] if k.startswith("A3") else eng["s2a" if k.endswith("s2a") else "s2b"].buf[
                eng["s2a" if k.endswith("s2a") else "s2b"].out[0]]
            out[k] = src.float().cpu().clone()
        for a, b, cut in (("A3 rute A", "Rantai s1g→s2a", None), ("A3 rute B", "Rantai s1g→s2b", n_b)):
            ya = out[a][:, :, :cut] if cut else out[a]
            d = (ya - out[b]).abs().max().item()
            rows.append(dict(pasangan=f"{a} vs {b}", maks_abs=d, maks_rel=d / (out[b].abs().max().item() + 1e-12)))
        if cut:
            rows.append(dict(pasangan="A3 rute B: padding nol", maks_abs=out["A3 rute B"][:, :, n_b:].abs().max().item(),
                             maks_rel=0.0))
    return pd.DataFrame(rows).groupby("pasangan", sort=False).max().reset_index()


def main():
    ap = argparse.ArgumentParser()
    for k in ("config", "weights", "gate", "vanilla", "p2static", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--n-warmup", type=int, default=30)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--n-builds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-l2", action="store_true", help="lewati latensi L2")
    ap.add_argument("--l2-conf", type=float, nargs="+", default=[0.25, 0.001])
    ap.add_argument("--l2-n-images", type=int, default=0, help="0 = seluruh test")
    ap.add_argument("--l2-reps", type=int, default=3)
    a = ap.parse_args()
    require_stock_ultralytics()
    cfg = load_config(a.config)
    out = Path(a.out); (out / "onnx").mkdir(parents=True, exist_ok=True); (out / "engines").mkdir(exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    data = load_data(cfg["data"], cfg.get("expected_counts"))
    loader = make_loader(data, "test", 1, int(cfg["train"]["imgsz"]), a.workers)
    imgs = []
    for b in loader:
        imgs.append(b["img"].clone())
        if len(imgs) >= max(a.n_images, a.n_warmup):
            break
    print(f"[1/4] {len(imgs)} gambar test, ekspor ONNX ...", flush=True)
    model, _ = load_detector(a.weights, dev)
    van, p2s = load_static(a.vanilla, dev, data["nc"]), load_static(a.p2static, dev, data["nc"])
    gate = load_gate(a.gate)
    x0 = imgs[0].float() / 255.0
    paths, mods = export_suite(model, van, p2s, gate, x0, out / "onnx", dev, ("fp16",) if cuda else ())
    prec = "fp16" if "fp16" in paths else "fp32"
    P, n_a, n_b = paths[prec], int(mods["n_a"]), int(mods["n_b"])
    del model, van, p2s, mods; gc.collect()

    s = torch.cuda.Stream() if cuda else None
    pin = (lambda t: torch.zeros(t.shape, dtype=t.dtype).pin_memory()) if cuda else (lambda t: torch.zeros(t.shape, dtype=t.dtype))
    clock_before = gpu_clock_snapshot()
    recs, build_s, parity = {}, {}, None
    recs_l2, l2_meta = {}, {}
    an = cfg.get("analysis", {})
    b2_args = dict(rho=float(cfg["budget"]["rho"]), window=int(cfg["budget"]["window"]),
                   min_hist=int(cfg["budget"]["min_hist"]), thr0=float(load_gate(a.gate)["thr0"]))
    nms_kw = dict(iou=float(an.get("iou", 0.7)), max_det=int(an.get("max_det", 300)))
    files = list(loader.dataset.im_files)
    if a.l2_n_images:
        files = files[:a.l2_n_images]
    dec = {"file": [Path(f).name for f in files]}                  # keputusan per gambar (dibandingkan dgn suite di h6)
    for bi in range(a.n_builds):
        print(f"[2/4] build {bi}: engine {', '.join(ENGINES)} ({prec}) ...", flush=True)
        eng = {}
        for k in ENGINES:
            if cuda:
                from rfinal.trt import build_engine
                info = build_engine(P[k], out / "engines" / f"{k}_b{bi}.engine")
                build_s[f"b{bi}|{k}"] = info["build_s"]
                eng[k] = TrtEng(info["engine"], dev)
            else:
                eng[k] = _OrtEng(P[k])
        ops, H = make_ops(eng, n_b, dev, pin)
        H["thr"].fill_(0.0)
        if bi == 0:
            parity = chain_parity(eng, ops, H, imgs[:8], n_b, s)
            print(parity.to_string(index=False), flush=True)
        print(f"[3/4] build {bi}: profil {len(ops)} varian × {len(imgs)} gambar × {a.n_reps} ulangan ...", flush=True)
        recs[f"b{bi}"] = profile_pass(ops, H, imgs, Timer(cuda, s), a.n_warmup, a.n_reps)
        if not a.no_l2:
            if bi == 0:
                print(f"[L2] keputusan acuan L1 (letterbox CPU, B2 efisien) pada {len(files)} gambar ...", flush=True)
                ld_full = make_loader(data, "test", 1, int(cfg["train"]["imgsz"]), a.workers)
                use_l1, files_l1 = l1_decisions(eng["a3g"], ld_full, b2_args, s, pin)
                assert [str(x) for x in files_l1[:len(files)]] == [str(x) for x in files], "urutan berkas L1 != L2"
                use_l1 = use_l1[:len(files)]
                dec["L1_b0"] = use_l1
                import cv2                                       # letterbox GPU vs letterbox Ultralytics (CPU), 16 gambar
                chk, dd = GpuLetterbox(int(cfg["train"]["imgsz"]), dev, pin=cuda), []
                for j, bb in enumerate(make_loader(data, "test", 1, int(cfg["train"]["imgsz"]), a.workers)):
                    if j >= 16:
                        break
                    chk.stage(cv2.imread(str(files[j]))); tmp = torch.zeros_like(eng["a3g"].buf["image"]); chk.run(tmp)
                    d = (tmp.float().cpu() - bb["img"].float() / 255.0).abs()
                    dd.append((d.max().item() * 255, d.mean().item() * 255, (d > 1.5 / 255).float().mean().item()))
                lb_chk = dict(maks_abs_level=max(x[0] for x in dd), rata_abs_level=float(np.mean([x[1] for x in dd])),
                              frac_piksel_beda_lebih_1_level=float(np.mean([x[2] for x in dd])))
                print("[L2] letterbox GPU vs CPU (skala 0–255):", lb_chk, flush=True)
            pre = GpuLetterbox(int(cfg["train"]["imgsz"]), dev, pin=cuda)
            for conf in a.l2_conf:
                print(f"[L2] build {bi}, conf {conf}: {len(files)} gambar × {a.l2_reps} ulangan ...", flush=True)
                l2ops, st, reset = make_l2_ops(eng, pre, n_a, n_b, dev, b2_args,
                                               dict(nms_kw, conf=conf, multi_label=conf < 0.01))   # 0,001: multi-label (val)
                rec2, use0 = l2_pass(l2ops, st, reset, pre, files, Timer(cuda, s), a.n_warmup, a.l2_reps)
                recs_l2[f"b{bi}|{conf}"] = rec2
                dec[f"L2_b{bi}_c{conf}"] = use0
                l2_meta[f"b{bi}|{conf}"] = dict(aktivasi_L2=float(np.mean(use0)), aktivasi_L1=float(np.mean(use_l1)),
                                                 kesepakatan_L2_L1=float(np.mean(np.asarray(use0) == np.asarray(use_l1))))
                print("   ", l2_meta[f"b{bi}|{conf}"], flush=True)
        del eng, ops, H; gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    clock_after = gpu_clock_snapshot()

    print("[4/4] ringkasan ...", flush=True)
    seg = summarize(recs)
    diff = derive(seg)
    t_np, t_eff = b2_host_overhead(int(cfg["budget"]["window"]), float(cfg["budget"]["rho"]))
    seg.to_csv(out / "segmen.csv", index=False, float_format="%.5f")
    diff.to_csv(out / "selisih.csv", index=False, float_format="%.5f")
    parity.to_csv(out / "paritas_rantai.csv", index=False, float_format="%.3g")
    (out / "clock_sebelum.txt").write_text(clock_before); (out / "clock_sesudah.txt").write_text(clock_after)
    tab = seg.groupby(["varian", "segmen"], sort=False)["mean"].mean().unstack()[list(SEGMENTS)]
    l2_txt = ""
    if recs_l2:
        rows = []
        for key, rec2 in recs_l2.items():
            bi, conf = key.split("|")
            for k, d in rec2.items():
                for sg, v in d.items():
                    v = np.asarray(v)
                    rows.append(dict(build=bi, conf=float(conf), varian=k, segmen=sg, mean=v.mean(), std=v.std(),
                                     p50=np.percentile(v, 50), p95=np.percentile(v, 95), n=int(v.size)))
        L2 = pd.DataFrame(rows)
        L2.to_csv(out / "latensi_l2.csv", index=False, float_format="%.5f")
        pd.DataFrame(dec).to_csv(out / "keputusan_l2.csv", index=False)
        summ = {}
        for conf, g in L2[L2.segmen == "wall"].groupby("conf"):
            T = g.groupby("varian")["mean"].mean().to_dict()
            TV, TP, TR = T["Vanilla"], T["P2-Static"], T["Router B2 efisien (L2)"]
            meta = [v for k, v in l2_meta.items() if k.endswith(f"|{conf}")]
            summ[str(conf)] = dict(T=T, f_L2=float(np.clip((TR - TV) / (TP - TV), 0, 1)),
                                   hemat_vs_P2S_persen=100 * (1 - TR / TP), tambah_vs_V_persen=100 * (TR / TV - 1),
                                   rute_B_minus_V=T["A3 rute B (potong 8.400)"] - TV, rute_A_minus_P2S=T["A3 rute A"] - TP,
                                   potong_minus_tanpa=T["A3 rute B (potong 8.400)"] - T["A3 rute B (tanpa potong)"],
                                   **{k: float(np.mean([m[k] for m in meta])) for k in meta[0]})
        write_json(out / "l2.json", dict(per_conf=summ, n_images=len(files), n_reps=a.l2_reps, n_builds=a.n_builds,
                                         nms=dict(nms_kw, multi_label="conf < 0,01"), b2=b2_args,
                                         letterbox_vs_cpu=lb_chk))
        tl2 = L2.groupby(["conf", "varian", "segmen"], sort=False)["mean"].mean().unstack()
        l2_txt = "\n".join(["", "Latensi L2 (ms per gambar, rata-rata build; pre = salin uint8 + letterbox GPU, "
                             "post = keputusan + NMS GPU + salin kotak):", tl2[["pre", "exec", "post", "host", "wall"]].round(4).to_string(), "",
                             "Ringkasan L2 per conf:"] + [f"  conf {c}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else "" for k, v in d.items() if k != "T")
                                                         for c, d in summ.items()])
    txt = "\n".join([
        f"Profil latensi per segmen ({prec}, {'TensorRT ' + str(torch.cuda.get_device_name(0)) if cuda else 'ONNX Runtime CPU'}), "
        f"{a.n_builds} build × {len(imgs)} gambar × {a.n_reps} ulangan; ms per gambar (rata-rata build):",
        tab.round(4).to_string(), "",
        "Selisih (ms; positif = varian pertama lebih lambat):", diff.round(4).to_string(index=False), "",
        f"B2 host: np.quantile {t_np:.4f} ms | jendela terurut {t_eff:.4f} ms per gambar", "",
        "Paritas rantai vs A3 (engine FP16 berbeda; selisih kecil wajar):", parity.to_string(index=False), "",
        "Clock sebelum:", clock_before, "Clock sesudah:", clock_after]) + l2_txt
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights, a.gate, a.vanilla, a.p2static],
                                                   extra=dict(n_images=len(imgs), n_warmup=a.n_warmup, n_reps=a.n_reps,
                                                              n_builds=a.n_builds, precision=prec, build_s=build_s,
                                                              b2_host_ms=dict(np_quantile=t_np, efisien=t_eff))))
    print("✅", out)


if __name__ == "__main__":
    main()
