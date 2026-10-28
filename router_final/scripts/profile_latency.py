"""
Profil latensi per segmen (diagnostik Subbab 4.8.2, di luar kriteria terkunci H1–H7).

    python scripts/profile_latency.py --config configs/kitti.yaml --weights RUN/weights/last.pt \
        --gate RUN/gate/gate.json --vanilla VAN.pt --p2static P2S.pt --out /kaggle/working/profil_kitti_seed42

Mengekspor ONNX FP16 dengan export_suite (sama dengan suite.py), membangun engine TensorRT (vanilla, p2static, a3g,
s1g, s2a, s2b) sebanyak --n-builds kali, lalu mengukur tiap varian dalam segmen h2d / exec / d2h / host (rfinal/profile.py).
Keluaran: segmen.csv, selisih.csv, paritas_rantai.csv, ringkasan.txt, clock_sebelum.txt, clock_sesudah.txt,
provenance.json. Tanpa GPU: backend ONNX Runtime FP32 di CPU (uji alur saja, angka tidak bermakna).
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
    P, n_b = paths[prec], int(mods["n_b"])
    del model, van, p2s, mods; gc.collect()

    s = torch.cuda.Stream() if cuda else None
    pin = (lambda t: torch.zeros(t.shape, dtype=t.dtype).pin_memory()) if cuda else (lambda t: torch.zeros(t.shape, dtype=t.dtype))
    clock_before = gpu_clock_snapshot()
    recs, build_s, parity = {}, {}, None
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
    txt = "\n".join([
        f"Profil latensi per segmen ({prec}, {'TensorRT ' + str(torch.cuda.get_device_name(0)) if cuda else 'ONNX Runtime CPU'}), "
        f"{a.n_builds} build × {len(imgs)} gambar × {a.n_reps} ulangan; ms per gambar (rata-rata build):",
        tab.round(4).to_string(), "",
        "Selisih (ms; positif = varian pertama lebih lambat):", diff.round(4).to_string(index=False), "",
        f"B2 host: np.quantile {t_np:.4f} ms | jendela terurut {t_eff:.4f} ms per gambar", "",
        "Paritas rantai vs A3 (engine FP16 berbeda; selisih kecil wajar):", parity.to_string(index=False), "",
        "Clock sebelum:", clock_before, "Clock sesudah:", clock_after])
    print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights, a.gate, a.vanilla, a.p2static],
                                                   extra=dict(n_images=len(imgs), n_warmup=a.n_warmup, n_reps=a.n_reps,
                                                              n_builds=a.n_builds, precision=prec, build_s=build_s,
                                                              b2_host_ms=dict(np_quantile=t_np, efisien=t_eff))))
    print("✅", out)


if __name__ == "__main__":
    main()
