"""
A12 — latensi ujung-ke-ujung L1 dengan engine A3T (keluaran ditranspos; rute B menyalin awalan 8.400 anchor saja),
dibandingkan router A3 asli DALAM SESI YANG SAMA, lalu (opsional) H5 pada f baru dari pass akurasi suite.

    python scripts/a12_engine_tanpa_padding.py --config configs/kitti.yaml --weights RUN/weights/last.pt \
        --gate RUN/gate/gate.json --vanilla VAN.pt --p2static P2S.pt --out /kaggle/working/a12_kitti_seed42 \
        [--suite-dir SUITE_INTER/kitti_seed42 --ref SUITE/kitti_seed42/bootstrap_test.csv]

Keluaran: latensi_a12.csv (per build × varian: mean, std, p50, p95, n), a12.json (T rata-rata build, f kontrol, f A12,
Δ, paritas A3T vs A3, keputusan vs pass akurasi), keputusan_l2.csv (format h6_latency), ringkasan.txt, provenance.json;
bila --suite-dir: h6/ (h6_latency.py: H5 pada f kontrol dan f A12, paritas L1 vs suite).

Aturan baca (daftar tindakan A12, ditulis sebelum dijalankan): H5 dibaca dengan aturan yang sama (CI > 0 = unggul).
Karena engine berubah, hasilnya rancangan lanjutan, bukan pengganti kondisi utama. Tambahan (ditulis di sini, sebelum
run): yang dilaporkan sebagai efek A12 adalah f_A12 − f_kontrol dari sesi yang sama; f suite hanya pembanding.
"""
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd
import torch

from rfinal.a12 import E2E, ROUTERS, e2e_pass, make_ops, parity, transpose_output
from rfinal.baselines import load_static
from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, require_stock_ultralytics, write_json
from rfinal.gate import load_gate
from rfinal.model import load_detector
from rfinal.profile import TrtEng, gpu_clock_snapshot
from rfinal.suite import _OrtEng, export_suite

ENGINES = ("vanilla", "p2static", "a3g", "a3t")


def main():
    ap = argparse.ArgumentParser()
    for k in ("config", "weights", "gate", "vanilla", "p2static", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n-warmup", type=int, default=30)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--n-builds", type=int, default=2)
    ap.add_argument("--n-images", type=int, default=0, help="0 = seluruh test")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--suite-dir", default=None, help="intermediate suite model yang sama -> H5 via h6_latency.py")
    ap.add_argument("--ref", default=None, help="bootstrap_test.csv suite (paritas H5 L1)")
    a = ap.parse_args()
    require_stock_ultralytics()
    cfg = load_config(a.config)
    out = Path(a.out); (out / "onnx").mkdir(parents=True, exist_ok=True); (out / "engines").mkdir(exist_ok=True)
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")

    data = load_data(cfg["data"], cfg.get("expected_counts"))
    loader = make_loader(data, "test", 1, int(cfg["train"]["imgsz"]), a.workers)
    stream, files = [], []
    for b in loader:
        stream.append(b["img"].clone()); files.append(Path(b["im_file"][0]).name)
        if a.n_images and len(stream) >= a.n_images:
            break
    print(f"[1/4] {len(stream)} gambar test (urutan aliran), ekspor ONNX ...", flush=True)
    model, _ = load_detector(a.weights, dev)
    van, p2s = load_static(a.vanilla, dev, data["nc"]), load_static(a.p2static, dev, data["nc"])
    gate = load_gate(a.gate)
    paths, mods = export_suite(model, van, p2s, gate, stream[0].float() / 255.0, out / "onnx", dev, ("fp16",) if cuda else ())
    prec = "fp16" if "fp16" in paths else "fp32"
    P, n_a, n_b = dict(paths[prec]), int(mods["n_a"]), int(mods["n_b"])
    P["a3t"] = out / "onnx" / f"a3t_{prec}.onnx"
    tinfo = transpose_output(P["a3g"], P["a3t"])
    del model, van, p2s, mods; gc.collect()

    an = cfg.get("analysis", {})
    rho, W, MH = float(cfg["budget"]["rho"]), int(cfg["budget"]["window"]), int(cfg["budget"]["min_hist"])
    thr0 = float(gate["thr0"])
    s = torch.cuda.Stream() if cuda else None
    sync = s.synchronize if cuda else (lambda: None)
    dsync = torch.cuda.synchronize if cuda else (lambda: None)
    ctx = (lambda: torch.cuda.stream(s)) if cuda else (lambda: __import__("contextlib").nullcontext())
    pin = (lambda t: torch.zeros(t.shape, dtype=t.dtype).pin_memory()) if cuda else (lambda t: torch.zeros(t.shape, dtype=t.dtype))
    clock0 = gpu_clock_snapshot()
    rows, par, dec, build_s = [], None, {"file": files}, {}
    for bi in range(a.n_builds):
        print(f"[2/4] build {bi}: engine {', '.join(ENGINES)} ({prec}) ...", flush=True)
        eng = {}
        for k in ENGINES:
            if cuda:
                from rfinal.trt import build_engine
                info = build_engine(P[k], out / "engines" / f"{k}_b{bi}.engine")
                build_s[f"b{bi}|{k}"] = info["build_s"]; eng[k] = TrtEng(info["engine"], dev)
            else:
                eng[k] = _OrtEng(P[k])
        ops, H = make_ops(eng, n_a, n_b, dev, pin, s, sync)
        if bi == 0:
            with (torch.cuda.stream(s) if cuda else __import__("contextlib").nullcontext()):
                par = parity(eng, ops, H, stream[:8], n_b)
            print("    paritas A3T vs A3:", {k: f"{v:.3g}" for k, v in par.items()}, flush=True)
        print(f"[3/4] build {bi}: e2e {len(E2E)} varian × {len(stream)} gambar × {a.n_reps} ulangan ...", flush=True)
        lat, use = e2e_pass(ops, H, stream, rho, W, MH, thr0, a.n_warmup, a.n_reps, ctx, dsync)
        for k, v in lat.items():
            v = np.asarray(v)
            rows.append(dict(build=f"b{bi}", varian=k, mean=v.mean(), std=v.std(), p50=np.percentile(v, 50),
                             p95=np.percentile(v, 95), n=int(v.size)))
        dec[f"L2_b{bi}_ckontrol"] = use[ROUTERS[0]]
        dec[f"L2_b{bi}_cA12"] = use[ROUTERS[1]]
        print("    aktivasi:", {k: round(float(np.mean(v)), 4) for k, v in use.items()},
              "| A3 vs A3T beda keputusan:", int(np.sum(np.asarray(use[ROUTERS[0]]) != np.asarray(use[ROUTERS[1]]))), flush=True)
        del eng, ops, H; gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    clock1 = gpu_clock_snapshot()

    print("[4/4] ringkasan ...", flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(out / "latensi_a12.csv", index=False, float_format="%.5f")
    pd.DataFrame(dec).to_csv(out / "keputusan_l2.csv", index=False)
    T = L.groupby("varian")["mean"].mean().to_dict()
    TV, TP = T["Vanilla"], T["P2-Static"]
    f = lambda tr: float(np.clip((tr - TV) / (TP - TV), 0, 1))
    act = {k: float(np.mean([np.mean(dec[c]) for c in dec if c.endswith(tag)])) for k, tag in
           (("kontrol", "_ckontrol"), ("A12", "_cA12"))}
    per_build = L.pivot_table(index="varian", columns="build", values="mean")
    summ = {
        "kontrol": dict(label="H5 L1 sesi ini, A3 asli (kontrol)", T={k: T[k] for k in ("Vanilla", "P2-Static", ROUTERS[0])},
                        f_L2=f(T[ROUTERS[0]]), aktivasi=act["kontrol"]),
        "A12": dict(label="H5 L1 sesi ini, A3T dua tahap (A12)", T={k: T[k] for k in ("Vanilla", "P2-Static", ROUTERS[1])},
                    f_L2=f(T[ROUTERS[1]]), aktivasi=act["A12"])}
    diag = dict(rute_B_minus_V=T["A3 rute B"] - TV, A3T_rute_B_awalan_minus_V=T["A3T rute B (salin awalan)"] - TV,
                A3T_rute_B_penuh_minus_V=T["A3T rute B (salin penuh)"] - TV, rute_A_minus_P2S=T["A3 rute A"] - TP,
                A3T_rute_A_minus_P2S=T["A3T rute A"] - TP,
                biaya_transpose_rute_B=T["A3T rute B (salin penuh)"] - T["A3 rute B"],
                hemat_awalan_rute_B=T["A3T rute B (salin penuh)"] - T["A3T rute B (salin awalan)"],
                router_A12_minus_kontrol=T[ROUTERS[1]] - T[ROUTERS[0]],
                f_A12_minus_f_kontrol=summ["A12"]["f_L2"] - summ["kontrol"]["f_L2"],
                per_build_router_A12_minus_kontrol={b: float(per_build.loc[ROUTERS[1], b] - per_build.loc[ROUTERS[0], b])
                                                   for b in per_build.columns})
    write_json(out / "a12.json", dict(per_conf=summ, diagnostik=diag, paritas=par, onnx_transpose=tinfo, n_a=n_a, n_b=n_b,
                                      n_images=len(stream), n_reps=a.n_reps, n_builds=a.n_builds, precision=prec))
    tab = L.groupby("varian", sort=False)[["mean", "p50", "p95"]].mean().reindex(list(E2E))
    lines = [f"A12 — latensi ujung-ke-ujung L1 ({prec}, {torch.cuda.get_device_name(0) if cuda else 'ORT CPU'}), "
             f"{a.n_builds} build × {len(stream)} gambar × {a.n_reps} ulangan; ms per gambar (rata-rata build):",
             tab.round(4).to_string(), "", "Per build (mean):", per_build.reindex(list(E2E)).round(4).to_string(), "",
             "f: " + json.dumps({k: round(v["f_L2"], 4) for k, v in summ.items()}),
             "aktivasi: " + json.dumps({k: round(v, 4) for k, v in act.items()}),
             "diagnostik (ms): " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()}, ensure_ascii=False),
             "paritas A3T vs A3: " + json.dumps({k: float(f"{v:.3g}") for k, v in par.items()}, ensure_ascii=False),
             "", "Clock sebelum:", clock0, "Clock sesudah:", clock1]
    txt = "\n".join(lines); print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], a.weights, a.gate, a.vanilla, a.p2static],
                                                   extra=dict(build_s=build_s, precision=prec)))
    if a.suite_dir:
        cmd = [sys.executable, str(Path(__file__).parent / "h6_latency.py"), "--config", a.config, "--suite-dir", a.suite_dir,
               "--l2-json", str(out / "a12.json"), "--out", str(out / "h6")] + (["--ref", a.ref] if a.ref else [])
        print("H5 pada f kontrol & f A12:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)
    print("✅", out)


if __name__ == "__main__":
    main()
