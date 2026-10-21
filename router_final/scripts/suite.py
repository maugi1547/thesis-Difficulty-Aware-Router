"""
Evaluasi final terkunci (H1–H7, bootstrap berpasangan, pembanding, latensi ujung-ke-ujung) — port Tahap 4/4B.

Satu model (satu sesi Kaggle T4; setelah train.py + fit_gate.py):
    python scripts/suite.py run --config configs/kitti.yaml --seed 42 --weights RUN/weights/last.pt \
        --gate RUN/gate/gate.json --vanilla VAN/weights/last.pt --p2static P2S/weights/last.pt \
        --out /kaggle/working/suite --zip /kaggle/working
Gabungan semua model (CPU cukup; folder/zip hasil run dari sesi lain):
    python scripts/suite.py merge --out /kaggle/working/suite --models kitti_seed42 kitti_seed7 kitti_seed2026 \
        visdrone_seed42 --glob "/kaggle/input/**/suite_*"

Uji lokal tanpa GPU: --override backend=ort precision=fp32 n_builds=1 n_prof_images=4 n_warmup=1 n_reps=1 n_boot=50
Keluaran per model: hasil.json (kriteria), bootstrap_test.csv, coco_lengkap.csv, latensi*.csv, per_gambar_test.csv,
bootstrap_klaster.csv (VisDrone), provenance.json; zip hasil + zip intermediate (analisis ulang tanpa GPU: --reuse).
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
import yaml

from rfinal.env import ROOT, load_config, provenance, require_stock_ultralytics, write_json


def suite_cfg(cfg, overrides):
    c = dict(cfg["analysis"], imgsz=int(cfg["train"]["imgsz"]), rho=float(cfg["budget"]["rho"]),
             b2_window=int(cfg["budget"]["window"]), b2_min_hist=int(cfg["budget"]["min_hist"]),
             opset=int(cfg["deploy"]["opset"]), workers=2)
    assert c["precision"] == cfg["deploy"]["precision"], "analysis.precision != deploy.precision"
    c.update(overrides)
    return c


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--seed", type=int, required=True)
    r.add_argument("--weights", required=True); r.add_argument("--gate", required=True)
    r.add_argument("--vanilla", required=True); r.add_argument("--p2static", required=True)
    r.add_argument("--out", required=True); r.add_argument("--name", default=None)
    r.add_argument("--zip", default=None, help="folder tujuan zip hasil + intermediate")
    r.add_argument("--reuse", default=None, help="folder berisi **/<nama>/stageA.pkl dst. (lewati tahap A / job)")
    r.add_argument("--override", nargs="*", default=[], help="kunci analysis, mis. backend=ort n_boot=50")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True); m.add_argument("--models", nargs="+", required=True)
    m.add_argument("--glob", default=None)
    a = ap.parse_args()

    from rfinal import suite as S
    if a.cmd == "merge":
        G = S.merge(a.out, [dict(name=n) for n in a.models], a.glob)
        out = Path(a.out)
        G["TAB"].to_csv(out / "ringkasan_tab.csv"); G["FULL"].to_csv(out / "ringkasan_full.csv")
        G["H3"].to_csv(out / "ringkasan_h3.csv", index=False); G["LATP"].to_csv(out / "ringkasan_latp.csv")
        if len(G["POOL"]):
            G["POOL"].to_csv(out / "ringkasan_pool.csv", index=False)
        S.pareto_plot(out, G["R"], G["FULL"], out / "pareto.png")
        print(G["TAB"].T.to_string())
        print(G["H3"].round(4).to_string(index=False))
        if len(G["POOL"]):
            print("Pooling KITTI:\n" + G["POOL"].round(4).to_string(index=False))
        return

    require_stock_ultralytics()
    cfg = load_config(a.config)
    ov = {k: yaml.safe_load(v) for k, v in (it.split("=", 1) for it in a.override)}
    c = suite_cfg(cfg, ov)
    if a.reuse:
        c["reuse_dir"] = a.reuse
    spec = dict(name=a.name or f"{cfg['name']}_seed{a.seed}", data_yaml=cfg["data"], seed=a.seed, weights=a.weights,
                gate=a.gate, vanilla_ckpt=a.vanilla, p2static_ckpt=a.p2static, max_det=int(c["max_det"]),
                coco_max_dets=list(cfg["eval"]["coco_max_dets"]), expected_counts=cfg.get("expected_counts") or None)
    out = Path(a.out); (out / spec["name"]).mkdir(parents=True, exist_ok=True)
    write_json(out / spec["name"] / "provenance.json",
               provenance(files=[cfg["_config_path"], a.weights, a.gate, a.vanilla, a.p2static],
                          extra=dict(suite_cfg=c, spec=spec)))
    S.run_model(spec, c, out, ROOT, log=lambda *x: print(*x, flush=True))
    if a.zip:
        S.zip_model(out, spec["name"], a.zip, intermediate=True)
    print("✅", out / spec["name"], json.dumps(json.loads((out / spec["name"] / "hasil.json").read_text())["kriteria"],
                                               indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
