"""
Cek last.pt vs best.pt di VAL untuk Branch A dan B (best.pt Ultralytics dipilih dari fitness Branch A saja).

    python scripts/check_checkpoint.py --config configs/kitti.yaml --run RUN --out RUN/cek_checkpoint

last.pt tetap dipakai semua tahap (dikunci sebelum training); ini laporan, bukan seleksi. Test tidak dipakai.
Keluaran: titik.csv (metrik COCO 4 sumber), uji.csv (bootstrap berpasangan + CI 95%), plafon_oracle.csv,
lintasan_epoch.csv (fitness A per epoch + mAP B yang tercatat tiap N epoch), ringkasan.txt, provenance.json.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import torch

from rfinal.ckcheck import best_epoch_from_results, branch_b_log, evaluate_pair, same_weights, verdict
from rfinal.data import load_data, make_loader
from rfinal.env import load_config, provenance, require_stock_ultralytics, write_json
from rfinal.model import load_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="folder run training (berisi weights/, results.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    require_stock_ultralytics()
    cfg = load_config(a.config)
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    w_last, w_best = run / "weights" / "last.pt", run / "weights" / "best.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

    ep_best, traj = best_epoch_from_results(run)
    blog = branch_b_log(run)
    lines = [f"epoch best.pt (fitness A, results.csv): {ep_best}"]
    if traj is not None:
        if blog is not None:
            traj = traj.merge(blog.add_prefix("B_").rename(columns={"B_epoch": "epoch"}), on="epoch", how="left")
        traj.to_csv(out / "lintasan_epoch.csv", index=False, float_format="%.6g")
        lines.append(f"epoch terakhir: {int(traj['epoch'].iloc[-1])}")
    if blog is not None:
        j = blog["mAP50-95"].idxmax()
        lines.append(f"Branch B (tercatat tiap N epoch): mAP50-95 maks di epoch {int(blog.epoch[j])} "
                     f"({blog['mAP50-95'][j]:.4f}); epoch terakhir {blog['mAP50-95'].iloc[-1]:.4f}")

    if not w_best.exists() or same_weights(w_last, w_best):
        lines.append("best.pt tidak ada / identik dengan last.pt -> tidak ada yang perlu dibandingkan")
    else:
        data = load_data(cfg["data"], cfg.get("expected_counts"))
        loader = make_loader(data, "val", 1, int(cfg["train"]["imgsz"]), a.workers)
        models = {"last": load_detector(w_last, device)[0], "best": load_detector(w_best, device)[0]}
        titik, uji, plafon = evaluate_pair(models, loader, device, cfg["gate"]["nms"], out / "gt_val.json",
                                           tuple(cfg["eval"]["coco_max_dets"]), rho=float(cfg["budget"]["rho"]),
                                           n_boot=a.n_boot, seed=a.seed, n_jobs=a.jobs)
        titik.to_csv(out / "titik.csv", index=False, float_format="%.6g")
        uji.to_csv(out / "uji.csv", index=False, float_format="%.6g")
        plafon.to_csv(out / "plafon_oracle.csv", index=False, float_format="%.6g")
        lines += ["", titik.round(4).to_string(index=False), "", verdict(uji, "mAP_small"),
                  verdict(uji, "mAP50-95"), "", plafon.round(4).to_string(index=False),
                  "", "Catatan: last.pt tetap dipakai (aturan terkunci). ⚠ = bahas di Bab IV/lampiran, bukan ganti."]
    txt = "\n".join(lines)
    print(txt); (out / "ringkasan.txt").write_text(txt + "\n")
    write_json(out / "provenance.json", provenance(files=[cfg["_config_path"], w_last, w_best],
                                                   extra=dict(n_boot=a.n_boot, seed=a.seed, split="val")))
    print("✅", out)


if __name__ == "__main__":
    main()
