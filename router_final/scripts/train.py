"""
Tahap 1 — training detektor dual-branch, atau baseline stok dengan resep yang sama.

    python scripts/train.py --config configs/kitti.yaml --seed 42
    python scripts/train.py --config configs/visdrone.yaml --seed 42 --model vanilla     # -> visdrone_vanilla_seed42
    python scripts/train.py --config configs/visdrone.yaml --seed 42 --model p2static    # -> visdrone_p2static_seed42
    python scripts/train.py --config configs/kitti.yaml --seed 42 --override epochs=2 fraction=0.1 name=smoke

Keluaran (folder run Ultralytics): weights/last.pt, results.csv, args.yaml, provenance.json, branch_b_metrics.csv,
weights_sha256.json, state_hash.json (sidik jari tensor, dipakai uji reproduksi).
"""
import argparse

import _path  # noqa: F401
import yaml

from rfinal.env import load_config
from rfinal.train import build_baseline_trainer, build_trainer


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="dualbranch", choices=("dualbranch", "vanilla", "p2static"))
    ap.add_argument("--override", nargs="*", help="argumen Ultralytics tambahan, mis. epochs=2 device=cpu")
    ap.add_argument("--strict-determinism", action="store_true",
                    help="operasi non-deterministik -> error (diagnosis bila uji reproduksi gagal)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ov = parse_overrides(a.override)
    name = a.name or ov.pop("name", None)
    if a.model == "dualbranch":
        trainer = build_trainer(cfg, a.seed, name=name, overrides=ov, strict_determinism=a.strict_determinism)
    else:
        trainer = build_baseline_trainer(cfg, a.seed, a.model, name=name, overrides=ov,
                                         strict_determinism=a.strict_determinism)
    trainer.train()
    print("✅ selesai:", trainer.save_dir)


if __name__ == "__main__":
    main()
