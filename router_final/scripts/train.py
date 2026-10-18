"""
Tahap 1 — training detektor dual-branch.

    python scripts/train.py --config configs/kitti.yaml --seed 42
    python scripts/train.py --config configs/kitti.yaml --seed 42 --override epochs=2 fraction=0.1 name=smoke

Keluaran (folder run Ultralytics): weights/last.pt, results.csv, args.yaml, provenance.json, branch_b_metrics.csv,
weights_sha256.json, state_hash.json (sidik jari tensor, dipakai uji reproduksi).
"""
import argparse

import _path  # noqa: F401
import yaml

from rfinal.env import load_config
from rfinal.train import build_trainer


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
    ap.add_argument("--override", nargs="*", help="argumen Ultralytics tambahan, mis. epochs=2 device=cpu")
    ap.add_argument("--strict-determinism", action="store_true",
                    help="operasi non-deterministik -> error (diagnosis bila uji reproduksi gagal)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    ov = parse_overrides(a.override)
    name = a.name or ov.pop("name", None)
    trainer = build_trainer(cfg, a.seed, name=name, overrides=ov, strict_determinism=a.strict_determinism)
    trainer.train()
    print("✅ selesai:", trainer.save_dir)


if __name__ == "__main__":
    main()
