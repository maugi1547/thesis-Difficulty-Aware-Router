"""
Uji reproduksi training: jalankan resep yang SAMA dua kali (proses terpisah) lalu bandingkan sidik jari tensor
last.pt (state_hash.json). Lolos = bobot identik bit per bit.

    # di Kaggle, sebelum training 100 epoch (≈ 2 × 2 epoch):
    python scripts/check_reproducibility.py --config configs/kitti.yaml --seed 42 --override epochs=2
    # membandingkan dua run yang sudah ada (mis. run 100 epoch hari ini vs ulangan minggu depan):
    python scripts/check_reproducibility.py --compare RUN_DIR_1 RUN_DIR_2

Bila gagal: ulangi dengan --strict-determinism untuk menemukan operasi non-deterministik, dan pastikan GPU, driver,
versi torch/CUDA, jumlah worker, dan batch sama (tercatat di provenance.json kedua run).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import _path  # noqa: F401

HERE = Path(__file__).resolve().parent


def compare(d1, d2):
    h1 = json.loads((Path(d1) / "state_hash.json").read_text())
    h2 = json.loads((Path(d2) / "state_hash.json").read_text())
    ok = h1 == h2
    print(f"{'✅ IDENTIK' if ok else '❌ BERBEDA'}\n  {d1}: {h1}\n  {d2}: {h2}")
    if not ok:
        import torch
        from rfinal.repro import max_abs_diff
        a = torch.load(Path(d1) / "weights/last.pt", map_location="cpu", weights_only=False)["model"]
        b = torch.load(Path(d2) / "weights/last.pt", map_location="cpu", weights_only=False)["model"]
        diff = max_abs_diff(a, b)
        worst = sorted(diff.items(), key=lambda kv: -kv[1])[:5]
        print("  selisih absolut terbesar:", worst)
        p1 = json.loads((Path(d1) / "provenance.json").read_text())
        p2 = json.loads((Path(d2) / "provenance.json").read_text())
        for k in ("versions", "device", "env"):
            if p1.get(k) != p2.get(k):
                print(f"  provenance '{k}' berbeda:\n    {p1.get(k)}\n    {p2.get(k)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--override", nargs="*", default=["epochs=2"])
    ap.add_argument("--strict-determinism", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar="RUN_DIR")
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    assert a.config, "--config wajib (atau --compare)"
    tag = time.strftime("%Y%m%d_%H%M%S")
    dirs = []
    for r in (1, 2):
        name = f"repro_{tag}_run{r}"
        cmd = [sys.executable, str(HERE / "train.py"), "--config", a.config, "--seed", str(a.seed), "--name", name,
               "--override", *a.override] + (["--strict-determinism"] if a.strict_determinism else [])
        print("▶", " ".join(cmd), flush=True)
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("✅ selesai:")]
        assert line, out.stdout[-3000:] + out.stderr[-3000:]
        dirs.append(line[-1].split(":", 1)[1].strip())
    sys.exit(0 if compare(*dirs) else 1)


if __name__ == "__main__":
    main()
