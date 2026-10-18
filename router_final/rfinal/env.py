"""
Penjaga lingkungan & provenance (router_final).

Reproduksi hanya bermakna bila kode yang berjalan sama. Karena itu:
  1. ultralytics HARUS paket stok 8.3.240 dari PyPI — bukan fork `ultralytics/` di root repo ini (fork itu masih
     memuat router lama). File inti yang dipakai training dicocokkan dengan sha256 wheel resmi.
  2. Semua run menulis provenance.json: commit router_final, versi library, GPU, sha256 konfigurasi & data.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ULTRALYTICS_VERSION = "8.3.240"
# sha256 file inti di wheel ultralytics-8.3.240-py3-none-any.whl (PyPI)
STOCK_SHA256 = {
    "nn/tasks.py": "eb7e8c376ed5be5ba921a1cfa55e2bd09ef4e5151d57e82234a8c87896d19312",
    "nn/modules/head.py": "1c02c485bd48e5536a09024507ce0e813f5da5102b20a59b8a0972a21ceb49f7",
    "nn/modules/block.py": "c514a83cfa63a752d929e7aa44608b111cac4f2b9beb1bc3ccc4e522b983854d",
    "utils/loss.py": "475b82d342255d51c559cf08f27823b5f45fb948ffb13fd9c39f4e9582b09856",
    "utils/tal.py": "c3ba22e9fa743662fa84787ecaf0825f5701bae07826e5fbc35c22478fd2319b",
    "engine/trainer.py": "9aa56b84bf319c9c30289563c431228b0bb4587e3c35ee5d078497c65c721e1e",
    "engine/validator.py": "0d1a2ac8f6211f9ac110c2e8f98f7809d88bb7141361a2473f4e1f3fd7be3493",
    "models/yolo/detect/train.py": "f8f1c7ea2efaeff5ca08fb017803a2ec0c5f1e9a2ae39d467e36384d132ea3b7",
}
ROOT = Path(__file__).resolve().parent.parent          # folder router_final/


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def require_stock_ultralytics():
    """Gagal cepat bila ultralytics bukan paket stok 8.3.240 (mis. fork di root repo ikut ter-import)."""
    import ultralytics
    pkg = Path(ultralytics.__file__).resolve().parent
    problems = []
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        problems.append(f"versi {ultralytics.__version__} != {ULTRALYTICS_VERSION}")
    if (pkg / "model_yolov8_router.py").exists():
        problems.append(f"{pkg} adalah fork router lama (ada model_yolov8_router.py)")
    for rel, want in STOCK_SHA256.items():
        p = pkg / rel
        if not p.is_file() or sha256_file(p) != want:
            problems.append(f"{rel} berbeda dari wheel stok")
    if problems:
        raise RuntimeError(
            "ultralytics yang ter-import bukan paket stok:\n  - " + "\n  - ".join(problems) +
            f"\nPasang: pip install ultralytics=={ULTRALYTICS_VERSION} ; jalankan skrip dari folder router_final/ "
            "(BUKAN dari root repo, karena folder ultralytics/ di sana menutupi paket pip).")
    return str(pkg)


def git_info(path=ROOT):
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", str(path), *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = g("status", "--porcelain", "--", ".")
    return dict(commit=g("rev-parse", "HEAD"), branch=g("rev-parse", "--abbrev-ref", "HEAD"),
                dirty_files=None if status is None else [l for l in status.splitlines() if l.strip()])


def library_versions():
    import numpy, torch
    v = dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
             cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    for mod in ("ultralytics", "sklearn", "pycocotools", "onnx", "onnxruntime", "tensorrt", "scipy", "cv2"):
        try:
            v[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            v[mod] = None
    return v


def device_info():
    import torch
    if not torch.cuda.is_available():
        return dict(gpu=None, cpu=platform.processor() or platform.machine())
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, gpu_capability=f"{p.major}.{p.minor}", gpu_count=torch.cuda.device_count(),
                driver=_nvidia_driver())


def _nvidia_driver():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                       text=True).strip().splitlines()[0]
    except Exception:
        return None


def provenance(files=(), extra=None):
    """files: path yang di-hash (konfigurasi, data yaml, checkpoint)."""
    return dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv,
        router_final=git_info(),
        ultralytics_path=require_stock_ultralytics(),
        versions=library_versions(),
        device=device_info(),
        env={k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")},
        files={str(f): sha256_file(f) for f in files if f and Path(f).is_file()},
        **(extra or {}),
    )


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def load_config(path):
    """YAML konfigurasi eksperimen (configs/*.yaml) -> dict, dengan path model relatif terhadap configs/."""
    import yaml
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    m = Path(cfg["model"])
    cfg["model"] = str(m if m.is_absolute() else (path.parent / m).resolve())
    cfg["_config_path"] = str(path.resolve())
    return cfg
