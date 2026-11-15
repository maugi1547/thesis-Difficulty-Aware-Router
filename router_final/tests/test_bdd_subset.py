"""BDD100K subset: alokasi bertingkat, generate deterministik, build label YOLO 2 kelas pada dataset sintetis."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import bdd_subset as B  # noqa: E402

from rfinal.env import load_config  # noqa: E402

PIL = pytest.importorskip("PIL.Image")


def fake_bdd(base, n=(40, 12, 20), W=64, H=36):
    """Struktur Kaggle alvaromalfaro/bdd100k: <base>/bdd100k/{images/100k,labels}/<split>/; strata tidak seimbang."""
    root = base / "bdd100k"
    tods = ["daytime", "daytime", "daytime", "night"]
    for split, k in zip(("train", "val", "test"), n):
        (root / "images" / "100k" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(k):
            nm = f"{split[:2]}{i:04d}-abcd"
            PIL.new("RGB", (W, H)).save(root / "images" / "100k" / split / f"{nm}.jpg")
            objs = [dict(category="car", id=1, box2d=dict(x1=10, y1=5, x2=30, y2=25)),          # Vehicle, dalam citra
                    dict(category="rider", id=2, box2d=dict(x1=-5, y1=-2, x2=4, y2=8)),         # Person, di-clip
                    dict(category="traffic sign", id=3, box2d=dict(x1=1, y1=1, x2=5, y2=5)),    # dibuang
                    dict(category="bike", id=4, box2d=dict(x1=1, y1=1, x2=9, y2=9)),            # dibuang
                    dict(category="truck", id=5, box2d=dict(x1=70, y1=1, x2=80, y2=9)),         # di luar citra -> dibuang
                    dict(category="lane", id=6, poly2d=[])]                                     # tanpa box2d
            lab = dict(name=nm, attributes=dict(weather="clear", scene="city street" if i % 3 else "highway",
                                                timeofday=tods[i % 4]),
                       frames=[dict(timestamp=10000, objects=objs)])
            (root / "labels" / split / f"{nm}.json").write_text(json.dumps(lab))
    return root


def test_allocate_proporsional_dan_tepat():
    a = B.allocate({"a": 70, "b": 20, "c": 10}, 9)
    assert sum(a.values()) == 9 and a == {"a": 6, "b": 2, "c": 1}
    a = B.allocate({"x": 1, "y": 1, "z": 1}, 2)                    # seri -> nama terurut
    assert a == {"x": 1, "y": 1, "z": 0}
    assert B.allocate({"a": 3, "b": 100}, 103) == {"a": 3, "b": 100}


def test_generate_build_sintetis(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "SPLITS", {"train": ("train", 21), "val": ("val", 6), "test": ("test", 10)})
    src = fake_bdd(tmp_path / "kaggle")
    csv1, csv2 = tmp_path / "s1" / "bdd_subset.csv", tmp_path / "s2" / "bdd_subset.csv"
    B.main(["generate", "--root", str(tmp_path / "kaggle"), "--out-csv", str(csv1)])
    B.main(["generate", "--root", str(src), "--out-csv", str(csv2)])
    T = pd.read_csv(csv1, dtype=str)
    assert T.equals(pd.read_csv(csv2, dtype=str))                   # deterministik, root induk/langsung sama
    assert T.groupby("split").size().to_dict() == {"train": 21, "val": 6, "test": 10}
    assert (T.split == T.split_resmi).all() and T.name.is_unique
    meta = json.loads(csv1.with_suffix(".json").read_text())
    assert meta["csv_sha256"] == B.sha256(csv1)
    # proporsi strata dipertahankan (train: 40 citra, malam 10/40 -> 21 * 0.25 ≈ 5)
    assert abs((T[T.split == "train"].timeofday == "night").sum() - 21 * 10 / 40) <= 1
    # seed lain -> subset lain
    csv3 = tmp_path / "s3" / "x.csv"
    B.main(["generate", "--root", str(src), "--out-csv", str(csv3), "--seed", "1"])
    assert not pd.read_csv(csv3, dtype=str).equals(T)

    out = tmp_path / "bdd100k_2class"
    B.main(["build", "--root", str(src), "--out", str(out), "--csv", str(csv1)])
    nm = T[T.split == "test"].name.iloc[0]
    link = out / "images" / "test" / f"{nm}.jpg"
    assert link.is_symlink() and link.resolve() == (src / "images" / "100k" / "test" / f"{nm}.jpg").resolve()
    L = (out / "labels" / "test" / f"{nm}.txt").read_text().split("\n")
    assert L[0] == "1 0.312500 0.416667 0.312500 0.555556"          # car (10..30, 5..25) pada 64x36
    assert L[1] == "0 0.031250 0.111111 0.062500 0.222222"          # rider di-clip ke (0..4, 0..8)
    assert L[2] == "" and len(L) == 3                                # sign, bike, truck luar citra, lane: dibuang
    rep = json.loads((out / "ringkasan_subset.json").read_text())
    assert {k: v["citra"] for k, v in rep["per_split"].items()} == {"train": 21, "val": 6, "test": 10}
    assert rep["per_split"]["test"]["kecil_per_citra"] == 2.0 and rep["ukuran_citra"] == {"64x36": 37}
    y = (out / "bdd100k_2class.yaml").read_text()
    assert f"path: {out.resolve()}" in y and "test: images/test" in y and "0: Person" in y and "1: Vehicle" in y

    # CSV diubah setelah dikunci -> build menolak
    csv1.write_text(csv1.read_text().replace(nm, nm, 1) + "\n")
    with pytest.raises(AssertionError, match="sha256"):
        B.main(["build", "--root", str(src), "--out", str(out), "--csv", str(csv1)])


def test_build_cek_config(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "SPLITS", {"train": ("train", 4), "val": ("val", 2), "test": ("test", 3)})
    src = fake_bdd(tmp_path / "k", n=(8, 4, 6))
    csv = tmp_path / "c" / "s.csv"
    B.main(["generate", "--root", str(src), "--out-csv", str(csv)])
    out = tmp_path / "o"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"data: {out}/bdd100k_2class.yaml\nexpected_counts: {{train: 4, val: 2, test: 3}}\n"
                   f"eval: {{coco_max_dets: [1, 10, 100]}}\n")
    B.main(["build", "--root", str(src), "--out", str(out), "--csv", str(csv), "--config", str(cfg)])
    cfg.write_text(cfg.read_text().replace("test: 3", "test: 4"))
    with pytest.raises(AssertionError, match="expected_counts"):
        B.main(["build", "--root", str(src), "--out", str(out), "--csv", str(csv), "--config", str(cfg)])
    cfg.write_text(cfg.read_text().replace("test: 4", "test: 3").replace("[1, 10, 100]", "[1, 1]"))
    with pytest.raises(AssertionError, match="coco_max_dets"):
        B.main(["build", "--root", str(src), "--out", str(out), "--csv", str(csv), "--config", str(cfg)])


def test_config_bdd_hanya_berbeda_di_kunci_data():
    a = load_config(ROOT / "configs" / "kitti.yaml"); b = load_config(ROOT / "configs" / "bdd100k.yaml")
    beda = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert beda <= {"name", "data", "expected_counts", "_config_path"}, beda
    assert b["expected_counts"] == {s: n for s, (_, n) in B.SPLITS.items()}
    assert b["name"] == "bdd100k" and b["data"].endswith("/bdd100k_2class/bdd100k_2class.yaml")


def test_subset_terkunci_konsisten():
    """CSV subset BDD100K yang di-commit: sha256, jumlah per split, alokasi strata = isi CSV."""
    if not B.LOCK_CSV.exists():
        pytest.skip("configs/bdd100k/bdd_subset.csv belum di-commit")
    T = pd.read_csv(B.LOCK_CSV, dtype=str)
    meta = json.loads(B.LOCK_CSV.with_suffix(".json").read_text())
    assert meta["csv_sha256"] == B.sha256(B.LOCK_CSV) and meta["seed"] == B.SEED
    assert meta["class_map"] == B.CLASS_MAP
    assert T.name.is_unique and (T.split == T.split_resmi).all()
    assert T.groupby("split").size().to_dict() == {s: n for s, (_, n) in B.SPLITS.items()}
    for s, inf in meta["info"].items():
        d = T[T.split == s]
        assert (d.timeofday + "|" + d.scene).value_counts().to_dict() == {k: v for k, v in inf["alokasi"].items() if v}
