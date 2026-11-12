"""A6: split per rekaman terkunci (configs/a6) + label klaster map_csv + build symlink pada dataset sintetis."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import a6_split_rekaman as A  # noqa: E402

from rfinal.cocomix import cluster_index, cluster_labels  # noqa: E402
from rfinal.env import load_config  # noqa: E402


def test_split_terkunci_konsisten():
    T = pd.read_csv(A.SPLIT_CSV, dtype=str)
    meta = json.loads(A.SPLIT_CSV.with_suffix(".json").read_text())
    for f, h in meta["sha256"].items():
        assert A.sha256(A.A6 / f) == h
    assert len(T) == 7481 and T.image_id.is_unique
    assert set(T[T.split_a6 == "test"].image_id) == set(A.read_list(A.A6 / "chen_val.txt"))
    assert set(T[T.split_a6 != "test"].image_id) == set(A.read_list(A.A6 / "chen_train.txt"))
    s = A.summarize(T)
    assert all(v == 0 for v in s["irisan_rekaman"].values())
    cfg = load_config(ROOT / "configs" / "kitti_rekaman.yaml")
    assert {k: s[k]["citra"] for k in A.SPLITS} == cfg["expected_counts"]
    # rekaman di peta = rekaman di split
    M = A.load_map()
    assert dict(zip(M.image_id, M.rekaman)) == dict(zip(T.image_id, T.rekaman))


def test_generate_mereproduksi_berkas_terkunci():
    st = A.A6 / "kitti_label_stat.csv"
    if not st.exists():
        pytest.skip("kitti_label_stat.csv tidak ada")
    M = A.load_map()
    T, info = A.make_split(M, A.stats_from_csv(st, M), A.read_list(A.A6 / "chen_train.txt"),
                           A.read_list(A.A6 / "chen_val.txt"), 20261112, 20000, 0.20, 20)
    assert pd.read_csv(A.SPLIT_CSV, dtype=str).equals(T.astype(str))


def test_config_hanya_berbeda_di_kunci_a6():
    a = load_config(ROOT / "configs" / "kitti.yaml"); b = load_config(ROOT / "configs" / "kitti_rekaman.yaml")
    for k in set(a) | set(b):
        if k in ("name", "data", "expected_counts", "_config_path", "analysis"):
            continue
        assert a[k] == b[k], k
    extra = {k: v for k, v in b["analysis"].items() if k not in a["analysis"]}
    assert set(extra) == {"cluster_by", "cluster_map", "cluster_seed"}
    assert {k: v for k, v in b["analysis"].items() if k in a["analysis"]} == a["analysis"]
    assert Path(b["analysis"]["cluster_map"]).exists()


def test_cluster_labels_map_dan_prefix():
    cfg = load_config(ROOT / "configs" / "kitti_rekaman.yaml")
    lab = cluster_labels(["000000.png", "x/000001.png"], "map_csv", cfg["analysis"]["cluster_map"])
    assert lab == ["2011_09_28_drive_0106_sync", "2011_09_26_drive_0101_sync"]
    assert cluster_labels(["uav0001_00001.jpg"], "video_prefix") == ["uav0001"]
    with pytest.raises(AssertionError):
        cluster_labels(["999999.png"], "map_csv", cfg["analysis"]["cluster_map"])
    idx = cluster_index([np.array([0, 1]), np.array([2])], 3, 1)
    assert len(idx) == 3


def test_build_symlink_dataset_sintetis(tmp_path, monkeypatch):
    """build: sumber images/labels per split asal -> keluaran per split A6, nama berkas tetap, yaml, jumlah per split."""
    T = pd.read_csv(A.SPLIT_CSV, dtype=str)
    src = tmp_path / "kaggle" / "kitti-2class"
    for r in T.itertuples():
        for kind, ext in (("images", ".png"), ("labels", ".txt")):
            p = src / kind / r.split_asal / f"{r.image_id}{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("0 0.5 0.5 0.1 0.1\n" if kind == "labels" else "")
    (src / "kitti_2class.yaml").write_text("names:\n  0: Person\n  1: Vehicle\n")
    out = tmp_path / "kitti_rekaman"
    A.main(["build", "--root", str(tmp_path / "kaggle"), "--out", str(out)])
    rep = json.loads((out / "ringkasan_split.json").read_text())
    assert rep["citra"] == {"train": 2970, "val": 742, "test": 3769} and rep["label"] == 7481
    some = T[T.split_a6 == "test"].iloc[0]
    link = out / "images" / "test" / f"{some.image_id}.png"
    assert link.is_symlink() and link.resolve() == (src / "images" / some.split_asal / f"{some.image_id}.png").resolve()
    y = (out / "kitti_rekaman_2class.yaml").read_text()
    assert f"path: {out.resolve()}" in y and "test: images/test" in y and "1: Vehicle" in y
