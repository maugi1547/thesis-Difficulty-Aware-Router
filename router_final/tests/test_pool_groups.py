"""Pooling multi-seed di suite merge: grup = nama tanpa _seed<n>; hanya grup >= 2 model (A10 VisDrone, BDD, A6, KITTI)."""
from rfinal.suite import pool_groups


def test_pool_groups_terpisah_per_dataset():
    names = ["kitti_seed42", "kitti_seed7", "kitti_rekaman_seed42", "kitti_rekaman_seed7", "visdrone_seed42",
             "visdrone_seed7", "visdrone_seed2026", "bdd100k_seed42", "bdd100k_seed42_baseline_best",
             "bdd100k_seed7_baseline_best"]
    G = pool_groups("/x", [dict(name=n) for n in names])
    assert {g: [p.name for p in v] for g, v in G.items()} == {
        "kitti": ["kitti_seed42", "kitti_seed7"],
        "kitti_rekaman": ["kitti_rekaman_seed42", "kitti_rekaman_seed7"],
        "visdrone": ["visdrone_seed42", "visdrone_seed7", "visdrone_seed2026"],
        "bdd100k_baseline_best": ["bdd100k_seed42_baseline_best", "bdd100k_seed7_baseline_best"]}
