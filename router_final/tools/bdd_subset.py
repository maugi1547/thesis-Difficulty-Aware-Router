"""
BDD100K — subset 10.000 citra, 2 kelas (Person, Vehicle), untuk dataset ketiga (pasca-hasil; aturan di rencana-BDD.md).

Sumber: dataset Kaggle alvaromalfaro/bdd100k (BDD100K resmi, label per citra generasi 2018, lisensi data UC Berkeley:
penelitian/pendidikan/nirlaba). Struktur: <root>/images/100k/{train,val,test}/<nama>.jpg dan
<root>/labels/{train,val,test}/<nama>.json ({name, frames:[{objects:[{category, box2d}]}], attributes}).

Subset (dikunci 15 Nov 2026, sebelum pelatihan):
  latih 7.000 dari train resmi | val 1.000 dari val resmi | uji 2.000 dari test resmi (split resmi = split per video).
  Sampling bertingkat per split resmi menurut timeofday × scene (atribut label): alokasi proporsional (sisa terbesar),
  di dalam strata acak dengan rng(--seed 20261115) atas nama terurut. Tidak ada penyaringan berdasarkan isi label.
Pemetaan kelas: Person = person + rider; Vehicle = car + truck + bus + train; bike, motor, traffic light, traffic sign,
  dan kategori lain (lane, drivable area, ...) dibuang. Hanya objek dengan box2d.
Objek kecil = luas kotak < 32² px pada resolusi asli (COCO).

    # 1) sekali, di Kaggle (CPU cukup): membuat CSV subset -> unggah/commit ke configs/bdd100k/bdd_subset.csv
    python tools/bdd_subset.py generate --root /kaggle/input/datasets/alvaromalfaro/bdd100k --out-csv bdd_subset.csv
    # 2) setiap sesi: dataset YOLO dari CSV terkunci (symlink citra + label txt)
    python tools/bdd_subset.py build --root /kaggle/input/datasets/alvaromalfaro/bdd100k \
        --out /kaggle/working/bdd100k_2class --config configs/bdd100k.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = ROOT / "configs" / "bdd100k"
LOCK_CSV = LOCK_DIR / "bdd_subset.csv"
SPLITS = {"train": ("train", 7000), "val": ("val", 1000), "test": ("test", 2000)}   # split subset: (split resmi, jumlah)
CLASS_MAP = {"person": 0, "rider": 0, "car": 1, "truck": 1, "bus": 1, "train": 1}
NAMES = {0: "Person", 1: "Vehicle"}
SMALL_AREA = 32.0 ** 2
SEED = 20261115


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find_root(root):
    root = Path(root)
    ok = lambda p: all((p / "images" / "100k" / s).is_dir() and (p / "labels" / s).is_dir() for s in ("train", "val", "test"))
    if ok(root):
        return root
    cand = sorted({p.parent.parent for p in root.rglob("labels/test") if ok(p.parent.parent)})
    assert len(cand) == 1, f"BDD100K (images/100k + labels per split) tidak ditemukan unik di {root}: {cand}"
    return cand[0]


def read_label(path):
    d = json.loads(Path(path).read_text())
    at = d.get("attributes") or {}
    objs = []
    for fr in d.get("frames") or []:
        for o in fr.get("objects") or []:
            if "box2d" in o and o.get("category") in CLASS_MAP:
                b = o["box2d"]
                objs.append((CLASS_MAP[o["category"]], float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])))
    return dict(timeofday=at.get("timeofday", "undefined"), scene=at.get("scene", "undefined"),
                weather=at.get("weather", "undefined")), objs


def allocate(counts, total):
    """Alokasi proporsional sisa terbesar (Hamilton); seri -> nama strata terurut."""
    keys = sorted(counts)
    n = sum(counts.values())
    raw = {k: total * counts[k] / n for k in keys}
    base = {k: min(int(np.floor(raw[k])), counts[k]) for k in keys}
    rest = total - sum(base.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - np.floor(raw[k])), k)):
        if rest == 0:
            break
        if base[k] < counts[k]:
            base[k] += 1; rest -= 1
    assert rest == 0, "alokasi gagal"
    return base


def stratified_pick(df, total, seed):
    """df: name, strata (timeofday|scene). Returns daftar nama terpilih."""
    cnt = df.strata.value_counts().to_dict()
    alloc = allocate(cnt, total)
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(alloc):
        names = np.array(sorted(df.loc[df.strata == k, "name"]))
        out += list(rng.choice(names, alloc[k], replace=False)) if alloc[k] else []
    return sorted(out), alloc


def cmd_generate(a):
    src = find_root(a.root)
    rows, info = [], {}
    for sub, (off, n) in SPLITS.items():
        recs = []
        for f in sorted((src / "labels" / off).glob("*.json")):
            at, _ = read_label(f)
            recs.append(dict(name=f.stem, **at))
        D = pd.DataFrame(recs); D["strata"] = D.timeofday + "|" + D.scene
        pick, alloc = stratified_pick(D, n, a.seed + {"train": 0, "val": 1, "test": 2}[sub])
        S = D[D.name.isin(set(pick))].copy(); S["split_resmi"] = off; S["split"] = sub
        rows.append(S); info[sub] = dict(populasi=int(len(D)), dipilih=int(len(S)), alokasi=alloc)
        print(f"[{sub}] dari {off}: {len(D)} citra -> {len(S)}", flush=True)
    T = pd.concat(rows)[["name", "split_resmi", "split", "timeofday", "scene", "weather"]].sort_values(["split", "name"])
    assert T.name.is_unique
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(a.out_csv, index=False)
    meta = dict(seed=a.seed, sumber=str(src), info=info, csv_sha256=sha256(a.out_csv), class_map=CLASS_MAP)
    Path(a.out_csv).with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))


def img_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def cmd_build(a):
    csv = Path(a.csv)
    assert csv.exists(), f"CSV subset terkunci tidak ada: {csv} (jalankan `generate` sekali lalu commit)"
    T = pd.read_csv(csv, dtype=str)
    lock = csv.with_suffix(".json")
    if lock.exists():
        assert json.loads(lock.read_text())["csv_sha256"] == sha256(csv), "sha256 CSV subset berubah"
    src = find_root(a.root)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    stats, sizes = [], {}
    for r in T.itertuples():
        si = src / "images" / "100k" / r.split_resmi / f"{r.name}.jpg"
        assert si.exists(), f"citra tidak ada: {si}"
        di = out / "images" / r.split / si.name; di.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copy2 if a.copy else os.symlink)(si.resolve(), di)
        w, h = img_size(si); sizes[(w, h)] = sizes.get((w, h), 0) + 1
        _, objs = read_label(src / "labels" / r.split_resmi / f"{r.name}.json")
        lines, n_small, n_cls = [], 0, [0, 0]
        for c, x1, y1, x2, y2 in objs:
            x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{c} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            n_small += bw * bh < SMALL_AREA; n_cls[c] += 1
        dl = out / "labels" / r.split / f"{r.name}.txt"; dl.parent.mkdir(parents=True, exist_ok=True)
        dl.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats.append(dict(name=r.name, split=r.split, timeofday=r.timeofday, scene=r.scene, n_obj=len(lines),
                          n_person=n_cls[0], n_vehicle=n_cls[1], n_small=n_small))
    S = pd.DataFrame(stats)
    S.to_csv(out / "bdd_subset_statistik.csv", index=False)
    yml = (f"# BDD100K subset 2 kelas — tools/bdd_subset.py (CSV {csv.name}, sha256 {sha256(csv)[:12]})\n"
           f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
           + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
    (out / "bdd100k_2class.yaml").write_text(yml)
    counts = {s: int((S.split == s).sum()) for s in SPLITS}
    per = {s: dict(citra=counts[s], obj_per_citra=round(float(S[S.split == s].n_obj.mean()), 3),
                   person_per_citra=round(float(S[S.split == s].n_person.mean()), 3),
                   vehicle_per_citra=round(float(S[S.split == s].n_vehicle.mean()), 3),
                   kecil_per_citra=round(float(S[S.split == s].n_small.mean()), 3),
                   p_citra_berobjek_kecil=round(float((S[S.split == s].n_small > 0).mean()), 4),
                   p_citra_tanpa_objek=round(float((S[S.split == s].n_obj == 0).mean()), 4),
                   maks_obj=int(S[S.split == s].n_obj.max()),
                   timeofday={k: int(v) for k, v in S[S.split == s].timeofday.value_counts().items()})
           for s in SPLITS}
    rep = dict(sumber=str(src), keluaran=str(out.resolve()), csv=str(csv), csv_sha256=sha256(csv),
               ukuran_citra={f"{w}x{h}": n for (w, h), n in sizes.items()}, per_split=per)
    if a.config:
        import yaml
        cfg = yaml.safe_load(Path(a.config).read_text())
        assert Path(cfg["data"]) == (out / "bdd100k_2class.yaml").resolve(), f"config data = {cfg['data']}"
        for s, n in (cfg.get("expected_counts") or {}).items():
            assert counts[s] == n, f"expected_counts {s}: {n} != {counts[s]}"
        md = cfg["eval"]["coco_max_dets"][-1]
        mx = int(S.n_obj.max())
        assert mx <= md, f"maks objek per citra {mx} > coco_max_dets {md}: naikkan eval.coco_max_dets sebelum run"
        rep["coco_max_dets"] = md
    (out / "ringkasan_subset.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--root", required=True); g.add_argument("--out-csv", default=str(LOCK_CSV))
    g.add_argument("--seed", type=int, default=SEED)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True); b.add_argument("--out", required=True)
    b.add_argument("--csv", default=str(LOCK_CSV)); b.add_argument("--config", default=None)
    b.add_argument("--copy", action="store_true")
    a = ap.parse_args(argv)
    {"generate": cmd_generate, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
