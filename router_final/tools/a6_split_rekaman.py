"""
A6 — pembagian KITTI per REKAMAN (split Chen dkk./3DOP) untuk pelatihan ulang dan uji ulang H1–H6.

Latar (verifikasi B3/B9): split utama KITTI acak per citra -> 999/1.000 citra uji berasal dari rekaman yang juga ada di
data latih (median 2 frame dari citra latih terdekat). A6 mengulang seluruh protokol pada split yang terpisah per rekaman.

Split (dikunci 12 Nov 2026, SEBELUM pelatihan):
  uji   = Chen-val (3.769 citra, 45 rekaman)                   <- configs/a6/chen_val.txt
  latih = Chen-train dikurangi val'                            <- configs/a6/chen_train.txt
  val'  = ±20% Chen-train, dipotong PER REKAMAN dari Chen-train (proporsi val:latih split utama = 1.297:5.184 = 0,20)
Daftar Chen = kitti/image_sets/{train,val}.txt repo charlesq34/frustum-pointnets (commit 2ffdd345, sha256 di
configs/a6/SUMBER.md). Rekaman = pemetaan devkit objek KITTI resmi (train_mapping.txt + train_rand.txt, verifikasi B3):
irisan rekaman Chen-train vs Chen-val = 0 (diperiksa ulang di sini).

Pemilihan val' (hanya memakai label Chen-train; data uji tidak disentuh):
  n_draws permutasi acak rekaman (rng seed --seed), isi serakah sampai ukuran target; terima bila ukuran dalam
  [0,97; 1,03] × target dan >= --min-rec rekaman; skor = |p_kecil(val') − p_kecil(latih)| + |m_kecil(val') − m_kecil(latih)| /
  m_kecil(latih) + |m_obj(val') − m_obj(latih)| / m_obj(latih)  (p_kecil = proporsi citra berobjek kecil COCO < 32² px asli,
  m_* = rata-rata per citra). Skor minimum dipilih (seri -> undian pertama). Hasilnya DISIMPAN di
  configs/a6/kitti_split_rekaman.csv; `build` membaca berkas itu, bukan mengundi ulang.

    # sekali, offline (sudah dijalankan; `generate --check` memastikan berkas terkunci bisa direproduksi)
    python tools/a6_split_rekaman.py generate --label-stat kitti_label_stat.csv --check
    # di Kaggle, sebelum training: dataset per rekaman = symlink ke dataset 2 kelas asli (nama berkas & label tidak berubah)
    python tools/a6_split_rekaman.py build --root /kaggle/input/datasets/agungmaugi/kitti-2class \
        --out /kaggle/working/kitti_rekaman --config configs/kitti_rekaman.yaml
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
A6 = ROOT / "configs" / "a6"
SPLIT_CSV = A6 / "kitti_split_rekaman.csv"
SPLITS = ("train", "val", "test")
SMALL_AREA = 32.0 ** 2


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_list(p):
    ids = Path(p).read_text().split()
    assert len(ids) == len(set(ids)), f"{p}: ID ganda"
    return ids


def kish(sizes):
    s = np.asarray(sizes, float)
    return float(s.sum() ** 2 / (s ** 2).sum()) if len(s) else 0.0


def load_map(path=A6 / "kitti_peta_rekaman.csv"):
    M = pd.read_csv(path, dtype={"image_id": str, "frame_raw": str})
    assert len(M) == 7481 and M.image_id.is_unique, "peta rekaman harus 7.481 citra unik"
    return M


# ------------------------------------------------------------------ statistik label (untuk skor val')
def stats_from_labels_root(root, M):
    """n_obj, n_small per citra dari labels/<split_asal>/<id>.txt (format YOLO ternormalisasi) + ukuran asli dari peta."""
    rows = []
    for r in M.itertuples():
        f = Path(root) / "labels" / r.split / f"{r.image_id}.txt"
        L = [l.split() for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
        rows.append((r.image_id, len(L), sum(float(l[3]) * r.lebar * float(l[4]) * r.tinggi < SMALL_AREA for l in L)))
    return pd.DataFrame(rows, columns=["image_id", "n_obj", "n_small"])


def stats_from_csv(path, M):
    """kitti_label_stat.csv: image_id, n_obj, wh = 'w×h;…' ternormalisasi (dibuat dari kitti_yolo_2class.zip)."""
    S = pd.read_csv(path, dtype={"image_id": str}).merge(M[["image_id", "lebar", "tinggi"]], on="image_id")
    assert len(S) == 7481

    def ns(r):
        if not isinstance(r.wh, str):
            return 0
        return sum(float(a) * r.lebar * float(b) * r.tinggi < SMALL_AREA for a, b in (x.split("x") for x in r.wh.split(";")))
    S["n_small"] = S.apply(ns, axis=1)
    return S[["image_id", "n_obj", "n_small"]]


def carve_val(D, target, seed, n_draws, min_rec, tol=0.03):
    """D: citra Chen-train (rekaman, n_obj, n_small). Returns (set rekaman val', info)."""
    R = D.groupby("rekaman").agg(n=("image_id", "size"), obj=("n_obj", "sum"), sm=("n_small", "sum"),
                                 has=("n_small", lambda s: int((s > 0).sum())))
    recs = np.array(sorted(R.index))
    n, obj, sm, has = (R.loc[recs, c].to_numpy(float) for c in ("n", "obj", "sm", "has"))
    lo, hi = (1 - tol) * target, (1 + tol) * target
    tot = n.sum(), obj.sum(), sm.sum(), has.sum()
    rng = np.random.default_rng(seed)
    best, n_ok = None, 0
    for d in range(n_draws):
        order = rng.permutation(len(recs))
        pick, c = [], 0.0
        for j in order:
            if c + n[j] <= hi:
                pick.append(j); c += n[j]
            if c >= lo:
                break
        if not (lo <= c <= hi) or len(pick) < min_rec:
            continue
        n_ok += 1
        v = np.zeros(len(recs), bool); v[pick] = True
        nv, nt = n[v].sum(), tot[0] - n[v].sum()
        p_v, p_t = has[v].sum() / nv, (tot[3] - has[v].sum()) / nt
        ms_v, ms_t = sm[v].sum() / nv, (tot[2] - sm[v].sum()) / nt
        mo_v, mo_t = obj[v].sum() / nv, (tot[1] - obj[v].sum()) / nt
        score = abs(p_v - p_t) + abs(ms_v - ms_t) / ms_t + abs(mo_v - mo_t) / mo_t
        if best is None or score < best[0]:
            best = (score, d, v.copy(), dict(p_kecil_val=p_v, p_kecil_latih=p_t, m_kecil_val=ms_v, m_kecil_latih=ms_t,
                                             m_obj_val=mo_v, m_obj_latih=mo_t))
    assert best is not None, "tidak ada undian val' yang memenuhi syarat"
    score, d, v, st = best
    info = dict(seed=seed, n_draws=n_draws, n_diterima=n_ok, undian_terpilih=d, skor=score, target=target, tol=tol,
                min_rek=min_rec, **{k: float(x) for k, x in st.items()})
    return set(recs[v]), info


def make_split(M, S, chen_tr, chen_va, seed, n_draws, val_frac, min_rec):
    D = M.merge(S, on="image_id")
    tr, va = set(chen_tr), set(chen_va)
    assert not (tr & va) and tr | va == set(D.image_id), "Chen train ∪ val harus = 7.481 citra, tanpa irisan"
    rt, rv = set(D[D.image_id.isin(tr)].rekaman), set(D[D.image_id.isin(va)].rekaman)
    assert not (rt & rv), f"Chen train/val berbagi rekaman: {sorted(rt & rv)[:5]}"
    Dt = D[D.image_id.isin(tr)]
    target = int(round(val_frac * len(Dt)))
    vrec, info = carve_val(Dt, target, seed, n_draws, min_rec)
    D["split_a6"] = np.where(D.image_id.isin(va), "test", np.where(D.rekaman.isin(vrec), "val", "train"))
    out = D[["image_id", "rekaman", "tanggal", "split", "split_a6"]].rename(columns={"split": "split_asal"})
    return out.sort_values("image_id").reset_index(drop=True), info


def summarize(T, S=None):
    """Ringkasan per split: citra, rekaman, Kish, rekaman terbesar, (statistik label), irisan rekaman antar-split."""
    res = {}
    if S is not None:
        T = T.merge(S, on="image_id", how="left")
    for s in SPLITS:
        d = T[T.split_a6 == s]
        sz = d.rekaman.value_counts()
        res[s] = dict(citra=int(len(d)), rekaman=int(len(sz)), kish=round(kish(sz), 2), terbesar=str(sz.index[0]),
                      citra_terbesar=int(sz.iloc[0]), per_tanggal={k: int(v) for k, v in d.tanggal.value_counts().sort_index().items()},
                      dari_split_asal={k: int(v) for k, v in d.split_asal.value_counts().sort_index().items()})
        if S is not None:
            res[s].update(obj_per_citra=round(float(d.n_obj.mean()), 4), kecil_per_citra=round(float(d.n_small.mean()), 4),
                          p_citra_berobjek_kecil=round(float((d.n_small > 0).mean()), 4))
    rs = {s: set(T[T.split_a6 == s].rekaman) for s in SPLITS}
    res["irisan_rekaman"] = {f"{a}&{b}": len(rs[a] & rs[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return res


# ------------------------------------------------------------------ generate
def cmd_generate(a):
    M = load_map(a.map)
    S = stats_from_csv(a.label_stat, M) if a.label_stat else stats_from_labels_root(a.labels_root, M)
    T, info = make_split(M, S, read_list(a.chen_train), read_list(a.chen_val), a.seed, a.n_draws, a.val_frac, a.min_rec)
    summ = summarize(T, S)
    assert all(v == 0 for v in summ["irisan_rekaman"].values()), summ["irisan_rekaman"]
    meta = dict(pemilihan_val=info, ringkasan=summ,
                sha256={"chen_train.txt": sha256(a.chen_train), "chen_val.txt": sha256(a.chen_val),
                        "kitti_peta_rekaman.csv": sha256(a.map)})
    if a.check:
        ref = pd.read_csv(SPLIT_CSV, dtype=str)
        same = ref.equals(T.astype(str))
        print("REPRODUKSI", "IDENTIK" if same else "BERBEDA", "dengan", SPLIT_CSV)
        if not same:
            sys.exit(1)
    else:
        T.to_csv(a.out_csv, index=False)
        Path(a.out_csv).with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        print("ditulis", a.out_csv)
    print(json.dumps(meta, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ build (Kaggle)
def find_root(root):
    """Folder yang memuat images/{train,val,test} dan labels/{train,val,test} (dataset 2 kelas asli)."""
    root = Path(root)
    ok = lambda p: all((p / k / s).is_dir() for k in ("images", "labels") for s in SPLITS)
    if ok(root):
        return root
    cand = sorted({p.parent.parent for p in root.rglob("images/test") if ok(p.parent.parent)})
    assert len(cand) == 1, f"dataset KITTI 2 kelas tidak ditemukan unik di {root}: {cand}"
    return cand[0]


def cmd_build(a):
    T = pd.read_csv(SPLIT_CSV, dtype=str)
    meta = json.loads(SPLIT_CSV.with_suffix(".json").read_text())
    for f, h in meta["sha256"].items():
        assert sha256(A6 / f) == h, f"sha256 {f} berubah"
    assert set(T[T.split_a6 == "test"].image_id) == set(read_list(A6 / "chen_val.txt"))
    assert set(T[T.split_a6 != "test"].image_id) == set(read_list(A6 / "chen_train.txt"))
    summ = summarize(T)
    assert all(v == 0 for v in summ["irisan_rekaman"].values()), summ["irisan_rekaman"]
    src = find_root(a.root)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    n_lab = 0
    for r in T.itertuples():
        for kind, ext in (("images", ".png"), ("labels", ".txt")):
            s = src / kind / r.split_asal / f"{r.image_id}{ext}"
            d = out / kind / r.split_a6 / s.name
            d.parent.mkdir(parents=True, exist_ok=True)
            if not s.exists():
                assert kind == "labels", f"citra tidak ada: {s}"
                continue                                   # citra tanpa objek: Ultralytics menerima label kosong/absen
            n_lab += kind == "labels"
            if a.copy:
                shutil.copy2(s, d)
            else:
                os.symlink(s.resolve(), d)
    names = None
    for y in sorted(src.glob("*.yaml")):
        import yaml
        names = (yaml.safe_load(y.read_text()) or {}).get("names") or names
    names = names or {0: "Person", 1: "Vehicle"}
    ytxt = (f"# A6: KITTI 2 kelas, split per rekaman (Chen/3DOP + val' per rekaman) — tools/a6_split_rekaman.py\n"
            f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
            + "".join(f"  {int(k)}: {v}\n" for k, v in dict(names).items()))
    (out / "kitti_rekaman_2class.yaml").write_text(ytxt)
    T.to_csv(out / "kitti_split_rekaman.csv", index=False)
    counts = {s: len(list((out / "images" / s).iterdir())) for s in SPLITS}
    assert counts == {s: summ[s]["citra"] for s in SPLITS}, counts
    if a.config:
        import yaml
        cfg = yaml.safe_load(Path(a.config).read_text())
        assert Path(cfg["data"]) == (out / "kitti_rekaman_2class.yaml").resolve(), \
            f"config data = {cfg['data']} != {(out / 'kitti_rekaman_2class.yaml').resolve()}"
        for s, n in (cfg.get("expected_counts") or {}).items():
            assert counts[s] == n, f"expected_counts {s}: {n} != {counts[s]}"
    rep = dict(sumber=str(src), keluaran=str(out.resolve()), mode="copy" if a.copy else "symlink", citra=counts,
               label=n_lab, ringkasan=summ, sha256=meta["sha256"], split_csv_sha256=sha256(SPLIT_CSV))
    (out / "ringkasan_split.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--map", default=str(A6 / "kitti_peta_rekaman.csv"))
    g.add_argument("--chen-train", default=str(A6 / "chen_train.txt"))
    g.add_argument("--chen-val", default=str(A6 / "chen_val.txt"))
    x = g.add_mutually_exclusive_group(required=True)
    x.add_argument("--label-stat"); x.add_argument("--labels-root")
    g.add_argument("--seed", type=int, default=20261112)
    g.add_argument("--n-draws", type=int, default=20000)
    g.add_argument("--val-frac", type=float, default=0.20)
    g.add_argument("--min-rec", type=int, default=20)
    g.add_argument("--out-csv", default=str(SPLIT_CSV))
    g.add_argument("--check", action="store_true", help="bandingkan dengan berkas terkunci, jangan menulis")
    b = sub.add_parser("build")
    b.add_argument("--root", required=True); b.add_argument("--out", required=True)
    b.add_argument("--config", default=None, help="cek data path + expected_counts config")
    b.add_argument("--copy", action="store_true", help="salin berkas, bukan symlink")
    a = ap.parse_args(argv)
    {"generate": cmd_generate, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
