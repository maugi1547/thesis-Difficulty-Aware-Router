"""
B8 — jalankan ulang sapuan ρ (scripts/sweep_rho.py) dari commit BERSIH, tanpa GPU, lalu bandingkan dengan sweep_rho_hasil.

Latar (verifikasi B4): sapuan ρ KITTI dijalankan dari 28822a2 + sweep_rho.py belum di-commit, VisDrone dari 1ce74dc +
sweep_rho.py/h6_latency.py dimodifikasi. Padanan bersihnya c0e833c; ef49480 (HEAD final-arsitektur) memuat sweep_rho.py
yang sama. Skrip ini:
  1. membuat git worktree terpisah pada --commit (checkout segar -> tidak ada perubahan CRLF / berkas tak terlacak),
     memastikan `git status --porcelain` kosong, dan mencatat blob sweep_rho.py (dibandingkan dengan c0e833c bila ada);
  2. mengambil masukan dari folder hasil (zip atau folder hasil ekstraksi, mis. dataset Kaggle):
       suite_<model>_intermediate  -> stageA.pkl, job_res.json, job_dets.pkl, gt_test.json
       suite_<model>               -> bootstrap_test.csv (paritas ρ = 0,5)
       ablasi_kitti_seed42.zip:cache_ablasi.pkl, arsip_visdrone_seed42.zip:ablasi/cache_ablasi.pkl (kurva val)
  3. menjalankan sweep_rho.py dengan argumen yang SAMA dengan run asli:
       kitti_seed42   --ablation-cache            (UnionBoot)
       kitti_seed7    tanpa cache val              (UnionBoot)
       kitti_seed2026 tanpa cache val              (UnionBoot)
       visdrone_seed42 --ablation-cache --fast     (FastUnionBoot; identik dengan UnionBoot, lihat tests/test_cocomix.py)
       pool kitti_seed42 kitti_seed7 kitti_seed2026
  4. membandingkan setiap keluaran dengan sweep_rho_hasil (byte; bila beda: numerik, selisih maks), memeriksa
     provenance.json tiap run (commit = --commit, dirty_files = []), lalu menulis b8_laporan.md dan b8_laporan.json.

Contoh (Linux / WSL / Kaggle CPU; butuh torch CPU, numpy, pandas, pycocotools, ultralytics==8.3.240 stok):

    python router_final/tools/b8_sweep_rho_bersih.py \
        --repo "/mnt/d/Tesis/Thesis/Kode Program/ultralytics_folder" --commit ef49480 \
        --hasil "/mnt/d/Tesis/Thesis/Kode Program/Hasil Eksperimen/Hasil router final" \
        --out ~/b8_run

    # uji cepat alur (bukan bukti B8): --models kitti_seed7 --n-boot 50 (perbandingan otomatis dilewati)

Waktu (2 inti CPU): ±9 menit per model KITTI, ±60 menit VisDrone. Windows asli tidak didukung: bootstrap paralel
cocomix memakai multiprocessing "fork" (jalankan di WSL atau Kaggle).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

KITTI = ("kitti_seed42", "kitti_seed7", "kitti_seed2026")
MODELS = KITTI + ("visdrone_seed42",)
CONFIG = {m: "configs/visdrone.yaml" if m.startswith("visdrone") else "configs/kitti.yaml" for m in MODELS}
EXTRA = {"kitti_seed42": ["ABL"], "kitti_seed7": [], "kitti_seed2026": [], "visdrone_seed42": ["ABL", "--fast"]}
ABL_SRC = {"kitti_seed42": ("ablasi_kitti_seed42.zip", "cache_ablasi.pkl"),
           "visdrone_seed42": ("arsip_visdrone_seed42.zip", "ablasi/cache_ablasi.pkl")}
SUITE_FILES = ("stageA.pkl", "job_res.json", "job_dets.pkl", "gt_test.json")
CLEAN_REF = "c0e833c"                     # commit bersih pertama yang memuat sweep_rho.py versi final


def sh(*cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.returncode:
        if check:
            raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
        return None
    return r.stdout.strip()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    print(f"[B8 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. worktree bersih
def make_worktree(repo, commit, dest):
    repo = Path(repo).resolve()
    full = sh("git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}")
    if dest.exists():
        head = sh("git", "-C", str(dest), "rev-parse", "HEAD", check=False)
        if head != full:
            raise SystemExit(f"{dest} sudah ada pada {head or '?'} (bukan {full}); hapus dulu atau pakai --out lain")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh("git", "-C", str(repo), "worktree", "add", "--detach", str(dest), full)
    st = sh("git", "-C", str(dest), "status", "--porcelain", "--untracked-files=all")
    if st:
        raise SystemExit(f"worktree {dest} tidak bersih:\n{st}")
    blob = sh("git", "-C", str(dest), "rev-parse", f"{full}:router_final/scripts/sweep_rho.py")
    ref_blob = sh("git", "-C", str(repo), "rev-parse", "--verify", "-q", f"{CLEAN_REF}:router_final/scripts/sweep_rho.py",
                  check=False)
    tree = sh("git", "-C", str(dest), "rev-parse", f"{full}^{{tree}}")          # sama antar-klon bila isi sama
    return dict(commit=full, tree=tree, worktree=str(dest), sweep_rho_blob=blob, sweep_rho_blob_c0e833c=ref_blob,
                sweep_rho_sama_dengan_c0e833c=None if ref_blob is None else blob == ref_blob)


# ---------------------------------------------------------------- 2. masukan
def find_dir_with(root, name, files, exclude=("baseline_best",)):
    """Folder bernama `name` yang memuat semua `files` (hasil ekstraksi / dataset Kaggle)."""
    hits = [p for p in Path(root).rglob(name) if p.is_dir() and all((p / f).is_file() for f in files)
            and not any(e in str(p) for e in exclude)]
    return hits


def extract_suite(hasil, model, kind, work):
    """kind 'i' = intermediate, 's' = hasil suite. Folder hasil ekstraksi diutamakan, selain itu zip."""
    files = SUITE_FILES if kind == "i" else ("bootstrap_test.csv",)
    zname = f"suite_{model}_intermediate.zip" if kind == "i" else f"suite_{model}.zip"
    zips = [p for p in Path(hasil).rglob(zname)]
    tgt = work / "masukan" / kind
    if (tgt / model).is_dir() and all((tgt / model / f).is_file() for f in files):
        return tgt / model, "ekstraksi sebelumnya"
    if len(zips) == 1:
        with zipfile.ZipFile(zips[0]) as z:
            members = [n for n in z.namelist() if n.startswith(f"{model}/")]
            z.extractall(tgt, members)
        assert all((tgt / model / f).is_file() for f in files), f"{zips[0]} tidak memuat {files}"
        return tgt / model, str(zips[0])
    dirs = [d for d in find_dir_with(hasil, model, files) if (kind == "i") == (d / "job_dets.pkl").is_file()]
    if len(dirs) == 1:
        return dirs[0], str(dirs[0])
    raise SystemExit(f"masukan suite {kind} {model}: zip {zips}, folder {dirs} (harus tepat satu)")


def extract_ablation(hasil, model, work):
    zname, member = ABL_SRC[model]
    tgt = work / "masukan" / "ablasi" / model / "cache_ablasi.pkl"
    if tgt.is_file():
        return tgt, "ekstraksi sebelumnya"
    zips = list(Path(hasil).rglob(zname))
    if len(zips) == 1:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as z, z.open(member) as src, open(tgt, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        return tgt, f"{zips[0]}:{member}"
    raise SystemExit(f"cache ablasi {model}: {zname} ditemukan {len(zips)} kali")


def reference_dir(hasil, work):
    tgt = work / "acuan"
    if (tgt / "kitti_pool" / "sweep_pool.csv").is_file():
        return tgt, "ekstraksi sebelumnya"
    zips = list(Path(hasil).rglob("sweep_rho_hasil.zip"))
    if len(zips) == 1:
        zipfile.ZipFile(zips[0]).extractall(tgt)
        return tgt, str(zips[0])
    dirs = [p.parent.parent for p in Path(hasil).rglob("kitti_pool/sweep_pool.csv")]
    if len(dirs) == 1:
        return dirs[0], str(dirs[0])
    raise SystemExit(f"sweep_rho_hasil: zip {zips}, folder {dirs}")


# ---------------------------------------------------------------- 3. jalankan
def run_sweep(wt, args, logf, pythonpath):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([pythonpath, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    cmd = [sys.executable, "scripts/sweep_rho.py", *args]
    t0 = time.time()
    with open(logf, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n"); f.flush()
        r = subprocess.run(cmd, cwd=Path(wt) / "router_final", env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"sweep_rho gagal (kode {r.returncode}), lihat {logf}")
    return time.time() - t0


# ---------------------------------------------------------------- 4. bandingkan
def compare_file(a, b):
    """a = ulang, b = acuan."""
    import numpy as np
    import pandas as pd
    if not b.exists():
        return dict(berkas=str(b.name), status="acuan tidak ada")
    if not a.exists():
        return dict(berkas=str(a.name), status="TIDAK DIHASILKAN")
    if a.read_bytes() == b.read_bytes():
        return dict(berkas=a.name, status="identik byte")
    if a.suffix == ".npz":
        A, B = np.load(a), np.load(b)
        if set(A.files) != set(B.files):
            return dict(berkas=a.name, status="BEDA kunci", detail=sorted(set(A.files) ^ set(B.files))[:10])
        neq = [k for k in A.files if not np.array_equal(A[k], B[k], equal_nan=True)]
        if not neq:
            return dict(berkas=a.name, status="identik isi (array sama; kompresi zip berbeda)")
        d = max(float(np.nanmax(np.abs(A[k].astype(float) - B[k].astype(float)))) for k in neq)
        return dict(berkas=a.name, status="BEDA", n_array_beda=len(neq), selisih_maks=d)
    if a.suffix == ".csv":
        A, B = pd.read_csv(a), pd.read_csv(b)
        if list(A.columns) != list(B.columns) or A.shape != B.shape:
            return dict(berkas=a.name, status="BEDA bentuk", ulang=list(A.shape), acuan=list(B.shape))
        num = A.select_dtypes("number").columns
        obj = [c for c in A.columns if c not in num]
        if not A[obj].astype(str).equals(B[obj].astype(str)):
            return dict(berkas=a.name, status="BEDA kolom teks")
        d = float(np.nanmax(np.abs(A[num].to_numpy(float) - B[num].to_numpy(float)))) if len(num) else 0.0
        nan_ok = bool((A[num].isna().to_numpy() == B[num].isna().to_numpy()).all())
        return dict(berkas=a.name, status="identik numerik (format teks beda)" if d == 0 and nan_ok else "BEDA",
                    selisih_maks=d)
    la, lb = a.read_text().splitlines(), b.read_text().splitlines()
    beda = [i + 1 for i, (x, y) in enumerate(zip(la, lb)) if x != y] + ([min(len(la), len(lb)) + 1] if len(la) != len(lb) else [])
    return dict(berkas=a.name, status="BEDA teks", baris=beda[:20])


def check_prov(p, commit):
    pr = json.loads(Path(p).read_text())
    g = pr.get("router_final", {})
    return dict(commit=g.get("commit"), commit_cocok=g.get("commit") == commit, dirty_files=g.get("dirty_files"),
                bersih=g.get("dirty_files") == [], versi=pr.get("versions"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="klon git thesis (berisi router_final/)")
    ap.add_argument("--commit", default="ef49480", help="commit bersih (ef49480 atau c0e833c)")
    ap.add_argument("--hasil", required=True, help="folder 'Hasil router final' (zip) atau /kaggle/input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--pythonpath", default=None, help="tambahan PYTHONPATH (mis. folder ultralytics stok)")
    ap.add_argument("--n-boot", type=int, default=None, help="HANYA uji alur; perbandingan dengan acuan dilewati")
    ap.add_argument("--skip-run", action="store_true", help="hanya bandingkan keluaran yang sudah ada")
    a = ap.parse_args()
    if not hasattr(os, "fork"):
        raise SystemExit("Platform tanpa fork (Windows asli): bootstrap paralel cocomix butuh 'fork'. Jalankan di WSL/Kaggle.")
    out = Path(a.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    rep = dict(mulai=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, python=sys.version.split()[0], platform=sys.platform)

    log(f"worktree {a.commit} ...")
    rep["git"] = make_worktree(a.repo, a.commit, out / "worktree")
    wt = Path(rep["git"]["worktree"])
    log(f"commit {rep['git']['commit']} bersih; blob sweep_rho.py {rep['git']['sweep_rho_blob'][:12]} "
        f"(sama dengan {CLEAN_REF}: {rep['git']['sweep_rho_sama_dengan_c0e833c']})")

    log("masukan ...")
    ins, ref_src = {}, None
    for m in a.models:
        i_dir, i_src = extract_suite(a.hasil, m, "i", out)
        s_dir, s_src = extract_suite(a.hasil, m, "s", out)
        d = dict(suite_dir=str(i_dir), ref=str(s_dir / "bootstrap_test.csv"), sumber=[i_src, s_src],
                 sha256={f: sha256(i_dir / f) for f in SUITE_FILES})
        if "ABL" in EXTRA[m]:
            p, src = extract_ablation(a.hasil, m, out)
            d.update(ablation_cache=str(p), sumber_ablasi=src, sha256_ablasi=sha256(p))
        ins[m] = d
    rep["masukan"] = ins
    if a.n_boot is None:
        ref_dir, ref_src = reference_dir(a.hasil, out)
        rep["acuan"] = ref_src

    res_dir = out / "sweep"
    rep["run"] = {}
    for m in a.models:
        args = ["run", "--config", CONFIG[m], "--suite-dir", ins[m]["suite_dir"], "--out", str(res_dir / m),
                "--ref", ins[m]["ref"]]
        for x in EXTRA[m]:
            args += ["--ablation-cache", ins[m]["ablation_cache"]] if x == "ABL" else [x]
        if a.n_boot:
            args += ["--n-boot", str(a.n_boot)]
        if a.skip_run:
            continue
        log(f"sweep_rho {m} ...")
        dt = run_sweep(wt, args, out / f"log_{m}.txt", a.pythonpath)
        rep["run"][m] = dict(args=args, detik=round(dt))
        log(f"  selesai {dt / 60:.1f} menit")
    pool_models = [m for m in KITTI if m in a.models]
    if len(pool_models) == 3 and not a.skip_run:
        args = ["pool", "--dirs", *[str(res_dir / m) for m in KITTI], "--out", str(res_dir / "kitti_pool")]
        rep["run"]["kitti_pool"] = dict(args=args, detik=round(run_sweep(wt, args, out / "log_kitti_pool.txt", a.pythonpath)))

    # provenance & paritas dari ringkasan
    rep["provenance"] = {m: check_prov(res_dir / m / "provenance.json", rep["git"]["commit"]) for m in a.models
                         if (res_dir / m / "provenance.json").is_file()}
    for m in a.models:
        txt = (res_dir / m / "ringkasan.txt").read_text() if (res_dir / m / "ringkasan.txt").is_file() else ""
        par = [l for l in txt.splitlines() if l.startswith("Paritas")]
        v = rep["provenance"].setdefault(m, {})
        v["baris_paritas"] = par
        try:
            v["paritas_maks"] = max(json.loads(par[0].split(":", 1)[1].strip()).values())
        except Exception:                                           # noqa: BLE001
            v["paritas_maks"] = None

    # perbandingan
    rep["perbandingan"] = {}
    if a.n_boot is None:
        for m in [*a.models, *(["kitti_pool"] if len(pool_models) == 3 else [])]:
            names = ["sweep_pool.csv", "ringkasan.txt"] if m == "kitti_pool" else \
                ["sweep_test.csv", "sweep_latensi.csv", "sweep_val.csv", "sweep_boot.npz", "ringkasan.txt"]
            rep["perbandingan"][m] = [compare_file(res_dir / m / n, ref_dir / m / n) for n in names]

    ok_num = all(r["status"].startswith("identik") or r["status"] == "acuan tidak ada"
                 for rs in rep["perbandingan"].values() for r in rs if r["berkas"] != "ringkasan.txt")
    ok_prov = all(v.get("commit_cocok") and v.get("bersih") for v in rep["provenance"].values() if "commit" in v)
    ok_par = all(v.get("paritas_maks") is not None and v["paritas_maks"] < 1e-12 for v in rep["provenance"].values())
    rep["kesimpulan"] = dict(angka_identik=ok_num if rep["perbandingan"] else None, provenans_bersih=ok_prov,
                             paritas_suite=ok_par, B8_lolos=bool(rep["perbandingan"]) and ok_num and ok_prov and ok_par)
    rep["selesai"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "b8_laporan.json").write_text(json.dumps(rep, indent=1, default=str))

    L = [f"# B8 — sapuan ρ ulang dari commit bersih ({rep['selesai']})", "",
         f"- Commit: `{rep['git']['commit']}`, tree `{rep['git']['tree']}` (worktree segar, `git status` kosong); blob `sweep_rho.py` "
         f"`{rep['git']['sweep_rho_blob'][:12]}`, sama dengan {CLEAN_REF}: {rep['git']['sweep_rho_sama_dengan_c0e833c']}",
         f"- Python {rep['python']} ({rep['platform']}); versi pustaka per run di `b8_laporan.json`", "",
         "| Model | commit di provenance | dirty_files | paritas ρ = 0,5 vs suite |", "|---|---|---|---|"]
    for m, v in rep["provenance"].items():
        L.append(f"| {m} | {'cocok' if v.get('commit_cocok') else v.get('commit')} | {v.get('dirty_files')} | "
                 f"{' / '.join(x.split(':', 1)[-1].strip() for x in v.get('baris_paritas', [])) or '—'} |")
    if rep["perbandingan"]:
        L += ["", "| Model | Berkas | Status | Selisih maks |", "|---|---|---|---|"]
        for m, rs in rep["perbandingan"].items():
            for r in rs:
                L.append(f"| {m} | {r['berkas']} | {r['status']} | {r.get('selisih_maks', '')} |")
    L += ["", f"**Angka identik dengan sweep_rho_hasil:** {rep['kesimpulan']['angka_identik']}  ",
          f"**Provenans bersih (commit cocok, dirty_files = []):** {ok_prov}  ",
          f"**Paritas baris ρ = 0,5 vs H2/H6 utama suite (< 1e-12):** {ok_par}  ",
          f"**B8 lolos:** {rep['kesimpulan']['B8_lolos']}", "",
          "ringkasan.txt dibandingkan sebagai teks dan tidak menentukan kelulusan (format cetak dapat berbeda antar-versi pandas)."]
    (out / "b8_laporan.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if (rep["kesimpulan"]["B8_lolos"] or a.n_boot) else 1)


if __name__ == "__main__":
    main()
