"""
A11 — replikasi profil per segmen: bandingkan selisih.csv dari beberapa run profile_latency.py (sesi dan/atau model).

    python scripts/a11_bandingkan_profil.py --dirs profil_kitti_seed42=PROFIL_29OKT profil_kitti_seed42_sesi2=PROFIL_BARU \
        profil_kitti_seed7=... profil_visdrone_seed42=... --out A11

Setiap --dirs = LABEL=FOLDER (atau FOLDER saja; label = nama folder). Folder berisi selisih.csv, segmen.csv,
provenance.json, clock_*.txt dari `scripts/profile_latency.py` (mode lengkap atau --no-l2; bukan --l2-only).

Aturan keputusan (daftar tindakan A11, ditulis sebelum dijalankan): bila rincian overhead rute B (rute B − Vanilla) berbeda
lebih dari 0,02 ms per segmen antar-sesi, tulis rentangnya; bila tidak, tulis "direplikasi pada dua sesi".
Diterapkan dua kali: (1) antar-sesi model yang sama (KITTI s42; inti A11) dan (2) antar-semua run (sesi + model).
Selisih diagnostik lain (rute A − P2-Static, rantai, salin 8.400) dilaporkan dengan kriteria yang sama sebagai informasi.
"""
import argparse
import json
import re
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd

SEG = ("h2d", "exec", "d2h", "host", "wall")
UTAMA = "rute B − Vanilla"
BATAS = 0.02


def load(label, d):
    d = Path(d)
    S = pd.read_csv(d / "selisih.csv")
    prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").is_file() else {}
    clk = (d / "clock_sesudah.txt").read_text() if (d / "clock_sesudah.txt").is_file() else ""
    mhz = re.findall(r"Tesla T4, (\d+) MHz", clk)
    S.insert(0, "run", label)
    meta = dict(run=label, folder=str(d), waktu=prov.get("timestamp"), commit=(prov.get("router_final") or {}).get("commit"),
                gpu=(prov.get("device") or {}).get("name") if isinstance(prov.get("device"), dict) else prov.get("device"),
                sm_mhz_akhir=int(mhz[0]) if mhz else None,
                bobot=next((v[:12] for k, v in (prov.get("files") or {}).items() if k.endswith(".pt") and "last" in k), None))
    return S, meta


def verdict(df, runs):
    rows = []
    for sel, g in df[df.run.isin(runs)].groupby("selisih", sort=False):
        if g.run.nunique() < 2:
            continue
        for sgm in SEG:
            v = g.set_index("run")[sgm]
            rows.append(dict(selisih=sel, segmen=sgm, n_run=len(v), min=v.min(), max=v.max(), rentang=v.max() - v.min(),
                             keputusan=("direplikasi" if v.max() - v.min() <= BATAS else f"rentang [{v.min():+.3f}; {v.max():+.3f}] ms"),
                             **{f"nilai[{r}]": v.get(r, np.nan) for r in runs}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="LABEL=FOLDER")
    ap.add_argument("--sesi", nargs="*", default=None,
                    help="label run yang merupakan SESI berbeda dari model yang sama (default: label berawalan sama sebelum '_sesi')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tabs, metas = [], []
    for x in a.dirs:
        lab, d = x.split("=", 1) if "=" in x else (Path(x).name, x)
        S, m = load(lab, d); tabs.append(S); metas.append(m)
    df = pd.concat(tabs, ignore_index=True)
    runs = [m["run"] for m in metas]
    seen = {}
    for m in metas:                                              # dua label untuk run yang sama bukan replikasi
        key = (m["waktu"], m["commit"], m["bobot"])
        if key in seen:
            print(f"⚠ {m['run']} dan {seen[key]} adalah run YANG SAMA (timestamp, commit, bobot identik), bukan sesi berbeda")
        seen.setdefault(key, m["run"])
    sesi = a.sesi or [r for r in runs if r.split("_sesi")[0] == runs[0].split("_sesi")[0]]
    pd.DataFrame(metas).to_csv(out / "a11_run.csv", index=False)
    df.to_csv(out / "a11_selisih_semua.csv", index=False, float_format="%.5f")
    V1, V2 = verdict(df, sesi), verdict(df, runs)
    V1.to_csv(out / "a11_antar_sesi.csv", index=False, float_format="%.5f")
    V2.to_csv(out / "a11_antar_semua.csv", index=False, float_format="%.5f")

    lines = ["Run:", pd.DataFrame(metas)[["run", "waktu", "commit", "sm_mhz_akhir", "bobot"]].to_string(index=False), ""]
    for title, V, rs in (("(1) antar-sesi, model sama", V1, sesi), ("(2) antar-semua run", V2, runs)):
        lines.append(f"{title}: {rs}")
        if V.empty:
            lines.append("  < 2 run -> tidak dapat dibandingkan"); continue
        u = V[V.selisih == UTAMA]
        lines.append(u[["segmen", "min", "max", "rentang", "keputusan"]].round(4).to_string(index=False))
        seg4 = u[u.segmen != "wall"]
        ok = bool((seg4.rentang <= BATAS).all())
        lines.append(f"  Aturan A11 ({UTAMA}, per segmen h2d/exec/d2h/host, batas {BATAS} ms): "
                     + ("\"direplikasi pada dua sesi\"" if ok and title.startswith("(1)") else
                        ("direplikasi antar-run" if ok else "tulis RENTANG: " +
                         "; ".join(f"{r.segmen} [{r['min']:+.3f}; {r['max']:+.3f}]" for _, r in seg4.iterrows()))))
        other = V[(V.selisih != UTAMA) & (V.segmen == "wall")][["selisih", "min", "max", "rentang", "keputusan"]]
        lines += ["  Diagnostik lain (wall):", other.round(4).to_string(index=False), ""]
    txt = "\n".join(lines); print(txt)
    (out / "ringkasan.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
