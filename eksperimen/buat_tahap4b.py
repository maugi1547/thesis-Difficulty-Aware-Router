"""
buat_tahap4b.py — membangkitkan tahap4b.py dari tahap4_final.py dengan patch teks yang kecil dan dapat diaudit.

Tahap 4B = analisis SENSITIVITAS pasca-hasil atas evaluasi final terkunci (Tahap 4). Protokol Tahap 4 tidak diubah;
hasil 4B dilaporkan terpisah. Dua tambahan:

  K-6  Latensi UJUNG-KE-UJUNG sistem yang di-deploy. Di Tahap 4, T_router dirakit sebagai
       aktivasi × T(rute A paksa) + (1 − aktivasi) × T(rute B paksa) + overhead host B2 (diukur terpisah).
       Tahap 4B mengukur langsung: per gambar, host menghitung threshold B2 (kuantil 200 skor sebelumnya), lalu
       engine A3 berjalan dengan threshold itu. Seluruh test diproses dalam urutan aliran yang sama dengan evaluasi
       akurasi, sehingga keputusan identik dengan Tahap 4. Vanilla, P2-Static, rute A paksa, dan rute B paksa diukur
       di pass yang sama (diselang-seling per gambar), sehingga perbandingannya berada dalam satu sesi.
       H6 dihitung ulang dengan fraksi campuran dari latensi ujung-ke-ujung.

  K-7  Baseline last.pt. Checkpoint Vanilla/P2-Static dipilih lewat konfigurasi notebook (best.pt atau last.pt);
       kode ini tidak bergantung pada pilihan itu.

Pemakaian:  python buat_tahap4b.py tahap4_final.py tahap4b.py
"""
import sys
from pathlib import Path

PATCHES = []


def patch(name, old, new):
    PATCHES.append((name, old, new))


# ---------------------------------------------------------------------------------------------------------------------
# 1. Fungsi pengukuran ujung-ke-ujung (tingkat modul agar dapat diuji dengan op tiruan)
patch("e2e_pass (fungsi modul)",
      "def job_main(job_path):\n",
      '''E2E_VARIANTS = ("Vanilla", "P2-Static", "A3 gate rute A", "A3 gate rute B",
                "Router B2 terkunci (e2e)", "Router B2 efisien (e2e)")
H6_E2E = "[4B, pasca-hasil] H6 | router (B2) − campuran acak V/P2S (latensi ujung-ke-ujung)"


def e2e_pass(ops, H, batches, rho, window, min_hist, thr0, n_warmup, n_reps, ctx, dsync):
    """[TAHAP 4B] Latensi ujung-ke-ujung (ms/gambar) pada aliran test berurutan.

    batches: list (img_uint8, small_area) dalam urutan aliran (= urutan acc_pass).
    Timer mencakup: hitung threshold B2 di host -> salin gambar & threshold ke GPU -> engine A3 -> salin y, use_a,
    skor ke host. Persiapan gambar (uint8 -> float, ke buffer pinned) di luar timer, sama dengan lat_pass Tahap 4.
    Riwayat B2 dimulai ulang di setiap pengulangan, sehingga keputusan setiap pengulangan identik dengan acc_pass.
    Kembali: (lat: varian -> list ms, use: varian -> list keputusan pengulangan pertama).
    """
    def b2_router(efisien):
        hist = SortedWindowQuantile(window) if efisien else collections.deque(maxlen=window)

        def f():
            n_h = len(hist.h) if efisien else len(hist)
            if n_h >= min_hist:
                thr = np.float32(hist.quantile(1 - rho) if efisien else np.quantile(np.asarray(hist, np.float64), 1 - rho))
            else:
                thr = np.float32(thr0)
            H["thr"]["a3g"].fill_(float(thr))
            ops[("a3g", -1.0)]()
            s_ = float(H["a3g"][2][0])
            if efisien:
                hist.push(s_)
            else:
                hist.append(s_)
            return bool(float(H["a3g"][1][0]) > 0.5)
        return f

    def fixed(op):
        def f():
            H["thr"]["a3g"].fill_(float(thr0))
            op()
            return None
        return f

    def fns_fresh():
        return {"Vanilla": ops["vanilla"], "P2-Static": ops["p2static"],
                "A3 gate rute A": fixed(ops[("a3g", 1.0)]), "A3 gate rute B": fixed(ops[("a3g", 0.0)]),
                "Router B2 terkunci (e2e)": b2_router(False), "Router B2 efisien (e2e)": b2_router(True)}

    def timed(fn, im, sa):
        H["img"].copy_(im.float() / 255.0); H["sa"].fill_(sa); dsync()       # konversi di luar timer
        w0 = time.perf_counter()
        with ctx():
            u = fn()
        return (time.perf_counter() - w0) * 1e3, u

    warm = fns_fresh()
    for im, sa in batches[:n_warmup]:
        for fn in warm.values():
            timed(fn, im, sa)
    lat = {k: [] for k in E2E_VARIANTS}
    use = {k: [] for k in E2E_VARIANTS if k.startswith("Router")}
    for rep in range(n_reps):
        fns = fns_fresh()
        order = list(fns)
        for i, (im, sa) in enumerate(batches):
            for k in (order if (rep + i) % 2 == 0 else order[::-1]):
                ms, u = timed(fns[k], im, sa)
                lat[k].append(ms)
                if rep == 0 and k in use:
                    use[k].append(u)
    return lat, use


def job_main(job_path):
''')

# ---------------------------------------------------------------------------------------------------------------------
# 2. Panggil e2e_pass di setiap build, setelah lat_pass Tahap 4 (lat_pass dan acc_pass tidak diubah)
patch("panggil e2e_pass",
      "        lat_pass(ops, H, imgs, bi)\n"
      "        Path(job[\"out_json\"]).write_text(json.dumps(res, default=float, indent=1))\n"
      "        log(f\"latensi build {bi} selesai\")\n",
      "        lat_pass(ops, H, imgs, bi)\n"
      "        Path(job[\"out_json\"]).write_text(json.dumps(res, default=float, indent=1))\n"
      "        log(f\"latensi build {bi} selesai\")\n"
      "        stream = [(b[\"img\"].clone(), small_area_of(b[\"ratio_pad\"][0])) for b in loader]   # [4B] seluruh test, urutan aliran\n"
      "        L_e, U_e = e2e_pass(ops, H, stream, rho, W, MINH, thr_g0, job[\"n_warmup\"], job[\"n_reps\"], ctx, dsync)\n"
      "        del stream\n"
      "        for k, v in L_e.items():\n"
      "            a = np.asarray(v)\n"
      "            res.setdefault(\"latency_e2e_builds\", {})[f\"b{bi}|{k}\"] = dict(mean=a.mean(), std=a.std(), p50=np.percentile(a, 50),\n"
      "                                                                         p95=np.percentile(a, 95), n=int(a.size))\n"
      "        res.setdefault(\"e2e_activation\", {}).update({f\"b{bi}|{k}\": float(np.mean(v)) for k, v in U_e.items()})\n"
      "        if bi == 0:\n"
      "            ref = list(map(bool, rec[\"use_g\"]))\n"
      "            res[\"e2e_parity\"] = {k: dict(n=len(v), beda_vs_akurasi=int(sum(a_ != b_ for a_, b_ in zip(v, ref))))\n"
      "                                 for k, v in U_e.items()}\n"
      "        Path(job[\"out_json\"]).write_text(json.dumps(res, default=float, indent=1))\n"
      "        log(f\"[4B] latensi ujung-ke-ujung build {bi} selesai | paritas {res.get('e2e_parity')}\")\n")

# rec dibuat di build 0 (acc_pass) dan dipakai lagi di build berikutnya -> harus terlihat di scope loop: sudah, karena
# variabel loop job_main. Tidak perlu patch.

# ---------------------------------------------------------------------------------------------------------------------
# 3. Analisis: latensi e2e, f campuran e2e, H6 e2e, kriteria tambahan
patch("analisis: latensi e2e",
      "    # ---- campuran untuk bootstrap\n",
      '''    # ---- [4B] latensi ujung-ke-ujung (pass terpisah, sesi yang sama)
    LE = None
    if res.get("latency_e2e_builds"):
        LE = pd.DataFrame([dict(build=k.split("|", 1)[0], varian=k.split("|", 1)[1], **v)
                           for k, v in res["latency_e2e_builds"].items()])
        te_ = LE.groupby("varian")["mean"].mean().to_dict()
        TrE, TVE, TPE = te_["Router B2 terkunci (e2e)"], te_["Vanilla"], te_["P2-Static"]
        TAE, TBE = te_["A3 gate rute A"], te_["A3 gate rute B"]
        act_e = float(np.mean([v for k, v in res["e2e_activation"].items() if k.endswith("terkunci (e2e)")]))
        lat.update(T_router_e2e=TrE, T_router_e2e_efisien=te_["Router B2 efisien (e2e)"], T_V_e2e=TVE, T_P2S_e2e=TPE,
                   T_A_e2e=TAE, T_B_e2e=TBE, aktivasi_e2e=act_e,
                   T_router_komposit_sepass=act_e * TAE + (1 - act_e) * TBE + b2ms,
                   f_mix_e2e=float(np.clip((TrE - TVE) / (TPE - TVE), 0, 1)),
                   e2e_parity=res.get("e2e_parity"))
        lat["selisih_e2e_minus_komposit_sepass"] = TrE - lat["T_router_komposit_sepass"]

    # ---- campuran untuk bootstrap
''')

patch("analisis: campuran e2e",
      "    for j, m_ in enumerate(P.random_masks(n, f_mix_e, n_draws, seed + 19)):\n"
      "        mixes[f\"campuran_b2ef_{j}\"] = vp(m_)\n",
      "    for j, m_ in enumerate(P.random_masks(n, f_mix_e, n_draws, seed + 19)):\n"
      "        mixes[f\"campuran_b2ef_{j}\"] = vp(m_)\n"
      "    if \"f_mix_e2e\" in lat:                                   # [4B]\n"
      "        for j, m_ in enumerate(P.random_masks(n, lat[\"f_mix_e2e\"], n_draws, seed + 19)):\n"
      "            mixes[f\"campuran_e2e_{j}\"] = vp(m_)\n")

patch("analisis: uji H6 e2e",
      "    tb = time.perf_counter()\n"
      "    log(f\"bootstrap: {len(mixes)} routing",
      "    if \"f_mix_e2e\" in lat:                                   # [4B]\n"
      "        tests.append((H6_E2E, \"router\", grp(\"campuran_e2e_\")))\n"
      "    tb = time.perf_counter()\n"
      "    log(f\"bootstrap: {len(mixes)} routing")

patch("analisis: kriteria 4B",
      "    RES = dict(model=spec[\"name\"], aktivasi_test=act,",
      "    if \"f_mix_e2e\" in lat:                                   # [4B]\n"
      "        KR.update(H1_latensi_e2e=bool(lat[\"T_V_e2e\"] < lat[\"T_router_e2e\"] < lat[\"T_P2S_e2e\"]),\n"
      "                  H5_rute_A_e2e=bool(lat[\"T_A_e2e\"] - lat[\"T_P2S_e2e\"] <= cfg[\"h5_ms\"]),\n"
      "                  H5_rute_B_e2e=bool(lat[\"T_B_e2e\"] - lat[\"T_V_e2e\"] <= cfg[\"h5_ms\"]),\n"
      "                  H6_e2e_status=stat(get(H6_E2E)) if get(H6_E2E) is not None else None)\n"
      "    RES = dict(model=spec[\"name\"], aktivasi_test=act,")

patch("simpan latensi e2e",
      "    PB.to_csv(out / \"h6_per_build.csv\", index=False)\n",
      "    PB.to_csv(out / \"h6_per_build.csv\", index=False)\n"
      "    if LE is not None:                                        # [4B]\n"
      "        LE.to_csv(out / \"latensi_e2e_per_build.csv\", index=False)\n")

patch("ringkasan gabungan: uji e2e",
      "              \"router (B2) − Branch A (P2 selalu)\", \"Branch A − P2-Static\", \"Branch B − Vanilla\")\n",
      "              \"router (B2) − Branch A (P2 selalu)\", \"Branch A − P2-Static\", \"Branch B − Vanilla\", H6_E2E)\n")

patch("ringkasan gabungan: kolom latensi e2e",
      "                   T_P2S=r[\"latensi\"][\"T_P2S\"], f_campuran=r[\"latensi\"][\"f_mix\"], **{k: v for k, v in r[\"kriteria\"].items()})\n",
      "                   T_P2S=r[\"latensi\"][\"T_P2S\"], f_campuran=r[\"latensi\"][\"f_mix\"],\n"
      "                   **{k: r[\"latensi\"].get(k) for k in (\"T_router_e2e\", \"T_V_e2e\", \"T_P2S_e2e\", \"aktivasi_e2e\",\n"
      "                                                       \"T_router_komposit_sepass\", \"selisih_e2e_minus_komposit_sepass\",\n"
      "                                                       \"f_mix_e2e\")},\n"
      "                   **{k: v for k, v in r[\"kriteria\"].items()})\n")

patch("job subprocess memanggil modul ini",
      "[sys.executable, \"-u\", \"-m\", \"tahap4_final\", str(out / \"job.json\")]",
      "[sys.executable, \"-u\", \"-m\", \"tahap4b\", str(out / \"job.json\")]")

patch("nama arsip",
      "    arc = Path(dest_dir) / f\"tahap4_{name}.zip\"\n",
      "    arc = Path(dest_dir) / f\"tahap4b_{name}.zip\"\n")
patch("nama arsip intermediate",
      "        arc2 = Path(dest_dir) / f\"tahap4_{name}_intermediate.zip\"\n",
      "        arc2 = Path(dest_dir) / f\"tahap4b_{name}_intermediate.zip\"\n")

patch("docstring",
      '"""\ntahap4_final.py — Evaluasi final sistem router (protokol dikunci 16 Okt 2026, sebelum dijalankan).\n',
      '"""\ntahap4b.py — Tahap 4B: SENSITIVITAS pasca-hasil atas Tahap 4 (dibangkitkan oleh buat_tahap4b.py dari tahap4_final.py).\n'
      'Tambahan: latensi ujung-ke-ujung B2 + A3 (K-6) dan pilihan checkpoint baseline (K-7, di notebook). Sisanya identik.\n\n'
      'Isi asli tahap4_final.py:\ntahap4_final.py — Evaluasi final sistem router (protokol dikunci 16 Okt 2026, sebelum dijalankan).\n')


def main(src, dst):
    s = Path(src).read_text(encoding="utf-8")
    for name, old, new in PATCHES:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"patch '{name}': pola ditemukan {n}x (harus 1)")
        s = s.replace(old, new)
        print("✅", name)
    Path(dst).write_text(s, encoding="utf-8")
    print("→", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
