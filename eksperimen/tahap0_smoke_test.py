"""
tahap0_smoke_test.py — TAHAP 0: Smoke test & sanity check
Router-P2 YOLOv8n Dual-Branch (KITTI 2-class), Kaggle T4

Jalankan per sel (# %%) BERURUTAN di kernel baru. Sel 1–2 hanya berisi fungsi
(tidak menjalankan apa pun). Sel 3 menjalankan training 3 epoch (~3x waktu 1 epoch).

Prasyarat (lihat tutorial, bagian "Persiapan"):
  P0. Commit lokal 5416543 ("Audit fix lanjutan ... kalibrasi persisten") SUDAH di-push ke
      GitHub dan repo di Kaggle di-clone ulang. Sel [0] memverifikasi ini.
  P1. script_training.py: bagian eksekusi (section 7) dibungkus `def main()` +
      `if __name__ == "__main__": main()` supaya kelas/fungsinya bisa di-import.
  P2. block.py (router, mode eval): simpan keputusan per-gambar ke `self._last_gate_decision`.
      Tanpa P2, uji 0.5 jatuh ke versi lemah (hanya membandingkan JUMLAH gambar aktif).

Keluaran di OUT_DIR:
  router_loss_components.csv   (11 kolom, per batch, hanya epoch pasca-warmup)
  loss_items.csv               (8 komponen loss per batch, semua epoch)
  epoch_timing.csv, budget.csv (waktu per epoch & estimasi anggaran)
  sanity_report.json           (status PASS/WARN/FAIL setiap pemeriksaan)
  router_trace.png             (GV_Offset, GV_Target, P2_Prob_Real, GV_Loss per batch)
"""

# %% [0] Verifikasi versi kode (WAJIB, sebelum apa pun)
import os, sys, inspect, subprocess
from pathlib import Path
REPO = Path(globals().get("T0_OVERRIDE", {}).get("REPO", "/kaggle/working/thesis-Difficulty-Aware-Router"))
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import ultralytics
import ultralytics.nn.modules.block as _blk
import ultralytics.model_yolov8_router as _mr
import ultralytics.utils.loss as _ls
CODE_COMMIT = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
_src_r = inspect.getsource(_blk.UltraLightWeightDifficultyAwareRouter)
CODE_FEATURES = {
    "ultralytics di-import dari repo (bukan pip)": Path(ultralytics.__file__).resolve().is_relative_to(REPO.resolve()),
    "AUDIT-FIX #9 kalibrasi = buffer int64": "_calib_T_micro" in _src_r,
    "GroupNorm di classifier gate": "GroupNorm(1, hidden_dim // 2)" in _src_r,
    "top-k spatial pool di gate": "_topk_spatial_pool" in _src_r,
    "patch P2 _last_gate_decision": "_last_gate_decision" in _src_r,
    "AUDIT-FIX #6 kolom GV tanpa lag": hasattr(_mr.DualBranchDetectionLoss, "_flush_pending_log"),
    "AUDIT-FIX #3 advantage di-clip": "gate_adv_clip" in inspect.getsource(_mr.DualBranchDetectionLoss),
    "AUDIT-FIX #7 normalisasi bg anchor": "normalize_bg_anchors" in inspect.getsource(_ls.v8DetectionLoss),
}
print(f"commit={CODE_COMMIT} | ultralytics {ultralytics.__version__} @ {ultralytics.__file__} | torch {__import__('torch').__version__}")
for k, v in CODE_FEATURES.items():
    print(("✅ " if v else "❌ ") + k)
assert all(CODE_FEATURES.values()), "Versi kode di sesi ini BUKAN versi yang dimaksud — push/clone ulang dulu."

# %% [1] Fungsi utilitas murni (tanpa Ultralytics) — scheduler, logger, timer, pemeriksa
import os, sys, csv, json, time, shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

CSV_COLS = ["Epoch", "Batch", "Lambda", "P2_Prob_Real", "Rel", "Diff_W", "Final_L3",
            "GV_Target", "GV_Loss", "GV_Weight", "GV_Offset"]
LOSS_NAMES = ["box_A", "cls_A", "dfl_A", "router", "proxy", "box_B", "cls_B", "dfl_B"]


def find_router(model):
    """Kembalikan modul router pertama (nama kelas mengandung 'DifficultyAwareRouter')."""
    for m in model.modules():
        if "DifficultyAwareRouter" in m.__class__.__name__:
            return m
    raise RuntimeError("Router tidak ditemukan di model.")


def _ramp(epoch, start, length, final):
    """0 sampai epoch == start, lalu naik linear ke `final` selama `length` epoch, lalu tetap."""
    if epoch <= start:
        return 0.0
    return final * min((epoch - start) / max(length, 1), 1.0)


def make_curriculum_scheduler(router_warmup_end, gv_start, gv_ramp, final_gv,
                              lam_start, lam_ramp, final_lambda, history=None):
    """
    Versi berparameter dari ideal_curriculum_scheduler (script_training.py).

    Konfigurasi run utama yang IDENTIK dengan kode lama:
        make_curriculum_scheduler(router_warmup_end=10, gv_start=10, gv_ramp=20, final_gv=0.5,
                                  lam_start=50, lam_ramp=30, final_lambda=<nilai Anda>)
    Perbedaan perilaku dgn versi lama: _is_warmup di-set dua arah (lama hanya mematikan).
    Untuk run baru hasilnya sama; untuk resume, versi ini lebih benar.

    history: dict opsional, diisi {epoch: {"lambda", "gv", "warmup"}} untuk diverifikasi.
    """
    def scheduler(trainer):
        epoch = trainer.epoch + 1
        lam = _ramp(epoch, lam_start, lam_ramp, final_lambda)
        gv = _ramp(epoch, gv_start, gv_ramp, final_gv)
        warm = epoch <= router_warmup_end
        targets = [trainer.model]
        if getattr(trainer, "ema", None):
            targets.append(trainer.ema.ema)
        for mdl in targets:
            mdl.router_penalty_lambda = lam
            mdl.gate_value_loss_weight = gv
            mdl.total_batches = len(trainer.train_loader)
            find_router(mdl)._is_warmup = warm
        if history is not None:
            history[epoch] = {"lambda": lam, "gv": gv, "warmup": warm}
        print(f"📈 [SCHEDULER] epoch {epoch} | warmup={warm} | lambda={lam:.4f} | gate_value_w={gv:.4f}")
    return scheduler


def make_router_cfg_applier(router_cfg, dump_path=None):
    """
    Callback on_pretrain_routine_end: tulis hyperparameter router secara EKSPLISIT ke model
    (bukan mengandalkan default getattr yang tersebar di loss.py & model_yolov8_router.py).
    Harus jalan sebelum batch pertama, karena DualBranchDetectionLoss membaca
    branch_b_loss_weight & gate_offset_momentum saat criterion dibuat (lazy, batch pertama).
    """
    def apply(trainer):
        for mdl in [trainer.model] + ([trainer.ema.ema] if getattr(trainer, "ema", None) else []):
            for k, v in router_cfg.items():
                setattr(mdl, k, v)
        if dump_path is not None:
            Path(dump_path).write_text(json.dumps(router_cfg, indent=1))
        print("🔧 [ROUTER CFG]", router_cfg)
    return apply


def make_router_csv_logger(csv_path):
    """
    Pengganti on_train_epoch_end di script_training.py:
      - path per-run (bukan /kaggle/working/... global yang di-append lintas run),
      - menolak file yang sudah ada (cegah data run lama tercampur),
      - memverifikasi tiap baris berisi 11 kolom,
      - kolom Epoch diganti epoch ASLI (label di loss.py = debug_counter//total_batches + 4,
        salah setiap kali warmup tidak berakhir di epoch 4). Tetap 11 kolom.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        raise FileExistsError(f"{csv_path} sudah ada — hapus/ganti nama dulu.")

    def on_train_epoch_end(trainer):
        buf = getattr(trainer.model, "router_buffer", None)
        if not buf:
            return
        epoch = trainer.epoch + 1
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_COLS)
            for row in buf:
                if len(row) != len(CSV_COLS):
                    raise ValueError(f"router_buffer berisi {len(row)} kolom, header {len(CSV_COLS)}: {row}")
                w.writerow([epoch] + list(row[1:]))
        buf.clear()
    return on_train_epoch_end


def make_loss_items_logger(store):
    """on_train_batch_end: simpan 8 komponen loss per batch + gagal cepat bila NaN/Inf."""
    def on_train_batch_end(trainer):
        li = trainer.loss_items.detach().float().cpu()
        epoch = trainer.epoch + 1
        store.append([epoch] + li.tolist())
        if not torch.isfinite(li).all() or not torch.isfinite(trainer.loss.detach()).all():
            raise FloatingPointError(f"Loss non-finite di epoch {epoch}, batch #{len(store)}: "
                                     f"{dict(zip(LOSS_NAMES, li.tolist()))}")
    return on_train_batch_end


class EpochTimer:
    """
    Ukur waktu per epoch, dipisah: train (loop batch), val_total (validasi + save + callback),
    dan valB (validasi Branch B, diukur lewat wrapper `timed`).
    cuda.synchronize() supaya kernel async tidak 'bocor' ke fase berikutnya.
    """
    def __init__(self):
        self.rows, self._t, self._valB = [], {}, 0.0

    @staticmethod
    def now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_epoch_start(self, trainer):
        self._t["t0"] = self.now()
        self._valB = 0.0

    def on_train_epoch_end(self, trainer):
        self._t["t1"] = self.now()

    def on_fit_epoch_end(self, trainer):
        # final_eval() Ultralytics memanggil on_fit_epoch_end SEKALI LAGI setelah loop
        # (validasi best.pt) -> abaikan panggilan yang tidak didahului on_train_epoch_end.
        if self._t.get("t1") is None:
            return
        t2 = self.now()
        self.rows.append(dict(epoch=trainer.epoch + 1,
                              n_batches=len(trainer.train_loader),
                              train_s=self._t["t1"] - self._t["t0"],
                              val_total_s=t2 - self._t["t1"],
                              valB_s=self._valB))
        self._t["t1"] = None

    def timed(self, cb):
        def wrapper(validator):
            t0 = self.now()
            cb(validator)
            self._valB += self.now() - t0
        return wrapper


# ---------------------------- pemeriksa ----------------------------
class Report:
    def __init__(self):
        self.items = []

    def add(self, name, status, detail=""):
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
        print(f"{icon} [{status}] {name}" + (f" — {detail}" if detail else ""))
        self.items.append({"check": name, "status": status, "detail": str(detail)})

    def check(self, name, ok, detail="", warn_only=False):
        self.add(name, "PASS" if ok else ("WARN" if warn_only else "FAIL"), detail)

    @property
    def n_fail(self):
        return sum(i["status"] == "FAIL" for i in self.items)


def check_router_csv(df, n_batches, epochs, router_warmup_end, sched_history,
                     offset_clip=1.0, report=None):
    """
    Semua pemeriksaan pada router_loss_components.csv. `df` dibaca dgn pd.read_csv.

    Artefak yang DIKETAHUI dan diperhitungkan:
      - Sejak AUDIT-FIX #6 kolom GV_* berasal dari step yang SAMA (tidak lagi tertinggal 1 step),
        jadi pemeriksaan di sini ketat untuk semua baris. Dengan kode lama, baris pertama
        pasca-warmup akan gagal di cek GV_Weight (nilai warmup = 0).
      - GV_Offset adalah offset yang DIPAKAI step ini (nilai EMA sebelum di-update). Baris
        pertama pasca-warmup = 0 (EMA belum terinisialisasi), ini normal.
      - GV_Offset dicetak 4 desimal -> perubahan EMA < 5e-5 tampak 'tidak berubah'.
      - Mode 'small': batch tanpa gambar valid (tidak ada objek kecil di KEDUA branch)
        -> GV_Loss = GV_Target = 0 dan offset tidak di-update.
    """
    r = report or Report()
    r.check("CSV: header 11 kolom dgn urutan benar", list(df.columns) == CSV_COLS, list(df.columns))
    if list(df.columns) != CSV_COLS:
        return r

    exp_rows = (epochs - router_warmup_end) * n_batches
    r.check("CSV: jumlah baris = epoch pasca-warmup × batch/epoch", len(df) == exp_rows,
            f"{len(df)} vs {exp_rows}")
    r.check("CSV: tidak ada baris pada epoch warmup", int((df.Epoch <= router_warmup_end).sum()) == 0)

    num = df[CSV_COLS[2:]].astype(float)
    r.check("CSV: semua nilai finite (tidak NaN/Inf)", bool(np.isfinite(num.values).all()))

    # --- jadwal tercatat sesuai scheduler ---
    for e, g in df.groupby("Epoch"):
        h = sched_history.get(int(e))
        if h is None:
            r.add(f"E{e}: riwayat scheduler", "FAIL", "epoch tidak ada di riwayat scheduler")
            continue
        r.check(f"E{e}: kolom Lambda = jadwal ({h['lambda']:.4f})",
                np.allclose(g.Lambda.astype(float), h["lambda"], atol=1e-4))
        r.check(f"E{e}: kolom GV_Weight = jadwal ({h['gv']:.4f})",
                np.allclose(g.GV_Weight.astype(float), h["gv"], atol=1e-4))
        if h["lambda"] == 0:
            r.check(f"E{e}: Final_L3 = 0 saat lambda = 0", bool((g.Final_L3.astype(float) == 0).all()))
        else:
            r.check(f"E{e}: Final_L3 > 0 saat lambda > 0", bool((g.Final_L3.astype(float) > 0).all()),
                    f"min={g.Final_L3.min():.6f}")

    # --- rentang yang dijamin oleh kode ---
    r.check("P2_Prob_Real di [0, 1]", bool(df.P2_Prob_Real.between(0, 1).all()))
    r.check("Diff_W di [0.1, 2.0] (clamp di compute_router_loss)", bool(df.Diff_W.between(0.1 - 1e-6, 2.0 + 1e-6).all()),
            f"[{df.Diff_W.min():.4f}, {df.Diff_W.max():.4f}]")
    r.check("Rel ≥ 0", bool((df.Rel >= 0).all()))
    r.check("|GV_Loss| ≤ clip advantage (mean(p·adv), |adv| ≤ clip)",
            bool((df.GV_Loss.abs() <= offset_clip + 1e-6).all()), f"max |GV_Loss| = {df.GV_Loss.abs().max():.4f}")

    # --- kolom GV berubah tiap batch (baris pertama: offset EMA belum ada) ---
    g = df.iloc[1:].reset_index(drop=True)
    empty = ((g.GV_Loss == 0) & (g.GV_Target == 0)).mean()
    r.check("Batch tanpa sampel valid (GV_Loss = GV_Target = 0) < 5%", empty < 0.05, f"{empty:.1%}", warn_only=True)
    for col, thr in [("GV_Target", 0.95), ("GV_Loss", 0.95), ("GV_Offset", 0.80)]:
        frac = float((g[col].diff().iloc[1:] != 0).mean()) if len(g) > 1 else 0.0
        r.check(f"{col} berubah antar-batch (≥ {thr:.0%} batch)", frac >= thr, f"{frac:.1%}")

    # --- GV_Offset tidak menempel di ±clip ---
    at_clip = float((df.GV_Offset.abs() >= offset_clip - 1e-3).mean())
    r.check(f"GV_Offset tidak menempel di ±{offset_clip}", at_clip == 0, f"{at_clip:.1%} baris di batas")
    r.check("GV_Offset jauh dari batas (|offset| < 0.5·clip)", bool(df.GV_Offset.abs().max() < 0.5 * offset_clip),
            f"rentang [{df.GV_Offset.min():.4f}, {df.GV_Offset.max():.4f}]", warn_only=True)
    tail = df.GV_Offset.iloc[-max(len(df) // 4, 2):]
    r.check("GV_Offset tidak monoton naik/turun terus di 25% akhir (indikasi drift)",
            not (tail.diff().dropna() > 0).all() and not (tail.diff().dropna() < 0).all(),
            f"awal {tail.iloc[0]:.4f} → akhir {tail.iloc[-1]:.4f}", warn_only=True)

    # --- aktivasi tidak kolaps (indikator awal saja, 1 epoch tidak konklusif) ---
    last_e = df.Epoch.max()
    p_last = float(df[df.Epoch == last_e].P2_Prob_Real.mean())
    r.check(f"E{last_e}: rata-rata P2_Prob_Real di [0.05, 0.95] (bukan kolaps)", 0.05 <= p_last <= 0.95,
            f"{p_last:.3f}", warn_only=True)
    return r


def check_loss_items(li_df, router_warmup_end, sched_history, report=None):
    """Pemeriksaan 8 komponen loss per batch (loss_items.csv)."""
    r = report or Report()
    vals = li_df[LOSS_NAMES].astype(float)
    r.check("loss_items: 8 komponen finite di semua batch", bool(np.isfinite(vals.values).all()))
    for e, g in li_df.groupby("epoch"):
        lam = sched_history.get(int(e), {}).get("lambda", 0.0)
        warm = int(e) <= router_warmup_end
        rmean = float(g.router.mean())
        if warm or lam == 0:
            r.check(f"E{e}: komponen 'router' = 0 (warmup / lambda 0)", bool((g.router == 0).all()), f"mean={rmean:.6f}")
        else:
            r.check(f"E{e}: komponen 'router' > 0 (lambda > 0)", rmean > 0, f"mean={rmean:.6f}")
    # informatif (bukan kriteria lulus): 3 epoch terlalu pendek untuk menilai konvergensi
    print(li_df.groupby("epoch")[LOSS_NAMES].mean().round(4).to_string())
    return r


def budget_table(timing_df, plan, branch_b_interval=10, overhead=0.10, session_limit_h=12.0):
    """
    Estimasi jam GPU per run dari waktu epoch 'steady' (epoch ≥ 2; epoch 1 memuat cache,
    autotune cuDNN, check_amp, dsb).

    jam = (E·(t_train + t_valA) + floor(E/interval_B)·t_valB) · (1 + overhead) / 3600
    t_valA = val_total − valB (validasi Branch A + simpan checkpoint + callback lain).
    """
    st = timing_df[timing_df.epoch >= 2]
    if st.empty:
        st = timing_df
    t_train = st.train_s.mean()
    t_valB = st.valB_s.mean()
    t_valA = (st.val_total_s - st.valB_s).mean()
    rows = []
    for name, (E, n_runs) in plan.items():
        h = (E * (t_train + t_valA) + (E // branch_b_interval) * t_valB) * (1 + overhead) / 3600
        rows.append(dict(run=name, epochs=E, n_runs=n_runs, jam_per_run=round(h, 2),
                         jam_total=round(h * n_runs, 2),
                         perlu_resume=h > session_limit_h - 0.5))
    summary = dict(t_train_s=round(t_train, 1), t_valA_s=round(t_valA, 1), t_valB_s=round(t_valB, 1),
                   epoch1_train_s=round(float(timing_df.iloc[0].train_s), 1))
    return pd.DataFrame(rows), summary


# %% [2] Fungsi uji kalibrasi & konsistensi gate (butuh model, tanpa trainer)
CALIB_T_TEST = 1.2345678901234567   # nilai "tidak bulat" -> pembulatan fp16/fp32 pasti ketahuan
CALIB_B_TEST = -0.3456789012345678


def _calib(router):
    return float(getattr(router, "calib_T", 1.0)), float(getattr(router, "calib_b", 0.0))


def test_calibration_persistence(model, build_fresh_model, tmp_dir, report=None):
    """
    Uji 0.4. Set calib_T/calib_b, simpan, muat ulang lewat 3 jalur yang benar-benar dipakai:
      J1 pickle checkpoint Ultralytics : torch.save({'ema': deepcopy(model).half()}) -> torch.load
         (= trainer.save_model + attempt_load_one_weight)
      J2 state_dict -> model baru       : load_state_dict(strict=False)
         (= load_dualbranch() di notebook uji-akurasi-latensi)
      J3 Ultralytics model.load()       : intersect_dicts + load_state_dict
         (= DualBranchTrainer.get_model(weights=...), mis. fine-tune)

    Sejak AUDIT-FIX #9 kalibrasi disimpan sebagai buffer int64 dalam mikro-unit, jadi nilai
    yang tersimpan = round(v·1e6)/1e6. "Sama persis" di sini berarti: nilai yang DIBACA
    sebelum simpan == nilai setelah muat ulang (==, bukan allclose), dan galat kuantisasi
    set→baca ≤ 5e-7.
    """
    r = report or Report()
    tmp_dir = Path(tmp_dir); tmp_dir.mkdir(parents=True, exist_ok=True)
    rt = find_router(model)
    rt.calib_T, rt.calib_b = CALIB_T_TEST, CALIB_B_TEST
    ref = (rt.calib_T, rt.calib_b)
    r.check("Kalibrasi: galat kuantisasi set→baca ≤ 5e-7",
            abs(ref[0] - CALIB_T_TEST) <= 5e-7 and abs(ref[1] - CALIB_B_TEST) <= 5e-7,
            f"set=({CALIB_T_TEST}, {CALIB_B_TEST}) baca={ref}")
    keys = [k for k in model.state_dict() if "calib" in k]
    r.check("Kalibrasi ikut state_dict (buffer)", len(keys) == 2, keys)

    # J1
    p = tmp_dir / "calib_pickle.pt"
    torch.save({"ema": deepcopy(model).half()}, p)
    m1 = torch.load(p, map_location="cpu", weights_only=False)["ema"].float()
    rt1 = find_router(m1)
    r.check("Kalibrasi J1 (pickle checkpoint Ultralytics, lewat .half()) identik", _calib(rt1) == ref, f"{_calib(rt1)}")
    r.check("Buffer kalibrasi tetap int64 setelah .half()/.float()",
            all(b.dtype == torch.int64 for n, b in rt1.named_buffers() if "calib" in n))

    # J2
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, tmp_dir / "calib_state_dict.pt")
    sd2 = torch.load(tmp_dir / "calib_state_dict.pt", map_location="cpu")
    m2 = build_fresh_model()
    missing, unexpected = m2.load_state_dict(sd2, strict=False)
    r.check("Kalibrasi J2 (state_dict -> model baru) identik", _calib(find_router(m2)) == ref,
            f"{_calib(find_router(m2))} | missing={len(missing)} unexpected={len(unexpected)}")

    # J3
    m3 = build_fresh_model()
    m3.load(m1, verbose=False)
    r.check("Kalibrasi J3 (Ultralytics model.load) identik", _calib(find_router(m3)) == ref, f"{_calib(find_router(m3))}",
            warn_only=True)   # jalur fine-tune: ikut mewarisi kalibrasi -> sadari saat fine-tune
    return r, (m1, m2, m3)


@torch.no_grad()
def gate_decisions_both_paths(model, img):
    """
    Satu forward eval penuh (model(img) -> _predict_once -> router.forward eval), tangkap input
    router via forward-pre-hook, lalu panggil compute_gate_only pada input yang SAMA.
    Returns: (dec_forward atau None, n_active_forward, dec_gate_only, prob_gate_only)
    """
    rt = find_router(model)
    cap = {}
    h = rt.register_forward_pre_hook(lambda m, inp: cap.update(p3=inp[0][0], p2=inp[0][1]))
    try:
        model(img)
    finally:
        h.remove()
    B = img.shape[0]
    dec_fwd = getattr(rt, "_last_gate_decision", None)        # butuh patch P2
    dec_fwd = dec_fwd.float().flatten().clone() if dec_fwd is not None else None
    n_fwd = float(rt.current_activation_prob) * B
    dec_only = rt.compute_gate_only(cap["p3"], cap["p2"]).float().flatten()
    # probabilitas (untuk margin ke threshold) — rumus sama dgn compute_gate_only
    z = rt._topk_spatial_pool(rt.squeeze_p3(cap["p3"]) + rt.squeeze_p2(cap["p2"]),
                              k=getattr(rt, "gate_topk", 10), alpha=getattr(rt, "gate_topk_alpha", 0.7))
    l = 5.0 * torch.tanh(rt.classifier(z).view(B, 2).float() / 5.0)
    prob = torch.sigmoid((l[:, 1] - l[:, 0]) * getattr(rt, "calib_T", 1.0) + getattr(rt, "calib_b", 0.0))
    return dec_fwd, n_fwd, dec_only, prob


def test_gate_consistency(model, batches, calibs, report=None):
    """
    Uji 0.5. Untuk tiap kalibrasi & tiap batch: keputusan forward(eval) == compute_gate_only.
    `calibs`: list (nama, T, b). Model harus .eval(); presisi mengikuti dtype model.
    """
    r = report or Report()
    rt = find_router(model)
    saved = _calib(rt)
    model.eval()
    try:
        for name, T, b in calibs:
            rt.calib_T, rt.calib_b = T, b
            n_img = n_mis = 0
            n_fwd_tot = n_only_tot = 0.0
            min_margin = 1.0
            weak = False
            for img in batches:
                dec_fwd, n_fwd, dec_only, prob = gate_decisions_both_paths(model, img)
                n_img += img.shape[0]
                n_fwd_tot += n_fwd
                n_only_tot += float(dec_only.sum())
                min_margin = min(min_margin, float((prob - 0.5).abs().min()))
                if dec_fwd is None:
                    weak = True
                else:
                    n_mis += int((dec_fwd.cpu() != dec_only.cpu()).sum())
            if weak:
                r.check(f"Gate [{name}] jumlah aktif forward == compute_gate_only (VERSI LEMAH, patch P2 belum)",
                        abs(n_fwd_tot - n_only_tot) < 0.5,
                        f"{n_fwd_tot:.0f} vs {n_only_tot:.0f} dari {n_img}", warn_only=True)
            else:
                r.check(f"Gate [{name}] keputusan per-gambar identik", n_mis == 0,
                        f"{n_mis}/{n_img} beda | aktif {n_only_tot:.0f}/{n_img} "
                        f"({n_only_tot / max(n_img, 1):.1%}) | margin min |p−0.5| = {min_margin:.2e}")
    finally:
        rt.calib_T, rt.calib_b = saved
    return r


def percentile_bias_for_rate(model, batches, target_rate, T=1.0):
    """b sehingga fraksi (logit·T + b > 0) ≈ target_rate pada `batches` (gaya DynamicDet).
    Dipakai HANYA untuk membuat kalibrasi uji yang menaruh banyak sampel dekat threshold."""
    rt = find_router(model)
    saved = _calib(rt)
    rt.calib_T, rt.calib_b = 1.0, 0.0
    logits = []
    for img in batches:
        _, _, _, prob = gate_decisions_both_paths(model, img)
        logits.append(torch.logit(prob.clamp(1e-7, 1 - 1e-7)).cpu())
    rt.calib_T, rt.calib_b = saved
    lg = torch.cat(logits) * T
    return float(-torch.quantile(lg, 1.0 - target_rate))


# =============================================================================
# SEL DI BAWAH INI BUTUH ULTRALYTICS + GPU (Kaggle). Jalankan satu per satu.
# =============================================================================

# %% [3] Konfigurasi & RUN 3 EPOCH  (~3× waktu 1 epoch + 3× validasi A & B)
RUN_SMOKE = True
T0 = globals().get("T0_OVERRIDE", {})          # hanya untuk uji otomatis di luar Kaggle
if RUN_SMOKE:
    from ultralytics import settings
    from script_training import (DualBranchTrainer, make_router_stat_checker,  # butuh patch P1
                                 make_branch_b_checker)

    OUT = Path(T0.get("OUT", "/kaggle/working/tahap0")); OUT.mkdir(parents=True, exist_ok=True)
    settings.update({"raytune": False})

    # ---- SAMAKAN dengan run utama; hanya epochs/project/name/save_period yang beda ----
    BASE_OVERRIDES = dict(
        model="ultralytics/cfg/models/v8/yolov8n-p2-router.yaml",
        data="/kaggle/input/datasets/agungmaugi/kitti-2class/kitti_2class.yaml",
        batch=16, imgsz=640, optimizer="AdamW", lr0=0.002, cos_lr=True, weight_decay=0.0005,
        device=0, workers=4, seed=42, amp=True, verbose=True, save=True,
    )
    BASE_OVERRIDES.update(T0.get("BASE_OVERRIDES", {}))
    SMOKE_OVERRIDES = {**BASE_OVERRIDES, "epochs": 3, "project": str(OUT / "runs"),
                       "name": "smoke3", "exist_ok": False, "save_period": -1}

    # ---- jadwal smoke: E1 warmup | E2 gate loss penuh, lambda 0 | E3 gate loss + lambda penuh ----
    FINAL_LAMBDA = 0.04          # ISI = nilai final_lambda yang AKAN dipakai run utama
    SMOKE_SCHED = dict(router_warmup_end=1, gv_start=1, gv_ramp=1, final_gv=0.5,
                       lam_start=2, lam_ramp=1, final_lambda=FINAL_LAMBDA)

    # ---- hyperparameter router yang selama ini diam-diam pakai default getattr ----
    ROUTER_CFG = dict(
        gate_signal_mode="small",            # default model_yolov8_router.py
        # KEPUTUSAN 2026-10-09: suku target dimatikan (alpha = 0). Sparsity = "harga" P2
        #   c = lambda * Diff_W / gate_value_weight; activation rate deployment dipilih post-hoc
        #   (kalibrasi persentil, lihat operating_point_sweep.py). Tiga nilai di bawah tetap
        #   ditulis eksplisit, tapi TIDAK berpengaruh selama alpha = 0 (kolom Rel tetap dihitung).
        router_hybrid_alpha=0.0,
        router_target_activation=0.35,
        router_activation_tolerance=0.10,
        router_min_activation_floor=0.05,
        branch_b_loss_weight=0.7,
        gate_offset_momentum=0.99,
    )

    SCHED_HISTORY, LOSS_ITEMS = {}, []
    timer = EpochTimer()
    trainer = DualBranchTrainer(overrides=SMOKE_OVERRIDES)
    CSV_PATH = OUT / "router_loss_components.csv"
    r_start, r_batch, r_end = make_router_stat_checker(trainer)

    trainer.add_callback("on_pretrain_routine_end", make_router_cfg_applier(ROUTER_CFG, OUT / "router_cfg.json"))
    trainer.add_callback("on_train_epoch_start", make_curriculum_scheduler(**SMOKE_SCHED, history=SCHED_HISTORY))
    trainer.add_callback("on_train_epoch_start", timer.on_train_epoch_start)
    trainer.add_callback("on_train_batch_end", make_loss_items_logger(LOSS_ITEMS))
    trainer.add_callback("on_train_epoch_end", timer.on_train_epoch_end)       # sebelum flush CSV
    trainer.add_callback("on_train_epoch_end", make_router_csv_logger(CSV_PATH))
    trainer.add_callback("on_val_start", r_start)
    trainer.add_callback("on_val_batch_end", r_batch)
    trainer.add_callback("on_val_end", r_end)
    trainer.add_callback("on_val_end", timer.timed(make_branch_b_checker(trainer, interval=1)))  # uji jalur Branch B
    trainer.add_callback("on_fit_epoch_end", timer.on_fit_epoch_end)

    t_all = time.perf_counter()
    trainer.train()
    print(f"⏱️  Total wall-clock smoke run: {(time.perf_counter() - t_all) / 60:.1f} menit")

    # simpan log mentah
    pd.DataFrame(LOSS_ITEMS, columns=["epoch"] + LOSS_NAMES).to_csv(OUT / "loss_items.csv", index=False)
    pd.DataFrame(timer.rows).to_csv(OUT / "epoch_timing.csv", index=False)
    json.dump({str(k): v for k, v in SCHED_HISTORY.items()}, open(OUT / "sched_history.json", "w"), indent=1)
    # branch_b_metrics.csv di-hardcode ke /kaggle/working -> pindahkan supaya tidak tercampur run utama
    bb = Path(T0.get("BRANCH_B_CSV", "/kaggle/working/branch_b_metrics.csv"))
    if bb.exists():
        shutil.move(str(bb), OUT / "branch_b_metrics.csv")

# %% [4] Pemeriksaan otomatis: CSV router, loss, results.csv, konfigurasi efektif
REPORT = Report()
df = pd.read_csv(CSV_PATH)
n_b = int(timer.rows[0]["n_batches"])
check_router_csv(df, n_b, SMOKE_OVERRIDES["epochs"], SMOKE_SCHED["router_warmup_end"], SCHED_HISTORY,
                 offset_clip=getattr(trainer.model.criterion, "offset_clip_range", 1.0), report=REPORT)
li_df = pd.read_csv(OUT / "loss_items.csv")
check_loss_items(li_df, SMOKE_SCHED["router_warmup_end"], SCHED_HISTORY, report=REPORT)

res = pd.read_csv(Path(trainer.save_dir) / "results.csv")
res.columns = [c.strip() for c in res.columns]
REPORT.check("results.csv: 3 baris epoch", len(res) == SMOKE_OVERRIDES["epochs"], len(res))
REPORT.check("results.csv: semua kolom numerik finite",
             bool(np.isfinite(res.select_dtypes("number").values).all()))
print(res.filter(regex="epoch|mAP50|train/").round(4).to_string(index=False))
bbm = OUT / "branch_b_metrics.csv"
# final_eval() memicu on_val_end sekali lagi -> 1 baris ekstra (duplikat epoch terakhir) itu normal
REPORT.check("Branch B tervalidasi tiap epoch (DualBranchValidatorB jalan)",
             bbm.exists() and set(pd.read_csv(bbm).epoch) >= set(range(1, SMOKE_OVERRIDES["epochs"] + 1)))

crit = trainer.model.criterion
eff = dict(branch_b_weight=crit.branch_b_weight, offset_momentum=crit.offset_momentum,
           use_bce_gate_loss=crit.use_bce_gate_loss,
           offset_clip_range=getattr(crit, "offset_clip_range", 1.0),
           gate_adv_clip=getattr(crit, "gate_adv_clip", getattr(crit, "offset_clip_range", 1.0)),
           running_diff_mean_akhir=float(crit.running_diff_mean),
           calib_T=getattr(find_router(trainer.model), "calib_T", "tidak di-set"))
print("Konfigurasi efektif criterion:", eff)
REPORT.check("Jalur Opsi C aktif (use_bce_gate_loss=False)", eff["use_bce_gate_loss"] is False)
REPORT.check("router_hybrid_alpha efektif = ROUTER_CFG (0 -> target_activation tidak berpengaruh)",
             float(getattr(trainer.model, "router_hybrid_alpha", -1)) == ROUTER_CFG["router_hybrid_alpha"])
REPORT.check("branch_b_weight & offset_momentum = ROUTER_CFG",
             eff["branch_b_weight"] == ROUTER_CFG["branch_b_loss_weight"]
             and eff["offset_momentum"] == ROUTER_CFG["gate_offset_momentum"])

# plot jejak router
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for a, col in zip(ax, ["GV_Offset", "GV_Target", "P2_Prob_Real", "GV_Loss"]):
    a.plot(df[col].values, lw=0.8); a.set_ylabel(col); a.grid(alpha=0.3)
    for k in range(1, df.Epoch.nunique()):
        a.axvline(k * n_b, color="grey", ls="--", lw=0.8)
ax[-1].set_xlabel("batch (pasca-warmup); garis putus = batas epoch")
fig.tight_layout(); fig.savefig(OUT / "router_trace.png", dpi=120)

# %% [5] Waktu per epoch & anggaran
timing = pd.DataFrame(timer.rows)
print(timing.round(1).to_string(index=False))
PLAN = {                               # ISI sesuai bagian Anggaran: nama -> (epoch, jumlah run)
    "utama_100ep": (100, 1),
    "ablasi_30ep": (30, 4),
}
budget, tsum = budget_table(timing, PLAN, branch_b_interval=10)
print(tsum); print(budget.to_string(index=False))
print(f"TOTAL jam GPU rencana: {budget.jam_total.sum():.1f} jam")
budget.to_csv(OUT / "budget.csv", index=False)
REPORT.check("Epoch 1 vs steady: selisih train < 25% (warmup router tidak mengubah biaya)",
             abs(tsum["epoch1_train_s"] - tsum["t_train_s"]) / tsum["t_train_s"] < 0.25,
             f"E1={tsum['epoch1_train_s']}s, steady={tsum['t_train_s']}s", warn_only=True)

# %% [6] Uji kalibrasi tersimpan (0.4) & konsistensi gate (0.5)
from ultralytics.model_yolov8_router import DualBranchDetectionModel
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NC = trainer.data["nc"]

def build_fresh_model():
    return DualBranchDetectionModel(cfg=BASE_OVERRIDES["model"], ch=3, nc=NC, verbose=False)

ck = torch.load(Path(trainer.save_dir) / "weights" / "last.pt", map_location="cpu", weights_only=False)
# strip_optimizer() di final_eval memindahkan ckpt["ema"] -> ckpt["model"] dan mengosongkan "ema"
model = (ck.get("ema") or ck["model"]).float().eval()

# --- 0.4 kalibrasi tersimpan ---
_, (m1, m2, m3) = test_calibration_persistence(model, build_fresh_model, OUT / "calib_test", report=REPORT)
# keputusan gate model asli vs hasil muat ulang J1/J2 harus identik (uji fungsional)
model = model.to(DEV); m1 = m1.to(DEV).eval(); m2 = m2.to(DEV).eval()
val_batches = []
val_loader = getattr(trainer, "test_loader", None) or trainer.get_dataloader(
    trainer.data["val"], batch_size=BASE_OVERRIDES["batch"], rank=-1, mode="val")
for i, b in enumerate(val_loader):
    val_batches.append(b["img"].to(DEV).float() / 255.0)
    if i == 4:
        break
for nm, mm in [("J1", m1), ("J2", m2)]:
    d0 = torch.cat([gate_decisions_both_paths(model, x)[2] for x in val_batches])
    d1 = torch.cat([gate_decisions_both_paths(mm, x)[2] for x in val_batches])
    REPORT.check(f"Keputusan gate model asli == model muat ulang {nm}", torch.equal(d0, d1),
                 f"{int((d0 != d1).sum())}/{d0.numel()} beda")

# --- 0.5 konsistensi gate forward(eval) vs compute_gate_only ---
b_pct = percentile_bias_for_rate(model, val_batches, target_rate=0.40)

# saturasi logit: kalibrasi persentil butuh logit yang tersebar; menumpuk di batas tanh ±10 = ranking seri
_saved = _calib(find_router(model)); find_router(model).calib_T, find_router(model).calib_b = 1.0, 0.0
_raw = torch.cat([torch.logit(gate_decisions_both_paths(model, x)[3].clamp(1e-7, 1 - 1e-7)).cpu() for x in val_batches])
find_router(model).calib_T, find_router(model).calib_b = _saved
_sat = float((_raw.abs() > 9.5).float().mean())
REPORT.check("Logit gate tidak tersaturasi (|raw| > 9.5 pada < 20% gambar val)", _sat < 0.20,
             f"{_sat:.1%} | raw mean={_raw.mean():.3f} std={_raw.std():.3f}", warn_only=True)
REPORT.check("Logit gate tersebar antar-gambar (std > 0.05; ranking tidak seri)", float(_raw.std()) > 0.05,
             f"std={_raw.std():.4f}, unik={_raw.unique().numel()}/{_raw.numel()}", warn_only=True)
CALIBS = [("default T=1,b=0", 1.0, 0.0),
          ("uji T,b tidak bulat", CALIB_T_TEST, CALIB_B_TEST),
          (f"persentil 40% (b={b_pct:.4f})", 1.0, b_pct)]
test_gate_consistency(model, val_batches, CALIBS, report=REPORT)              # FP32 (= engine TRT)
test_gate_consistency(deepcopy(model).half(), [x.half() for x in val_batches], CALIBS[:1], report=REPORT)  # FP16 (= val saat training)

# %% [7] Ringkasan
json.dump(REPORT.items, open(OUT / "sanity_report.json", "w"), indent=1)
n = {s: sum(i["status"] == s for i in REPORT.items) for s in ("PASS", "WARN", "FAIL")}
print(f"\nRINGKASAN TAHAP 0: {n}")
print("➡️  LANJUT ke Tahap 1" if n["FAIL"] == 0 else "⛔ JANGAN lanjut — perbaiki item FAIL dulu.")
