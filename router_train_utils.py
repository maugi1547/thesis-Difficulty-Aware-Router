"""
router_train_utils.py — utilitas murni untuk training & sanity check Router-P2 Dual-Branch.

Dipakai bersama oleh script_training.py (run utama) dan notebook smoke test, supaya yang
diuji di Tahap 0 adalah kode yang sama dengan yang dipakai run 100 epoch.
Isi: scheduler berparameter, penerap ROUTER_CFG, logger CSV router (per-run, 11 kolom,
label epoch diperbaiki), logger loss + penjaga NaN, timer epoch, pemeriksa CSV/loss,
estimasi anggaran, uji kalibrasi tersimpan, uji konsistensi gate.
"""
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


def make_router_csv_logger(csv_path, allow_existing=False):
    """
    Pengganti on_train_epoch_end di script_training.py:
      - path per-run (bukan /kaggle/working/... global yang di-append lintas run),
      - menolak file yang sudah ada (cegah data run lama tercampur),
      - memverifikasi tiap baris berisi 11 kolom,
      - kolom Epoch diganti epoch ASLI (label di loss.py = debug_counter//total_batches + 4,
        salah setiap kali warmup tidak berakhir di epoch 4). Tetap 11 kolom.
    """
    csv_path = Path(csv_path)
    if csv_path.exists() and not allow_existing:      # allow_existing=True hanya untuk resume run yang sama
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


def make_loss_items_logger(store, csv_path=None):
    """
    on_train_batch_end: simpan 8 komponen loss per batch + GAGAL CEPAT bila NaN/Inf.
    (Ultralytics 8.3.240 punya NaN recovery yang diam-diam memuat ulang checkpoint.)
    Bila csv_path diberikan, `store` ditulis ke file & dikosongkan lewat .flush(trainer)
    (dipasang di on_train_epoch_end) supaya run 100 epoch tidak menumpuk di RAM.
    """
    state = {"header": csv_path is not None and not Path(csv_path).exists()}

    def on_train_batch_end(trainer):
        li = trainer.loss_items.detach().float().cpu()
        epoch = trainer.epoch + 1
        store.append([epoch] + li.tolist())
        if not torch.isfinite(li).all() or not torch.isfinite(trainer.loss.detach()).all():
            raise FloatingPointError(f"Loss non-finite di epoch {epoch}, batch #{len(store)}: "
                                     f"{dict(zip(LOSS_NAMES, li.tolist()))}")

    def flush(trainer):
        if csv_path is None or not store:
            return
        pd.DataFrame(store, columns=["epoch"] + LOSS_NAMES).to_csv(
            csv_path, mode="a", header=state["header"], index=False)
        state["header"] = False
        store.clear()

    on_train_batch_end.flush = flush
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
# PATCH SUMBER (EOL-aware, idempotent) — dipakai notebook sebelum import ultralytics
# =============================================================================
def apply_source_patch(path, old, new, done_marker, name):
    """
    Ganti `old` -> `new` tepat satu kali di file `path`, mempertahankan EOL file (LF/CRLF).
    Idempotent: bila `done_marker` sudah ada, file tidak diubah. Returns status string.
    """
    path = Path(path)
    raw = path.read_bytes().decode("utf-8")
    if done_marker in raw:
        return f"ℹ️ {name}: sudah ada — tidak di-patch."
    if "\r\n" in raw:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    n = raw.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: pola ditemukan {n}x (harus 1) di {path} — versi file berbeda, cek manual.")
    path.write_bytes(raw.replace(old, new).encode("utf-8"))
    return f"✅ {name}: di-patch."


PATCH_P2 = dict(
    name="P2 keputusan & logit gate per-gambar (block.py, eval)",
    rel="ultralytics/nn/modules/block.py",
    done_marker="self._last_gate_logit_raw = raw_logit_diff.detach()",
    old=("            if not torch.jit.is_tracing():\n"
         "                self.current_activation_prob = gate_mask.mean().detach()\n\n"
         "            f_c2f = self.compute_expert(f_p3, f_p2_back)\n"),
    new=("            if not torch.jit.is_tracing():\n"
         "                self.current_activation_prob = gate_mask.mean().detach()\n"
         "                # [TAHAP0] per-gambar: keputusan (uji konsistensi vs compute_gate_only) dan logit\n"
         "                # mentah sebelum kalibrasi (sebaran logit di log validasi)\n"
         "                self._last_gate_decision = gate_mask.view(B).detach()\n"
         "                self._last_gate_logit_raw = raw_logit_diff.detach()\n\n"
         "            f_c2f = self.compute_expert(f_p3, f_p2_back)\n"),
)

PATCH_CONF_SIGN = dict(
    name="Tanda conf di difficulty score (loss.py): - menjadi +",
    rel="ultralytics/utils/loss.py",
    done_marker="diff_score = entropy_f + var_f + conf_f",
    old=("                # TODO: verifikasi -- router.last_conf adalah UNCERTAINTY (1 - prob, tinggi =\n"
         "                # tidak yakin), sama arah dengan entropy dan var. Mengurangkannya di sini\n"
         "                # berarti ketidakpastian kelas yang lebih tinggi MENURUNKAN skor kesulitan.\n"
         "                # Apakah tanda minus ini disengaja (mis. dulu conf berarti keyakinan)?\n"
         "                diff_score = entropy_f + var_f - conf_f\n"),
    new=("                # [FIX 2026-10-09] router.last_conf adalah UNCERTAINTY (1 - prob, tinggi = tidak\n"
         "                # yakin), searah dengan entropy dan var. Tanda minus versi lama adalah salah ketik\n"
         "                # (dikonfirmasi penulis): ketiga sinyal ketidakpastian kini DIJUMLAHKAN.\n"
         "                # Run sebelum fix ini memakai entropy + var - conf (tidak sebanding langsung).\n"
         "                diff_score = entropy_f + var_f + conf_f\n"),
)


# =============================================================================
# GO/NO-GO KUALITAS RANKING GATE (mis. checkpoint epoch 50, sebelum lambda naik)
# =============================================================================
def _raw_gate_logit(rt, p3, p2):
    """Logit gate mentah (sebelum kalibrasi) — rumus identik dengan compute_gate_only."""
    z = rt._topk_spatial_pool(rt.squeeze_p3(p3) + rt.squeeze_p2(p2),
                              k=getattr(rt, "gate_topk", 10), alpha=getattr(rt, "gate_topk_alpha", 0.7))
    l = 5.0 * torch.tanh(rt.classifier(z).view(p3.shape[0], 2).float() / 5.0)
    return l[:, 1] - l[:, 0]


def _auc_with_ci(labels, scores, n_boot=1000, seed=0):
    """AUC (Mann-Whitney, ties = 0.5) + CI 95% bootstrap persentil. labels: 1 = Branch A lebih baik."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)

    def auc(lb, sc):
        n1, n0 = lb.sum(), (~lb).sum()
        if n1 == 0 or n0 == 0:
            return np.nan
        r = rankdata(sc)
        return (r[lb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    a = auc(labels, scores)
    rng = np.random.default_rng(seed); n = len(labels); boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(auc(labels[i], scores[i]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return float(a), float(lo), float(hi)


@torch.no_grad()
def gate_ranking_report(model, loader, device, max_batches=None, signal_mode="small",
                        min_std=0.05, n_boot=1000):
    """
    Apakah gate sudah MERANKING gambar sesuai sinyal yang melatihnya? (model beku, FP32, eval)

    Per gambar val: logit gate mentah + loss_A/loss_B per-sample (total & small, dari criterion
    yang sama dengan training). Oracle = 1 bila loss_A < loss_B. Untuk mode 'small' hanya
    gambar dengan objek kecil ter-assign di KEDUA branch yang dipakai (sama dgn valid_mask training).

    GO bila: std logit > min_std DAN batas bawah CI 95% AUC (mode sinyal training) > 0.5.
    Catatan: oracle memakai sinyal loss yang noisy — AUC menjawab "apakah gate mempelajari
    sinyal yang diberikan", bukan "apakah keputusan itu benar secara semantik".
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    model = deepcopy(model).float().eval().to(device)
    if isinstance(getattr(model, "args", None), dict) or not hasattr(model, "args"):
        a = getattr(model, "args", {}) or {}
        model.args = get_cfg(DEFAULT_CFG, overrides={k: a[k] for k in ("box", "cls", "dfl") if k in a})
    rt = find_router(model)
    crit = model.init_criterion()
    feats, rows = {}, {k: [] for k in ("raw", "la", "lb", "las", "lbs", "nsa", "nsb")}
    h = rt.register_forward_pre_hook(lambda m, inp: feats.update(p3=inp[0][0], p2=inp[0][1]))
    try:
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            img = batch["img"].to(device).float() / 255.0
            b = {"img": img, **{k: batch[k].to(device) for k in ("cls", "bboxes", "batch_idx")}}
            det_A, det_B = model._predict_once_dual(img)
            crit.loss_A(det_A, b); crit.loss_B(det_B, b)
            rows["raw"].append(_raw_gate_logit(rt, feats["p3"], feats["p2"]).cpu())
            rows["la"].append(crit.loss_A._last_per_sample_loss.float().cpu())
            rows["lb"].append(crit.loss_B._last_per_sample_loss.float().cpu())
            rows["las"].append(crit.loss_A._last_per_sample_loss_small.float().cpu())
            rows["lbs"].append(crit.loss_B._last_per_sample_loss_small.float().cpu())
            rows["nsa"].append(crit.loss_A._last_small_obj_count.float().cpu())
            rows["nsb"].append(crit.loss_B._last_small_obj_count.float().cpu())
    finally:
        h.remove()
    R = {k: torch.cat(v).numpy() for k, v in rows.items()}
    raw = R["raw"]
    prob = 1.0 / (1.0 + np.exp(-raw))
    out = dict(n_images=int(len(raw)), logit_mean=float(raw.mean()), logit_std=float(raw.std()),
               prob_std_uncalibrated=float(prob.std()),
               logit_p10=float(np.percentile(raw, 10)), logit_p90=float(np.percentile(raw, 90)),
               frac_saturated=float((np.abs(raw) > 9.5).mean()),
               activation_uncalibrated=float((raw > 0).mean()))
    a, lo, hi = _auc_with_ci(R["la"] < R["lb"], raw, n_boot)
    out.update(total=dict(n=int(len(raw)), oracle_rate=float((R["la"] < R["lb"]).mean()), auc=a, auc_ci=[lo, hi]))
    v = (R["nsa"] > 0) & (R["nsb"] > 0)
    if v.sum() >= 20:
        a, lo, hi = _auc_with_ci(R["las"][v] < R["lbs"][v], raw[v], n_boot)
        out.update(small=dict(n=int(v.sum()), oracle_rate=float((R["las"][v] < R["lbs"][v]).mean()),
                              auc=a, auc_ci=[lo, hi]))
    key = signal_mode if signal_mode in out else "total"
    out["decision_signal"] = key
    out["go"] = bool(out["logit_std"] > min_std and out[key]["auc_ci"][0] > 0.5)
    out["reason"] = (f"std={out['logit_std']:.4f} (> {min_std}?), AUC[{key}]={out[key]['auc']:.3f} "
                     f"CI95=[{out[key]['auc_ci'][0]:.3f}, {out[key]['auc_ci'][1]:.3f}] (batas bawah > 0.5?)")
    return out


def stop_training(trainer, reason):
    """
    Hentikan training dengan rapi setelah epoch ini. final_eval() -> strip_optimizer() membuat
    last.pt TIDAK bisa di-resume (epoch=-1, optimizer dibuang), jadi salinan utuh disimpan SEKARANG
    (save_model sudah jalan untuk epoch ini). Returns path salinan.
    """
    epoch = trainer.epoch + 1
    keep = Path(trainer.wdir) / f"last_resumable_epoch{epoch}.pt"
    # final_eval() memicu on_fit_epoch_end sekali lagi SETELAH last.pt di-strip -> jangan timpa salinan
    if getattr(trainer, "stop", False) or (Path(trainer.save_dir) / "STOPPED.txt").exists():
        return str(keep)
    if Path(trainer.last).exists():
        shutil.copy(trainer.last, keep)
    (Path(trainer.save_dir) / "STOPPED.txt").write_text(f"epoch {epoch}: {reason}\nresume: {keep}\n")
    print(f"⛔ Training dihentikan di epoch {epoch}: {reason}. Untuk lanjut: resume dari {keep}")
    trainer.stop = True
    return str(keep)


def make_activation_stop_rule(gate_val_csv, after_epoch=60, low=0.05, high=0.95, patience=5):
    """
    Aturan berhenti Tahap 1: aktivasi val (EMA, tanpa kalibrasi, dari gate_val_stats.csv) < low atau
    > high selama `patience` epoch berturut-turut SETELAH `after_epoch`. Dipasang SESUDAH callback yang
    menulis gate_val_stats.csv (on_val_end -> sudah tertulis saat on_fit_epoch_end).
    """
    def cb(trainer):
        epoch = trainer.epoch + 1
        if getattr(trainer, "stop", False) or epoch <= after_epoch or not Path(gate_val_csv).exists():
            return
        gv = pd.read_csv(gate_val_csv).drop_duplicates("epoch", keep="first").set_index("epoch")
        win = [e for e in range(epoch - patience + 1, epoch + 1)]
        if all(e in gv.index and e > after_epoch for e in win):
            a = gv.loc[win, "act_uncalibrated"]
            if (a < low).all() or (a > high).all():
                stop_training(trainer, f"aktivasi val {a.round(3).tolist()} di luar [{low}, {high}] "
                                       f"selama {patience} epoch (epoch {win[0]}–{win[-1]})")
    return cb


def make_criterion_cfg_applier(criterion_cfg, dump_path=None):
    """
    on_pretrain_routine_end (SETELAH router cfg): buat criterion sekarang dan set atribut yang dibaca
    v8DetectionLoss dari DIRINYA SENDIRI (mis. normalize_bg_anchors), bukan dari model.
    Model.loss() hanya membuat criterion bila belum ada, jadi criterion ini yang dipakai training.
    """
    def apply(trainer):
        m = trainer.model
        m.criterion = m.init_criterion()
        for k, v in criterion_cfg.items():
            setattr(m.criterion.loss_A, k, v); setattr(m.criterion.loss_B, k, v)
        if dump_path is not None:
            Path(dump_path).write_text(json.dumps(criterion_cfg, indent=1))
        print("🔧 [CRITERION CFG]", criterion_cfg)
    return apply


def make_gate_ranking_check(trainer_ref, at_epoch, out_path, signal_mode="small", stop_on_nogo=True,
                            max_batches=None):
    """
    on_fit_epoch_end: di akhir epoch `at_epoch`, jalankan gate_ranking_report pada model EMA
    (val loader trainer). NO-GO + stop_on_nogo -> trainer.stop = True (training berhenti rapi,
    last.pt tersimpan, bisa dilanjutkan dengan resume=True bila diputuskan lanjut).
    """
    def cb(trainer):
        epoch = trainer.epoch + 1
        if epoch != at_epoch or Path(out_path).exists():
            return
        device = next(trainer.model.parameters()).device
        rep = gate_ranking_report(trainer.ema.ema, trainer.test_loader, device, max_batches=max_batches,
                                  signal_mode=signal_mode)
        rep.update(epoch=epoch, stop_on_nogo=stop_on_nogo)
        Path(out_path).write_text(json.dumps(rep, indent=1))
        print(f"\n🚦 [GATE CHECK epoch {epoch}] {'GO' if rep['go'] else 'NO-GO'} — {rep['reason']}\n")
        if not rep["go"] and stop_on_nogo:
            rep["resumable_checkpoint"] = stop_training(trainer, "gate check NO-GO")
            Path(out_path).write_text(json.dumps(rep, indent=1))
    return cb
