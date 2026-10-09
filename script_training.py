"""
script_training.py — Training Router-P2 YOLOv8n Dual-Branch (Skenario 3), versi importable.

Perubahan dibanding kode_berkomentar/script_training.py (logika model/loss TIDAK berubah):
  - Section 1 lama (patch loss.py di disk lewat penggantian string) DIHAPUS: guard
    _compute_router_penalty dan default lambda sudah ada di loss.py, sehingga patch itu
    tidak pernah mengubah apa pun, dan menulis file library saat runtime berisiko.
  - Eksekusi dibungkus build_trainer() + main(); import file ini tidak memulai training.
  - ideal_curriculum_scheduler diganti make_curriculum_scheduler (router_train_utils),
    yang dengan FULL_SCHED identik dengan jadwal lama (diuji epoch 1–100) kecuali lambda
    final = 0.04 (keputusan 2026-10-09).
  - Hyperparameter router ditulis eksplisit (ROUTER_CFG); router_hybrid_alpha = 0.
  - Semua log ke save_dir run: router_loss_components.csv (11 kolom, label epoch benar),
    loss_items.csv, epoch_timing.csv, branch_b_metrics.csv, router_cfg.json, sched_history.json.
  - Penjaga NaN per batch (gagal cepat, bukan NaN recovery diam-diam Ultralytics).
  - Sebaran logit gate per validasi (gate_val_stats.csv) dan go/no-go ranking gate di akhir
    epoch lam_start (50), tepat sebelum lambda naik (gate_check.json; NO-GO -> berhenti).
  - Butuh patch sumber (notebook sel 4): P2 logit/keputusan gate per-gambar, tanda conf (+).
"""
import os
import csv
import json
from copy import copy
from pathlib import Path

import pandas as pd
import torch
from ultralytics import settings
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.model_yolov8_router import DualBranchDetectionModel

from router_train_utils import (EpochTimer, make_curriculum_scheduler, make_gate_ranking_check,
                                make_loss_items_logger, make_router_cfg_applier, make_router_csv_logger)

# ==============================================================================
# KONFIGURASI RUN UTAMA
# ==============================================================================
BASE_OVERRIDES = dict(
    model="ultralytics/cfg/models/v8/yolov8n-p2-router.yaml",
    data="/kaggle/input/datasets/agungmaugi/kitti-2class/kitti_2class.yaml",
    batch=16, imgsz=640, optimizer="AdamW", lr0=0.002, cos_lr=True, weight_decay=0.0005,
    device=0, workers=4, seed=42, amp=True, verbose=True, save=True,
)
FULL_OVERRIDES = {**BASE_OVERRIDES, "epochs": 100, "project": "Tesis_KITTI_Yolov8_nano",
                  "name": "Skenario3_YOLOv8n_P2_Router_DualBranch", "save_period": 50}

# Fase 1 (1–50): lambda 0; router warmup 1–10; gate_value_weight 0 -> 0.5 di epoch 11–30.
# Fase 2 (51–80): lambda 0 -> final linear. Fase 3 (81–100): lambda final.
FULL_SCHED = dict(router_warmup_end=10, gv_start=10, gv_ramp=20, final_gv=0.5,
                  lam_start=50, lam_ramp=30, final_lambda=0.04)

# Keputusan 2026-10-09: alpha = 0 -> router_target_activation tidak berpengaruh; sparsity bekerja
# sebagai harga P2 (c = lambda * Diff_W / gate_value_weight); activation rate deployment dipilih
# post-hoc (operating_point_sweep.py). Nilai target tetap ditulis supaya konfigurasi eksplisit.
ROUTER_CFG = dict(
    gate_signal_mode="small",
    router_hybrid_alpha=0.0,
    router_target_activation=0.35,
    router_activation_tolerance=0.10,
    router_min_activation_floor=0.05,
    branch_b_loss_weight=0.7,
    gate_offset_momentum=0.99,
)


# ==============================================================================
# TRAINER & VALIDATOR
# ==============================================================================
class DualBranchTrainer(DetectionTrainer):
    """DetectionTrainer yang membangun DualBranchDetectionModel dan 8 nama komponen loss."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DualBranchDetectionModel(cfg=cfg or self.args.model, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        # Urutan = combined_items di DualBranchDetectionLoss: 5 item loss_A + 3 item loss_B.
        self.loss_names = ("box_A", "cls_A", "dfl_A", "router", "proxy", "box_B", "cls_B", "dfl_B")
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                  _callbacks=self.callbacks)


class DualBranchValidatorB(DetectionValidator):
    """Validator yang mengevaluasi Branch B: _predict_once diganti sementara, selalu dipulihkan."""

    def __call__(self, trainer=None, model=None):
        if model is not None:
            target_model = model
        elif trainer is not None:
            target_model = trainer.ema.ema if (hasattr(trainer, "ema") and trainer.ema) else trainer.model
        else:
            raise ValueError("DualBranchValidatorB butuh `trainer` atau `model`.")
        assert hasattr(target_model, "_predict_once_dual"), "Model bukan DualBranchDetectionModel."

        original = target_model._predict_once

        def branch_b_predict_once(x, profile=False, visualize=False, embed=None):
            _, det_B = target_model._predict_once_dual(x, profile, visualize, embed)
            return det_B

        target_model._predict_once = branch_b_predict_once
        try:
            return super().__call__(trainer=trainer, model=model)
        finally:
            target_model._predict_once = original


# ==============================================================================
# CALLBACK VALIDASI
# ==============================================================================
def make_router_stat_checker(trainer_ref, csv_path=None):
    """
    Aktivasi P2 + sebaran logit gate mentah di validasi, per epoch -> csv_path.
    Angka ini TANPA kalibrasi, FP16, dataloader rect -> bukan angka deployment; gunanya memantau
    apakah logit gate tersebar antar-gambar (syarat kalibrasi persentil) sepanjang training.
    Butuh patch P2 (router._last_gate_logit_raw); tanpa patch hanya aktivasi yang tercatat.
    """
    stats = {"total": 0.0, "n": 0, "logits": []}

    def on_val_start(validator):
        stats["total"], stats["n"], stats["logits"] = 0.0, 0, []

    def on_val_batch_end(validator):
        m_ = trainer_ref.ema.ema if (hasattr(trainer_ref, "ema") and trainer_ref.ema) else trainer_ref.model
        if m_ is None:
            return
        for m in m_.modules():
            if "DifficultyAwareRouter" in m.__class__.__name__:
                v = getattr(m, "current_activation_prob", None)
                if v is not None:
                    stats["total"] += v.item() if hasattr(v, "item") else float(v)
                    stats["n"] += 1
                lg = getattr(m, "_last_gate_logit_raw", None)
                if lg is not None:
                    stats["logits"].append(lg.detach().float().flatten().cpu())
                break

    def on_val_end(validator):
        if not stats["n"]:
            return
        act = stats["total"] / stats["n"]
        msg = f"P2 AKTIF: {act * 100:.2f}%"
        row = {"epoch": trainer_ref.epoch + 1, "act_uncalibrated": act}
        if stats["logits"]:
            lg = torch.cat(stats["logits"])
            row.update(n=int(lg.numel()), logit_mean=float(lg.mean()), logit_std=float(lg.std()),
                       logit_p10=float(lg.quantile(0.1)), logit_p90=float(lg.quantile(0.9)),
                       frac_saturated=float((lg.abs() > 9.5).float().mean()))
            msg += f" | logit mean={row['logit_mean']:.3f} std={row['logit_std']:.4f}"
        print(f"\n📉 [EVALUASI ROUTER] {msg}\n")
        if csv_path is not None:
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not Path(csv_path).exists(), index=False)

    return on_val_start, on_val_batch_end, on_val_end


def make_branch_b_checker(trainer_ref, interval=10, csv_path=None):
    """Evaluasi Branch B tiap `interval` epoch; hasil ke csv_path (default: save_dir/branch_b_metrics.csv)."""
    def on_val_end_check_branch_b(validator):
        epoch = trainer_ref.epoch + 1
        if epoch % interval != 0:
            return
        path = Path(csv_path or Path(trainer_ref.save_dir) / "branch_b_metrics.csv")
        vb = DualBranchValidatorB(dataloader=trainer_ref.test_loader, save_dir=trainer_ref.save_dir,
                                  args=copy(trainer_ref.args))
        try:
            vb(trainer=trainer_ref)
            bm = vb.metrics.box
            print(f"🔍 [BRANCH B] Epoch {epoch} | mAP50: {bm.map50:.4f} | mAP50-95: {bm.map:.4f} | "
                  f"P: {bm.mp:.4f} | R: {bm.mr:.4f}")
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["epoch", "mAP50", "mAP50-95", "precision", "recall"])
                w.writerow([epoch, bm.map50, bm.map, bm.mp, bm.mr])
        except Exception as e:  # jangan hentikan training karena evaluasi tambahan
            import traceback
            print(f"⚠️  [BRANCH B CHECK GAGAL] {type(e).__name__}: {e}")
            traceback.print_exc()

    return on_val_end_check_branch_b


# ==============================================================================
# BANGUN TRAINER (dipakai run utama DAN smoke test)
# ==============================================================================
def build_trainer(overrides, sched, router_cfg=ROUTER_CFG, branch_b_interval=10,
                  gate_check_epoch="auto", stop_on_nogo=True):
    """
    Returns (trainer, ctx). ctx berisi objek log yang bisa diperiksa setelah train():
      timer, sched_history, loss_items_cb, paths (semua di trainer.save_dir).
    Catatan: save_dir sudah ditentukan di konstruktor trainer, jadi path log aman dipakai.

    gate_check_epoch: epoch go/no-go ranking gate ("auto" = sched["lam_start"], tepat sebelum
    lambda naik; None = mati). NO-GO + stop_on_nogo -> training berhenti setelah epoch itu.
    """
    settings.update({"raytune": False})
    resuming = bool(overrides.get("resume"))
    trainer = DualBranchTrainer(overrides=overrides)
    sd = Path(trainer.save_dir); sd.mkdir(parents=True, exist_ok=True)
    paths = dict(router_csv=sd / "router_loss_components.csv", loss_items=sd / "loss_items.csv",
                 timing=sd / ("epoch_timing_resume.csv" if resuming else "epoch_timing.csv"), branch_b=sd / "branch_b_metrics.csv",
                 router_cfg=sd / "router_cfg.json", sched=sd / "sched_history.json",
                 gate_val=sd / "gate_val_stats.csv", gate_check=sd / "gate_check.json")

    sched_history, timer = {}, EpochTimer()
    li_store = []
    li_cb = make_loss_items_logger(li_store, paths["loss_items"])
    r_start, r_batch, r_end = make_router_stat_checker(trainer, paths["gate_val"])
    if gate_check_epoch == "auto":
        gate_check_epoch = sched["lam_start"]

    def save_epoch_logs(tr):
        pd.DataFrame(timer.rows).to_csv(paths["timing"], index=False)
        paths["sched"].write_text(json.dumps({str(k): v for k, v in sched_history.items()}, indent=1))

    trainer.add_callback("on_pretrain_routine_end", make_router_cfg_applier(router_cfg, paths["router_cfg"]))
    trainer.add_callback("on_train_epoch_start", make_curriculum_scheduler(**sched, history=sched_history))
    trainer.add_callback("on_train_epoch_start", timer.on_train_epoch_start)
    trainer.add_callback("on_train_batch_end", li_cb)
    trainer.add_callback("on_train_epoch_end", timer.on_train_epoch_end)          # sebelum flush CSV
    trainer.add_callback("on_train_epoch_end", make_router_csv_logger(paths["router_csv"], allow_existing=resuming))
    trainer.add_callback("on_train_epoch_end", li_cb.flush)
    trainer.add_callback("on_val_start", r_start)
    trainer.add_callback("on_val_batch_end", r_batch)
    trainer.add_callback("on_val_end", r_end)
    trainer.add_callback("on_val_end", timer.timed(make_branch_b_checker(trainer, branch_b_interval, paths["branch_b"])))
    trainer.add_callback("on_fit_epoch_end", timer.on_fit_epoch_end)
    trainer.add_callback("on_fit_epoch_end", save_epoch_logs)
    if gate_check_epoch:
        trainer.add_callback("on_fit_epoch_end", make_gate_ranking_check(
            trainer, gate_check_epoch, paths["gate_check"], signal_mode=router_cfg.get("gate_signal_mode", "small"),
            stop_on_nogo=stop_on_nogo))

    ctx = dict(timer=timer, sched_history=sched_history, loss_items_cb=li_cb, paths=paths,
               sched=sched, router_cfg=router_cfg, overrides=overrides, gate_check_epoch=gate_check_epoch)
    return trainer, ctx


def main():
    print("🔥 SKENARIO 3: Router-P2 YOLOv8n Dual-Branch (Difficulty-Aware Router)")
    trainer, ctx = build_trainer(FULL_OVERRIDES, FULL_SCHED, ROUTER_CFG, branch_b_interval=10)
    trainer.train()
    return trainer, ctx


if __name__ == "__main__":
    main()
