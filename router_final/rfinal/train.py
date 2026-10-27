"""
Training detektor dual-branch (router_final) di atas DetectionTrainer stok Ultralytics 8.3.240.

Yang berbeda dari DetectionTrainer bawaan hanya:
  - model = DualBranchDetector, loss = DualBranchLoss (6 item: box/cls/dfl × A/B);
  - validasi per epoch (dan best.pt) = Branch A (perilaku bawaan, jalur _predict_once);
  - opsional: mAP Branch B di val tiap N epoch -> branch_b_metrics.csv (pemantauan, tidak mengubah training);
  - provenance.json dan sha256 bobot akhir (weights_sha256.json); resep efektif lengkap ada di args.yaml Ultralytics;
  - penjaga: AMP efektif harus sama dengan yang diminta (check_amp Ultralytics bisa diam-diam mematikannya
    bila uji AMP gagal/offline — resep berubah tanpa terlihat).
"""
from __future__ import annotations

import csv
from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from .env import provenance, require_stock_ultralytics, sha256_file, write_json
from .loss import DualBranchLoss
from .model import DualBranchDetector


class DualBranchTrainer(DetectionTrainer):
    branch_b_weight = 0.7
    branch_b_val_every = 10

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DualBranchDetector(cfg=cfg or self.args.model, ch=3, nc=self.data["nc"], verbose=verbose)
        model.branch_b_loss_weight = float(self.branch_b_weight)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = DualBranchLoss.names
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                  _callbacks=self.callbacks)


class BranchBValidator(DetectionValidator):
    """Validator bawaan, tetapi model.eval_branch = 'B' selama validasi (selalu dipulihkan)."""

    def __call__(self, trainer=None, model=None):
        target = trainer.ema.ema if trainer is not None and getattr(trainer, "ema", None) else (model or trainer.model)
        old = getattr(target, "eval_branch", "A")
        target.eval_branch = "B"
        try:
            return super().__call__(trainer=trainer, model=model)
        finally:
            target.eval_branch = old


def _cb_branch_b(trainer):
    every = int(getattr(trainer, "branch_b_val_every", 0) or 0)
    epoch = trainer.epoch + 1
    if every <= 0 or (epoch % every and epoch != trainer.epochs):
        return
    if getattr(trainer, "_branch_b_logged_epoch", None) == epoch:   # final_eval memanggil callback ini sekali lagi
        return
    trainer._branch_b_logged_epoch = epoch
    args = copy(trainer.args)
    args.plots = False                                              # jangan timpa plot validasi Branch A
    v = BranchBValidator(trainer.test_loader, save_dir=trainer.save_dir, args=args)
    v(trainer=trainer)
    bm = v.metrics.box
    path = Path(trainer.save_dir) / "branch_b_metrics.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "mAP50", "mAP50-95", "precision", "recall"])
        w.writerow([epoch, bm.map50, bm.map, bm.mp, bm.mr])


def _cb_provenance(cfg_files):
    def f(trainer):
        if bool(trainer.amp) != bool(trainer.args.amp):
            raise RuntimeError(f"AMP efektif ({bool(trainer.amp)}) != diminta ({trainer.args.amp}): check_amp "
                               "Ultralytics gagal/offline -> resep berubah. Perbaiki lingkungan, jangan lanjut.")
        write_json(Path(trainer.save_dir) / "provenance.json",
                   provenance(files=[*cfg_files, trainer.args.data],
                              extra=dict(amp_effective=bool(trainer.amp), model_kind=getattr(trainer, "model_kind", "dualbranch"),
                                         branch_b_weight=getattr(trainer, "branch_b_weight", None),
                                         seed=trainer.args.seed, deterministic=trainer.args.deterministic,
                                         torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())))
    return f


def _cb_final_hash(trainer):
    w = Path(trainer.save_dir) / "weights"
    write_json(Path(trainer.save_dir) / "weights_sha256.json",
               {p.name: sha256_file(p) for p in sorted(w.glob("*.pt"))})
    # Hash tensor (tidak bergantung pada metadata pickle seperti tanggal) -> dipakai uji reproduksi
    from .repro import state_hash
    ck = torch.load(w / "last.pt", map_location="cpu", weights_only=False)
    write_json(Path(trainer.save_dir) / "state_hash.json",
               {k: state_hash(ck[k]) for k in ("model", "ema") if ck.get(k) is not None})


def build_trainer(cfg, seed, name=None, overrides=None, strict_determinism=False):
    """
    cfg: dict dari env.load_config. Returns DualBranchTrainer siap .train().
    strict_determinism: torch.use_deterministic_algorithms(True) TANPA warn_only -> operasi non-deterministik
                        langsung error (dipakai untuk diagnosis bila uji reproduksi gagal).
    """
    require_stock_ultralytics()
    args = dict(cfg["train"], model=cfg["model"], data=cfg["data"], seed=int(seed),
                name=name or f"{cfg['name']}_seed{seed}", exist_ok=False, **(overrides or {}))
    trainer = DualBranchTrainer(overrides=args)
    trainer.branch_b_weight = float(cfg["loss"]["branch_b_weight"])
    trainer.branch_b_val_every = int(cfg.get("monitor", {}).get("branch_b_val_every", 0))
    trainer.add_callback("on_pretrain_routine_end", _cb_provenance([cfg["_config_path"], cfg["model"]]))
    trainer.add_callback("on_fit_epoch_end", _cb_branch_b)
    trainer.add_callback("on_train_end", _cb_final_hash)
    if strict_determinism:
        def _strict(_):
            torch.use_deterministic_algorithms(True, warn_only=False)
        trainer.add_callback("on_pretrain_routine_start", _strict)
    return trainer


# ---------------------------------------------------------------------------- baseline (Vanilla / P2-Static)
# Model STOK Ultralytics 8.3.240, dilatih dari nol dengan resep yang SAMA dengan detektor dual-branch (blok `train`
# konfigurasi: epoch, batch, optimizer, lr, augmentasi bawaan, seed, AMP, worker), di GPU yang sama. Run lama
# (VisDrone, Colab A100) juga memakai yolov8n.yaml / yolov8n-p2.yaml dari nol dengan argumen yang sama.
BASELINE_YAMLS = {"vanilla": "yolov8n.yaml", "p2static": "yolov8n-p2.yaml"}


def build_baseline_trainer(cfg, seed, which, name=None, overrides=None, strict_determinism=False):
    """which: 'vanilla' | 'p2static'. Keluaran sama dengan build_trainer (provenance, state_hash) tanpa Branch B."""
    require_stock_ultralytics()
    assert which in BASELINE_YAMLS, which
    args = dict(cfg["train"], model=BASELINE_YAMLS[which], data=cfg["data"], seed=int(seed),
                name=name or f"{cfg['name']}_{which}_seed{seed}", exist_ok=False, **(overrides or {}))
    trainer = DetectionTrainer(overrides=args)
    trainer.model_kind = which
    trainer.add_callback("on_pretrain_routine_end", _cb_provenance([cfg["_config_path"]]))
    trainer.add_callback("on_train_end", _cb_final_hash)
    if strict_determinism:
        def _strict(_):
            torch.use_deterministic_algorithms(True, warn_only=False)
        trainer.add_callback("on_pretrain_routine_start", _strict)
    return trainer
