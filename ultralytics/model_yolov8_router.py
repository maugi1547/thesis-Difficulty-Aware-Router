import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss


class DualBranchDetectionModel(DetectionModel):
    """
    DetectionModel dengan dua Detect head terpisah:
      - detect_A: P2 + P3 + P4 + P5 (branch WithP2, dipakai saat objek sulit/kecil)
      - detect_B: P3 + P4 + P5      (branch NoP2,   dipakai saat objek mudah/tidak ada)

    Kedua branch dilatih bersamaan (multi-task), tapi punya bobot Detect terpisah.
    Router (UltraLightWeightDifficultyAwareRouter) menentukan gate, namun SAAT
    TRAINING kedua branch tetap dihitung penuh untuk supervisi.
    True-skip baru terjadi nanti saat export terpisah ke TensorRT (Stage 1/2A/2B).
    """

    def __init__(self, cfg="yolov8-p2-router.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.detect_layers = [m for m in self.model if isinstance(m, Detect)]
        assert len(self.detect_layers) == 2, (
            f"Expected exactly 2 Detect heads in YAML, found {len(self.detect_layers)}."
        )
        self.detect_A, self.detect_B = self.detect_layers

        # --- Stride & bias init untuk Detect_A ---
        s = 256
        dummy = torch.zeros(1, ch, s, s)
        was_training = self.training

        self.model.eval()          # matikan BN/dropout stat update
        self.detect_A.training = True   # TAPI paksa Detect_A/B return list mentah (bukan decoded tuple)
        self.detect_B.training = True

        with torch.no_grad():
            det_A_out, det_B_out = self._predict_once_dual(dummy)

        self.detect_A.stride = torch.tensor([s / x.shape[-2] for x in det_A_out])
        self.stride = self.detect_A.stride
        self.detect_A.bias_init()

        self.detect_B.stride = torch.tensor([s / x.shape[-2] for x in det_B_out])
        self.detect_B.bias_init()

        # kembalikan training flag Detect ke default (akan di-set benar oleh self.train()/eval() nanti)
        self.detect_A.training = was_training
        self.detect_B.training = was_training
        self.model.train(was_training)

        if verbose:
            print(f"[DualBranchDetectionModel] Detect_A stride: {self.detect_A.stride.tolist()} "
                f"({len(self.detect_A.stride)} scales)")
            print(f"[DualBranchDetectionModel] Detect_B stride: {self.detect_B.stride.tolist()} "
                f"({len(self.detect_B.stride)} scales)")
            
    # -----------------------------------------------------------------
    # FORWARD PASS — replikasi persis _predict_once bawaan, + tangkap
    # output kedua Detect head secara terpisah.
    # -----------------------------------------------------------------
    def _predict_once_dual(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)

        det_A_out, det_B_out = None, None
        detect_A = getattr(self, "detect_A", None)  # <-- guard: None saat masih di dalam super().__init__()
        detect_B = getattr(self, "detect_B", None)

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if profile:
                self._profile_one_layer(m, x, dt)

            x = m(x)

            if detect_A is not None and m is detect_A:
                det_A_out = x
            elif detect_B is not None and m is detect_B:
                det_B_out = x
            elif isinstance(m, Detect) and detect_A is None and detect_B is None:
                # fallback SEMENTARA saat super().__init__() masih berjalan
                # (detect_A/detect_B belum di-assign) — anggap ini "single detect"
                det_A_out = x
                det_B_out = x

            y.append(x if m.i in self.save else None)

            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)

            if m.i in embed:
                embeddings.append(
                    torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return det_A_out, det_B_out

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Dipanggil oleh semua jalur internal Ultralytics (predict/val/export/profile)
        yang mengharapkan SATU output. Default: branch A (full P2, kualitas tertinggi).
        """
        det_A_out, _ = self._predict_once_dual(x, profile, visualize, embed)
        return det_A_out

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # training path — trainer kirim batch dict
            return self.loss(x, *args, **kwargs)
        return super().forward(x, *args, **kwargs)  # predict/val/export biasa -> branch A saja

    # -----------------------------------------------------------------
    # LOSS — kirim KEDUA branch ke criterion
    # -----------------------------------------------------------------
    def loss(self, batch, preds=None):
        """
        Override loss() Ultralytics. CATATAN PENTING:
        Parameter `preds` SENGAJA DIABAIKAN meski dikirim oleh DetectionValidator,
        karena format preds dari situ (hasil forward() biasa -> branch A only)
        TIDAK KOMPATIBEL dengan DualBranchDetectionLoss yang butuh (det_A, det_B).
        Konsekuensinya: saat validasi, forward pass terjadi 2x per batch
        (1x oleh validator utk metric, 1x di sini utk loss) — sedikit overhead,
        tapi mencegah bug shape-mismatch yang jauh lebih berbahaya.
        """
        if not hasattr(self, "criterion") or self.criterion is None:
            self.criterion = self.init_criterion()

        img = batch["img"]
        preds_dual = self._predict_once_dual(img)  # SELALU recompute, jangan pakai preds argumen

        return self.criterion(preds_dual, batch)

    def init_criterion(self):
        return DualBranchDetectionLoss(self)

"""
OPSI C — Soft-mixture router objective dengan EMA offset (tanpa freeze, tanpa BCE target)
============================================================================

Menggantikan gate_value_loss lama (BCEWithLogits terhadap sigmoid(diff/temperature))
dengan objective linear ala DynamicDet:

    p_i = sigmoid(gate_logit_i)
    L_router_i = (1 - p_i) * (loss_A_i.detach() - offset/2) + p_i * (loss_B_i.detach() + offset/2)

di mana `offset` adalah EMA berjalan dari (loss_A_i - loss_B_i), BUKAN raw diff per-batch,
supaya sinyal yang dikejar gate lebih stabil (meredam "moving target problem").

Gradient efektif ke gate_logit_i: dL/dgate_logit_i ∝ -(loss_A_i - loss_B_i - offset)
— sampel di mana P2 (Branch A) jauh lebih baik dari Branch B (relatif thd offset)
akan mendorong p_i naik (P2 dinyalakan); sebaliknya mendorong p_i turun.

CATATAN PENTING:
- loss_A_i, loss_B_i di-detach() SEBELUM dipakai di sini, sehingga gradient dari
  L_router TIDAK ikut mengalir ke bobot Branch A/B/backbone/neck. Update bobot
  branch tetap murni dari `loss_A_sum + branch_b_weight * loss_B_sum` seperti biasa.
- offset (EMA) mengoreksi bias sistematis akibat perbedaan kecepatan konvergen
  Branch A vs B (temuan Anda sebelumnya), sehingga gate tidak selalu bias ke satu
  arah, tapi merespons DEVIASI RELATIF dari level sistematis saat itu.
- compute_router_loss (sparsity) TIDAK diubah di sini — tetap jalan seperti biasa
  (nanti bisa didekopling terpisah sesuai diskusi Opsi A, tapi itu perubahan lain).
"""

class DualBranchDetectionLoss:
    """
    Wrapper yang membungkus 2 instance v8DetectionLoss — satu untuk Detect_A
    (4 skala, termasuk router penalty), satu untuk Detect_B (3 skala, TANPA
    router penalty supaya tidak dobel-hitung).

    Versi ini menggunakan OPSI C untuk gate training signal: soft-mixture
    objective dengan EMA offset, bukan BCE terhadap target sigmoid+temperature.
    """

    def __init__(self, model):
        raw_model = model.module if hasattr(model, "module") else model

        # --- Loss A: dapat router penalty (compute_router_loss aktif) ---
        self.loss_A = v8DetectionLoss(model)
        self.loss_A.stride = raw_model.detect_A.stride
        self.loss_A.nc = raw_model.detect_A.nc
        self.loss_A.no = raw_model.detect_A.nc + raw_model.detect_A.reg_max * 4
        self.loss_A.reg_max = raw_model.detect_A.reg_max
        self.loss_A.use_dfl = raw_model.detect_A.reg_max > 1
        self.loss_A.assigner.num_classes = raw_model.detect_A.nc
        self.loss_A._compute_router_penalty = True

        # --- Loss B: TANPA router penalty (cegah double-count) ---
        self.loss_B = v8DetectionLoss(model)
        self.loss_B.stride = raw_model.detect_B.stride
        self.loss_B.nc = raw_model.detect_B.nc
        self.loss_B.no = raw_model.detect_B.nc + raw_model.detect_B.reg_max * 4
        self.loss_B.reg_max = raw_model.detect_B.reg_max
        self.loss_B.use_dfl = raw_model.detect_B.reg_max > 1
        self.loss_B.assigner.num_classes = raw_model.detect_B.nc
        self.loss_B._compute_router_penalty = False

        self.branch_b_weight = getattr(raw_model, "branch_b_loss_weight", 0.7)

        # --- Hyperparameter gate routing (OPSI C) ---
        self.gate_value_weight = getattr(raw_model, "gate_value_loss_weight", 0.5)

        # Momentum EMA untuk offset. 0.99 = offset berubah pelan, sangat stabil.
        # Kalau ingin offset lebih responsif terhadap perubahan training dinamis,
        # bisa diturunkan (mis. 0.9), tapi risiko lebih noisy.
        self.offset_momentum = getattr(raw_model, "gate_offset_momentum", 0.99)

        # Dipertahankan untuk kompatibilitas mundur (tidak dipakai di jalur Opsi C,
        # tapi bisa diaktifkan lagi kalau mau A/B test vs BCE lama).
        self.gate_value_temperature = getattr(raw_model, "gate_value_temperature", 0.15)
        self.use_bce_gate_loss = getattr(raw_model, "use_bce_gate_loss", False)  # default: pakai Opsi C

        self._raw_model = raw_model
        self._router_cache = None

        # State EMA offset — di-registrasi sebagai buffer biasa (bukan nn.Parameter,
        # karena tidak dioptimasi via gradient, hanya diupdate manual tiap step).
        self._offset_initialized = False
        self.running_diff_mean = torch.tensor(0.0)

        # Logging state
        self._last_gate_value_loss = torch.tensor(0.0)
        self._last_target_gate_mean = torch.tensor(0.0)  # dipertahankan utk kompatibilitas logging lama
        self._last_offset = torch.tensor(0.0)
        self._last_mean_p = torch.tensor(0.0)

    def _find_router(self):
        if self._router_cache is not None:
            return self._router_cache
        for m in self._raw_model.modules():
            if 'DifficultyAwareRouter' in m.__class__.__name__:
                self._router_cache = m
                return m
        return None

    def _update_offset(self, batch_diff_mean: torch.Tensor):
        """EMA update untuk running_diff_mean. batch_diff_mean HARUS sudah detached."""
        if self.running_diff_mean.device != batch_diff_mean.device:
            self.running_diff_mean = self.running_diff_mean.to(batch_diff_mean.device)

        if not self._offset_initialized:
            # Inisialisasi langsung ke nilai batch pertama, hindari bias awal dari 0.0
            # saat loss_A/loss_B masih jauh dari skala matang.
            self.running_diff_mean = batch_diff_mean.clone()
            self._offset_initialized = True
        else:
            m = self.offset_momentum
            self.running_diff_mean = m * self.running_diff_mean + (1.0 - m) * batch_diff_mean

    def __call__(self, preds, batch):
        det_A, det_B = preds

        loss_A_sum, loss_A_items = self.loss_A(det_A, batch)
        loss_B_sum, loss_B_items = self.loss_B(det_B, batch)

        total_loss = loss_A_sum + self.branch_b_weight * loss_B_sum
        combined_items = torch.cat([loss_A_items, loss_B_items[:3]])

        current_gate_value_weight = getattr(self._raw_model, "gate_value_loss_weight", 0.0)

        router = self._find_router()
        is_warmup = getattr(router, "_is_warmup", True) if router is not None else True

        can_compute_gate_loss = (
            router is not None
            and self._raw_model.training
            and not is_warmup
            and current_gate_value_weight > 0.0
            and hasattr(router, "_last_gate_logit_per_sample")
            and hasattr(self.loss_A, "_last_per_sample_loss")
            and hasattr(self.loss_B, "_last_per_sample_loss")
        )

        if can_compute_gate_loss:
            # --- PILIH SUMBER SINYAL: 'total' | 'small' | 'hybrid' ---
            gate_signal_mode = getattr(self._raw_model, 'gate_signal_mode', 'small')

            if gate_signal_mode == 'small':
                per_sample_loss_A = self.loss_A._last_per_sample_loss_small.detach()
                per_sample_loss_B = self.loss_B._last_per_sample_loss_small.detach()
                valid_A = self.loss_A._last_small_obj_count > 0
                valid_B = self.loss_B._last_small_obj_count > 0
                valid_mask = valid_A & valid_B

            elif gate_signal_mode == 'hybrid':
                raw_w_s, raw_w_m, raw_w_l = 0.7, 0.2, 0.1
                # Ganti bobot tetap dengan normalisasi dinamis per-gambar
                has_small_a = (self.loss_A._last_small_obj_count > 0).float()
                has_medium_a = (self.loss_A._last_medium_obj_count > 0).float()  # perlu tambah tracking count_medium juga
                has_large_a = (self.loss_A._last_large_obj_count > 0).float()

                active_w_sum_a = raw_w_s * has_small_a + raw_w_m * has_medium_a + raw_w_l * has_large_a
                active_w_sum_a = active_w_sum_a.clamp(min=1e-6)

                per_sample_loss_A = ((
                    raw_w_s * has_small_a * self.loss_A._last_per_sample_loss_small 
                    + raw_w_m * has_medium_a * self.loss_A._last_per_sample_loss_medium 
                    + raw_w_l * has_large_a * self.loss_A._last_per_sample_loss_large
                ) / active_w_sum_a).detach()

                has_small_b = (self.loss_B._last_small_obj_count > 0).float()
                has_medium_b = (self.loss_B._last_medium_obj_count > 0).float()  
                has_large_b = (self.loss_B._last_large_obj_count > 0).float()

                active_w_sum_b = raw_w_s * has_small_b + raw_w_m * has_medium_b + raw_w_l * has_large_b
                active_w_sum_b = active_w_sum_b.clamp(min=1e-6)

                per_sample_loss_B = ((
                    raw_w_s * has_small_b * self.loss_B._last_per_sample_loss_small 
                    + raw_w_m * has_medium_b * self.loss_B._last_per_sample_loss_medium 
                    + raw_w_l * has_large_b * self.loss_B._last_per_sample_loss_large
                ) / active_w_sum_b).detach()

                valid_mask = torch.ones_like(per_sample_loss_A, dtype=torch.bool)

            else:  # 'total' — perilaku lama
                per_sample_loss_A = self.loss_A._last_per_sample_loss.detach()
                per_sample_loss_B = self.loss_B._last_per_sample_loss.detach()
                valid_mask = torch.ones_like(per_sample_loss_A, dtype=torch.bool)

            gate_logit = router._last_gate_logit_per_sample  # (B,) TIDAK di-detach

            # ==========================================================
            # SATU jalur perhitungan: masking diterapkan SEBELUM apa pun,
            # supaya offset & loss sama-sama hanya melihat sampel valid.
            # ==========================================================
            if valid_mask.sum() == 0:
                zero = torch.tensor(0.0, device=combined_items.device)
                gate_value_loss = zero
                mean_p = zero
                offset = self.running_diff_mean.clone()
                n_valid = 0
            else:
                gate_logit_v = gate_logit[valid_mask]
                loss_A_v = per_sample_loss_A[valid_mask]
                loss_B_v = per_sample_loss_B[valid_mask]
                n_valid = int(valid_mask.sum().item())

                # offset dari step SEBELUMNYA, di-update SEKALI saja
                offset = self.running_diff_mean.clone()
                self._update_offset((loss_A_v - loss_B_v).mean().detach())

                if self.use_bce_gate_loss:
                    # jalur lama (BCE) — kini juga menghormati valid_mask
                    diff = (loss_B_v - loss_A_v)
                    target_gate = torch.sigmoid(diff / self.gate_value_temperature)
                    gate_value_loss = F.binary_cross_entropy_with_logits(gate_logit_v, target_gate)
                    mean_p = torch.sigmoid(gate_logit_v).mean().detach()
                else:
                    # OPSI C: soft-mixture dengan EMA offset
                    p = torch.sigmoid(gate_logit_v)
                    weighted_loss = (1.0 - p) * (loss_B_v + offset / 2.0) \
                                    + p * (loss_A_v - offset / 2.0)
                    gate_value_loss = weighted_loss.mean()
                    mean_p = p.mean().detach()

            # scaling pakai jumlah sampel VALID (0 kalau tidak ada -> gate loss mati)
            batch_size = combined_items.new_tensor(float(n_valid))
            total_loss = total_loss + current_gate_value_weight * gate_value_loss * batch_size

            if router is not None:
                router.last_target_gate_mean = mean_p
                router.last_gate_value_loss = gate_value_loss.detach()
                router.last_gate_value_weight_active = torch.tensor(current_gate_value_weight)
                router.last_gate_offset = offset  # baru: expose offset utk logging/debug

            self._last_gate_value_loss = gate_value_loss.detach()
            self._last_target_gate_mean = mean_p
            self._last_offset = offset
            self._last_mean_p = mean_p
        else:
            zero = torch.tensor(0.0, device=combined_items.device)
            if router is not None:
                router.last_target_gate_mean = zero
                router.last_gate_value_loss = zero
                router.last_gate_value_weight_active = zero
                router.last_gate_offset = zero
            self._last_gate_value_loss = zero
            self._last_target_gate_mean = zero
            self._last_offset = zero
            self._last_mean_p = zero

        return total_loss, combined_items.detach()