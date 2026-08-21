import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DualBranchDetectionLoss:
    """
    Wrapper yang membungkus 2 instance v8DetectionLoss — satu untuk Detect_A
    (4 skala, termasuk router penalty), satu untuk Detect_B (3 skala, TANPA
    router penalty supaya tidak dobel-hitung).
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
        self.loss_A._compute_router_penalty = True  # flag kontrol (lihat catatan di bawah)

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

        # --- TAMBAHAN TAHAP 2: hyperparameter baru ---
        self.gate_value_weight = getattr(raw_model, "gate_value_loss_weight", 0.5)
        self.gate_value_margin = getattr(raw_model, "gate_value_margin", 0.02)

        self._raw_model = raw_model
        self._router_cache = None  # cache supaya tidak loop modules() tiap panggilan

    def _find_router(self):
        if self._router_cache is not None:
            return self._router_cache
        for m in self._raw_model.modules():
            if 'DifficultyAwareRouter' in m.__class__.__name__:
                self._router_cache = m
                return m
        return None


class DualBranchDetectionLoss:
    def __init__(self, model):
        raw_model = model.module if hasattr(model, "module") else model

        self.loss_A = v8DetectionLoss(model)
        self.loss_A.stride = raw_model.detect_A.stride
        self.loss_A.nc = raw_model.detect_A.nc
        self.loss_A.no = raw_model.detect_A.nc + raw_model.detect_A.reg_max * 4
        self.loss_A.reg_max = raw_model.detect_A.reg_max
        self.loss_A.use_dfl = raw_model.detect_A.reg_max > 1
        self.loss_A.assigner.num_classes = raw_model.detect_A.nc
        self.loss_A._compute_router_penalty = True

        self.loss_B = v8DetectionLoss(model)
        self.loss_B.stride = raw_model.detect_B.stride
        self.loss_B.nc = raw_model.detect_B.nc
        self.loss_B.no = raw_model.detect_B.nc + raw_model.detect_B.reg_max * 4
        self.loss_B.reg_max = raw_model.detect_B.reg_max
        self.loss_B.use_dfl = raw_model.detect_B.reg_max > 1
        self.loss_B.assigner.num_classes = raw_model.detect_B.nc
        self.loss_B._compute_router_penalty = False

        self.branch_b_weight = getattr(raw_model, "branch_b_loss_weight", 0.7)

        # --- TAMBAHAN TAHAP 2: hyperparameter baru ---
        self.gate_value_weight = getattr(raw_model, "gate_value_loss_weight", 0.5)
        self.gate_value_margin = getattr(raw_model, "gate_value_margin", 0.02)

        self._raw_model = raw_model
        self._router_cache = None  # cache supaya tidak loop modules() tiap panggilan

    def _find_router(self):
        if self._router_cache is not None:
            return self._router_cache
        for m in self._raw_model.modules():
            if 'DifficultyAwareRouter' in m.__class__.__name__:
                self._router_cache = m
                return m
        return None

    def __call__(self, preds, batch):
        det_A, det_B = preds

        loss_A_sum, loss_A_items = self.loss_A(det_A, batch)
        loss_B_sum, loss_B_items = self.loss_B(det_B, batch)

        total_loss = loss_A_sum + self.branch_b_weight * loss_B_sum
        combined_items = torch.cat([loss_A_items, loss_B_items[:3]])

        # =====================================================================
        # TAMBAHAN TAHAP 2: gate_value_loss
        # =====================================================================
        router = self._find_router()
        if (
            router is not None
            and self._raw_model.training
            and hasattr(router, "_last_gate_prob_per_sample")
            and hasattr(self.loss_A, "_last_per_sample_loss")
            and hasattr(self.loss_B, "_last_per_sample_loss")
        ):
            per_sample_loss_A = self.loss_A._last_per_sample_loss  # (B,)
            per_sample_loss_B = self.loss_B._last_per_sample_loss  # (B,)
            gate_prob = router._last_gate_prob_per_sample           # (B,)

            target_gate = (per_sample_loss_A < per_sample_loss_B - self.gate_value_margin).float().detach()

            gate_value_loss = F.binary_cross_entropy(
                gate_prob.clamp(1e-6, 1 - 1e-6), target_gate
            )

            batch_size = combined_items.new_tensor(gate_prob.shape[0])
            total_loss = total_loss + self.gate_value_weight * gate_value_loss * batch_size

            # simpan utk logging/debug opsional
            self._last_gate_value_loss = gate_value_loss.detach()
            self._last_target_gate_mean = target_gate.mean().detach()
        # =====================================================================

        return total_loss, combined_items.detach()