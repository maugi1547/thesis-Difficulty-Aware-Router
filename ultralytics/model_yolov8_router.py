"""
model_yolov8_router.py
======================

Model dan loss dual-branch untuk Difficulty-Aware Router pada YOLOv8n-P2.

Gambaran besar:
- DualBranchDetectionModel: satu graf YOLOv8 (dari YAML yolov8-p2-router.yaml) yang
  punya DUA Detect head. Backbone + neck top-down dipakai bersama; Branch A (WithP2,
  4 skala P2/P3/P4/P5) dan Branch B (NoP2, 3 skala P3/P4/P5) punya jalur bottom-up dan
  Detect head masing-masing. Router (UltraLightWeightDifficultyAwareRouter, di block.py)
  duduk di antara neck dan Branch A, menghitung fitur P2 (C2f-P2) + logit gate.
- DualBranchDetectionLoss: menjumlahkan loss deteksi kedua branch, lalu (setelah warmup)
  menambahkan gate_value_loss yang melatih gate memilih branch per-gambar berdasarkan
  perbandingan loss nyata Branch A vs Branch B.

Prinsip desain yang dipegang di seluruh file:
1. SAAT TRAINING kedua branch SELALU dihitung penuh (gate tidak memotong komputasi),
   supaya kedua head selalu mendapat supervisi dan loss_A/loss_B per-sampel selalu
   tersedia sebagai sinyal pelatihan gate.
2. True-skip (benar-benar melewati komputasi P2) hanya terjadi saat deployment
   TensorRT 3-engine (Stage 1 / 2A / 2B), di luar file ini.
3. Jalur bawaan Ultralytics (predict/val/export) yang mengharapkan SATU output
   diarahkan ke Branch A (lihat _predict_once).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time  # catatan: tidak dipakai lagi di versi ini (sisa profiling waktu loss_A/loss_B)
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

    Kenapa dua Detect head terpisah (bukan satu head yang di-skip sebagian)?
    Modul Detect Ultralytics mengikat jumlah skala input ke bobotnya (cv2/cv3 per
    skala), sehingga satu head tidak bisa dipakai bergantian untuk 4 skala dan
    3 skala. Selain itu, keputusan "pakai P2 atau tidak" mengubah topologi jalur
    bottom-up (P2->P3->P4->P5 vs P3->P4->P5), bukan cuma satu layer.

    Atribut penting (diset di __init__):
        detect_A, detect_B : referensi ke dua modul Detect di self.model (urutan YAML).
        stride             : stride Detect_A ([4, 8, 16, 32]) — dipakai Ultralytics untuk
                             letterbox/anchor; diambil dari branch A karena itu jalur default.
    """

    def __init__(self, cfg="yolov8-p2-router.yaml", ch=3, nc=None, verbose=True):
        """
        Bangun graf dari YAML, identifikasi dua Detect head, lalu hitung ulang stride
        dan inisialisasi bias untuk MASING-MASING head.

        Args:
            cfg: path/nama YAML arsitektur (harus berisi tepat 2 layer Detect).
            ch: jumlah channel input (3 = RGB).
            nc: jumlah kelas (override nilai di YAML).
            verbose: cetak stride kedua head.

        Efek samping: mengisi self.detect_A, self.detect_B, self.stride, serta
        detect_A.stride / detect_B.stride dan bias awal kedua head.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.detect_layers = [m for m in self.model if isinstance(m, Detect)]
        assert len(self.detect_layers) == 2, (
            f"Expected exactly 2 Detect heads in YAML, found {len(self.detect_layers)}."
        )
        # Urutan sesuai YAML: layer Detect pertama = Detect_A (4 skala), kedua = Detect_B.
        self.detect_A, self.detect_B = self.detect_layers

        # --- Stride & bias init untuk Detect_A ---
        # Kenapa dihitung ulang di sini: DetectionModel.__init__ bawaan hanya menghitung
        # stride untuk model.model[-1] (= Detect_B). Detect_A juga butuh stride yang benar
        # (4 nilai) untuk make_anchors dan bias_init, jadi keduanya diukur dari forward dummy.
        s = 256
        dummy = torch.zeros(1, ch, s, s)
        was_training = self.training

        self.model.eval()          # matikan BN/dropout stat update
        self.detect_A.training = True   # TAPI paksa Detect_A/B return list mentah (bukan decoded tuple)
        self.detect_B.training = True

        with torch.no_grad():
            det_A_out, det_B_out = self._predict_once_dual(dummy)

        # stride = ukuran input / resolusi feature map tiap skala (mis. 256/64 = 4 untuk P2).
        self.detect_A.stride = torch.tensor([s / x.shape[-2] for x in det_A_out])
        self.stride = self.detect_A.stride
        self.detect_A.bias_init()

        self.detect_B.stride = torch.tensor([s / x.shape[-2] for x in det_B_out])
        # bias_init juga sudah dipanggil untuk Detect_B di super().__init__(); memanggilnya
        # lagi aman karena bias_init meng-ASSIGN nilai konstan (idempoten), bukan menambah.
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
        """
        Satu forward pass melalui seluruh graf, menangkap output KEDUA Detect head.

        Isi loop sengaja disalin dari BaseModel._predict_once Ultralytics (routing
        m.f, self.save, profile, visualize, embed) supaya perilakunya identik; satu-
        satunya tambahan adalah penangkapan output Detect_A dan Detect_B secara terpisah.

        Args:
            x: tensor gambar (B, 3, H, W).
            profile, visualize, embed: sama seperti _predict_once bawaan.

        Returns:
            (det_A_out, det_B_out). Bentuknya tergantung mode Detect:
            - training (atau Detect.training=True): list feature map mentah per skala
              (4 untuk A, 3 untuk B) -> dipakai v8DetectionLoss.
            - eval: tuple (decoded, raw) bawaan Detect.
            Jika `embed` diisi, mengembalikan tuple embedding (perilaku bawaan) dan
            BUKAN pasangan (det_A, det_B).
            # Catatan audit: _predict_once di bawah membongkar hasil ini menjadi 2
            # nilai, sehingga jalur `embed` tidak kompatibel (crash bila B != 2).
            # Jalur ini tidak dipakai di pipeline thesis.

        Efek samping: tidak ada pada model; Router di dalam graf menyimpan state-nya
        sendiri (lihat block.py) saat forward berjalan.
        """
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

            # Identifikasi via identitas objek (`is`), bukan indeks layer, supaya tetap
            # benar walau urutan/jumlah layer di YAML berubah.
            if detect_A is not None and m is detect_A:
                det_A_out = x
            elif detect_B is not None and m is detect_B:
                det_B_out = x
            elif isinstance(m, Detect) and detect_A is None and detect_B is None:
                # fallback SEMENTARA saat super().__init__() masih berjalan
                # (detect_A/detect_B belum di-assign) — anggap ini "single detect"
                # Karena Detect_B adalah layer Detect terakhir, nilai yang tersisa di
                # akhir loop = output Detect_B; itulah yang dipakai DetectionModel bawaan
                # untuk mengukur stride model.model[-1] (= Detect_B, 3 skala).
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

        Konsekuensi penting: DetectionValidator bawaan (dipakai trainer tiap epoch)
        mengevaluasi BRANCH A saja. Evaluasi Branch B dilakukan dengan menimpa method
        ini sementara (DualBranchValidatorB di script training); evaluasi sistem
        ter-route (gate memilih per gambar) dilakukan oleh skrip evaluasi terpisah.
        """
        det_A_out, _ = self._predict_once_dual(x, profile, visualize, embed)
        return det_A_out

    def forward(self, x, *args, **kwargs):
        """
        Titik masuk model.

        - x berupa dict (batch dari trainer) -> hitung loss dual-branch via self.loss().
        - x berupa tensor -> jalur standar Ultralytics (predict/val/export), yang
          berujung di _predict_once -> output Branch A.
        """
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

        Args:
            batch: dict batch Ultralytics (img, batch_idx, cls, bboxes, ...).
            preds: diabaikan (lihat di atas).

        Returns:
            (total_loss, loss_items) dari DualBranchDetectionLoss:
            total_loss skalar untuk backward; loss_items 8 komponen untuk logging
            (box_A, cls_A, dfl_A, router, proxy, box_B, cls_B, dfl_B).

        Efek samping: membuat self.criterion secara lazy pada panggilan pertama
        (model.args harus sudah terpasang, biasanya oleh trainer).
        """
        if not hasattr(self, "criterion") or self.criterion is None:
            self.criterion = self.init_criterion()

        img = batch["img"]
        preds_dual = self._predict_once_dual(img)  # SELALU recompute, jangan pakai preds argumen

        return self.criterion(preds_dual, batch)

    def init_criterion(self):
        """Buat criterion dual-branch (dipanggil lazy dari loss())."""
        return DualBranchDetectionLoss(self)

"""
OPSI C — Soft-mixture router objective dengan EMA offset (tanpa freeze, tanpa BCE target)
============================================================================

Menggantikan gate_value_loss lama (BCEWithLogits terhadap sigmoid(diff/temperature))
dengan objective linear ala DynamicDet:

    p_i = sigmoid(gate_logit_i)
    L_router_i = p_i * (loss_A_i.detach() - offset/2) + (1 - p_i) * (loss_B_i.detach() + offset/2)
    (p_i = P(P2 aktif / Branch A). [AUDIT-FIX #M1: docstring lama tertukar; kode sudah benar.])
    Implementasi sejak AUDIT-FIX #3: gate_value_loss = mean(p_i * clamp(loss_A_i - loss_B_i - offset)),
    gradien identik dgn rumus di atas selama selisih di dalam clip, tapi terbatas di luar itu.

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

    Komposisi loss total per batch:
        total = loss_A_sum                      (box+cls+dfl A, + router penalty, + proxy)
              + branch_b_weight * loss_B_sum    (box+cls+dfl B)
              + gate_value_weight * gate_value_loss * n_valid   (hanya setelah warmup)

    Pembagian tugas gradien:
        - loss deteksi A/B  -> backbone, neck, C2f-P2, Detect_A/B.
        - router penalty    -> classifier gate (via router.loss_prob).
        - proxy loss        -> cabang proxy router saja (input-nya di-detach).
        - gate_value_loss   -> classifier gate saja (loss_A/B di-detach, fitur gate di-detach).
    """

    def __init__(self, model):
        """
        Siapkan dua v8DetectionLoss yang dikonfigurasi ulang per branch.

        v8DetectionLoss bawaan membaca stride/nc/reg_max dari model.model[-1]
        (= Detect_B). Karena itu kedua instance di-override secara eksplisit dengan
        atribut head-nya masing-masing; tanpa ini loss_A akan memakai stride 3-skala
        dan anchor tidak cocok dengan 4 feature map Branch A.

        Args:
            model: DualBranchDetectionModel (boleh terbungkus DDP; .module di-unwrap).

        Hyperparameter dibaca dari atribut raw_model (diset oleh script/scheduler):
            branch_b_loss_weight (0.7), gate_value_loss_weight, gate_offset_momentum (0.99),
            gate_value_temperature (0.15, hanya jalur BCE), use_bce_gate_loss (False),
            gate_signal_mode ('small' | 'hybrid' | 'total', dibaca di __call__).
        """
        raw_model = model.module if hasattr(model, "module") else model

        # --- Loss A: dapat router penalty (compute_router_loss aktif) ---
        self.loss_A = v8DetectionLoss(model)
        self.loss_A.stride = raw_model.detect_A.stride
        self.loss_A.nc = raw_model.detect_A.nc
        self.loss_A.no = raw_model.detect_A.nc + raw_model.detect_A.reg_max * 4
        self.loss_A.reg_max = raw_model.detect_A.reg_max
        self.loss_A.use_dfl = raw_model.detect_A.reg_max > 1
        self.loss_A.assigner.num_classes = raw_model.detect_A.nc
        # Flag ini dibaca v8DetectionLoss: True -> hitung router penalty (loss[3]) dan
        # proxy supervision (loss[4]). Hanya satu instance yang boleh True, karena
        # keduanya bergantung pada router yang sama (bukan per-branch).
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

        # Bobot Branch B < 1: Branch A (P2) dianggap jalur kualitas utama; Branch B
        # tetap dilatih penuh tapi kontribusinya ke gradien bersama (backbone/neck)
        # sedikit lebih kecil.
        # TODO: verifikasi -- apakah nilai 0.7 dipilih dari eksperimen/ablation tertentu
        # atau heuristik? (Catatan: asimetri ini juga tercatat sebagai salah satu sumber
        # noise sinyal loss_A vs loss_B.)
        self.branch_b_weight = getattr(raw_model, "branch_b_loss_weight", 0.7)

        # --- Hyperparameter gate routing (OPSI C) ---
        # Catatan: atribut ini TIDAK dipakai di __call__; bobot aktual dibaca ulang tiap
        # step dari raw_model.gate_value_loss_weight (supaya scheduler bisa mengubahnya).
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
        # Catatan audit: kelas ini bukan nn.Module, jadi tensor ini atribut Python biasa
        # (BUKAN buffer terdaftar) -> tidak ikut tersimpan di checkpoint; saat resume
        # training, offset mulai lagi dari 0 dan diinisialisasi ulang dari batch pertama.
        self._offset_initialized = False
        self.running_diff_mean = torch.tensor(0.0)

        # Logging state
        self._last_gate_value_loss = torch.tensor(0.0)
        self._last_target_gate_mean = torch.tensor(0.0)  # dipertahankan utk kompatibilitas logging lama
        self._last_offset = torch.tensor(0.0)
        self._last_mean_p = torch.tensor(0.0)

    def _find_router(self):
        """
        Cari modul router di dalam model (sekali saja, lalu di-cache).

        Pencarian berbasis substring nama kelas ('DifficultyAwareRouter') supaya
        kompatibel dengan semua varian router (biasa / LightWeight / UltraLightWeight).

        Returns:
            Modul router, atau None jika model tidak punya router.
        Efek samping: mengisi self._router_cache.
        """
        if self._router_cache is not None:
            return self._router_cache
        for m in self._raw_model.modules():
            if 'DifficultyAwareRouter' in m.__class__.__name__:
                self._router_cache = m
                return m
        return None

    def _update_offset(self, batch_diff_mean: torch.Tensor):
        """
        VERSI ROBUST: clip nilai sebelum EMA update, mencegah outlier tunggal
        merusak offset secara permanen.

        Args:
            batch_diff_mean: skalar statistik (loss_A - loss_B) batch ini — dari
                __call__ yang dikirim adalah MEDIAN sampel valid (bukan mean), karena
                median tahan terhadap satu-dua sampel ekstrem.

        Efek samping: mengubah self.running_diff_mean (dipindah ke device yang sama
        dengan input bila perlu) dan self._offset_initialized.
        Batas clip dibaca dari atribut opsional self.offset_clip_range (default 1.0).
        """
        if self.running_diff_mean.device != batch_diff_mean.device:
            self.running_diff_mean = self.running_diff_mean.to(batch_diff_mean.device)

        # --- LAPIS PERTAHANAN 1: clip nilai ekstrem sebelum masuk EMA ---
        # Batas ini SEHARUSNYA jauh lebih besar dari fluktuasi normal offset
        # Anda (yang biasanya di rentang -0.05 s.d 0.1 dari data historis),
        # tapi cukup ketat utk memblokir lonjakan seperti -5.53 kemarin.
        clip_range = getattr(self, 'offset_clip_range', 1.0)
        batch_diff_mean_clipped = torch.clamp(batch_diff_mean, -clip_range, clip_range)

        # Inisialisasi langsung dari batch pertama (bukan dari 0), supaya EMA tidak
        # butuh ratusan step untuk "naik" dari nol ke level sistematis yang sebenarnya.
        if not self._offset_initialized:
            self.running_diff_mean = batch_diff_mean_clipped.clone()
            self._offset_initialized = True
        else:
            m = self.offset_momentum
            self.running_diff_mean = m * self.running_diff_mean + (1.0 - m) * batch_diff_mean_clipped

        # --- LAPIS PERTAHANAN 2 (tambahan): safety net di level offset itu sendiri ---
        # Kalau entah bagaimana offset TETAP keluar rentang wajar (mis. akibat
        # beberapa batch beruntun ekstrem di arah yang sama, bukan cuma 1 outlier),
        # clamp juga hasil akhirnya sbg pengaman terakhir sebelum dipakai di loss.
        self.running_diff_mean = torch.clamp(self.running_diff_mean, -clip_range, clip_range)

    def __call__(self, preds, batch):
        """
        Hitung loss total dual-branch untuk satu batch.

        Args:
            preds: tuple (det_A, det_B) dari DualBranchDetectionModel._predict_once_dual.
            batch: dict batch Ultralytics.

        Returns:
            total_loss: skalar untuk backward (sudah dikali batch size, konvensi Ultralytics).
            combined_items (detached): 8 komponen untuk logging —
                [box_A, cls_A, dfl_A, router, proxy] + [box_B, cls_B, dfl_B].
                (router & proxy hanya dihitung di loss_A; slot B-nya dibuang.)

        Efek samping (hanya saat training & setelah warmup):
            - memperbarui EMA offset (self.running_diff_mean),
            - menulis statistik logging ke router (last_target_gate_mean,
              last_gate_value_loss, last_gate_value_weight_active, last_gate_offset)
              yang dibaca compute_router_loss untuk CSV pada step BERIKUTNYA
              (compute_router_loss berjalan di dalam self.loss_A, sebelum blok ini).
        """
        det_A, det_B = preds

        # Urutan penting: loss_A dihitung lebih dulu. Di dalamnya compute_router_loss
        # dan proxy supervision ikut berjalan (flag _compute_router_penalty=True).
        loss_A_sum, loss_A_items = self.loss_A(det_A, batch)
        loss_B_sum, loss_B_items = self.loss_B(det_B, batch)

        total_loss = loss_A_sum + self.branch_b_weight * loss_B_sum
        # loss_B_items[:3] = box, cls, dfl saja; slot router/proxy B selalu 0 (guard), jadi dibuang.
        combined_items = torch.cat([loss_A_items, loss_B_items[:3]])

        # Dibaca ulang setiap step (bukan self.gate_value_weight) karena scheduler
        # mengubah raw_model.gate_value_loss_weight tiap epoch (ramp 0 -> 0.5).
        current_gate_value_weight = getattr(self._raw_model, "gate_value_loss_weight", 0.0)

        router = self._find_router()
        is_warmup = getattr(router, "_is_warmup", True) if router is not None else True

        # Gate loss hanya dihitung jika semua prasyarat ada. hasattr(...) melindungi
        # step-step awal ketika router/loss belum pernah menyimpan state per-sampel.
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
            # 'total' : loss per-gambar seluruh anchor (perilaku versi hasil final KITTI).
            # 'small' : hanya anchor fg yang ter-assign ke objek kecil (< 32x32 px),
            #           karena P2 dirancang untuk objek kecil -> sinyal lebih terfokus.
            # 'hybrid': campuran berbobot small/medium/large.
            gate_signal_mode = getattr(self._raw_model, 'gate_signal_mode', 'small')

            if gate_signal_mode == 'small':
                per_sample_loss_A = self.loss_A._last_per_sample_loss_small.detach()
                per_sample_loss_B = self.loss_B._last_per_sample_loss_small.detach()
                # Gambar hanya dipakai jika KEDUA branch punya anchor fg objek kecil;
                # tanpa itu loss_small salah satu branch = 0 dan selisihnya tidak bermakna.
                # Catatan audit: ini juga membuang gambar yang objek kecilnya hanya
                # tertangkap anchor P2 (Branch B tidak dapat anchor) -> bias seleksi.
                valid_A = self.loss_A._last_small_obj_count > 0
                valid_B = self.loss_B._last_small_obj_count > 0
                valid_mask = valid_A & valid_B

            elif gate_signal_mode == 'hybrid':
                # TODO: verifikasi -- dasar pemilihan bobot 0.7/0.2/0.1 untuk strata
                # small/medium/large (heuristik "P2 paling penting untuk objek kecil",
                # atau hasil tuning)?
                raw_w_s, raw_w_m, raw_w_l = 0.7, 0.2, 0.1
                # Ganti bobot tetap dengan normalisasi dinamis per-gambar
                # (strata yang tidak ada di gambar ini tidak ikut, dan bobot sisanya
                # dinormalisasi ulang supaya skala loss antar-gambar sebanding).
                has_small_a = (self.loss_A._last_small_obj_count > 0).float()
                has_medium_a = (self.loss_A._last_medium_obj_count > 0).float()  # perlu tambah tracking count_medium juga
                has_large_a = (self.loss_A._last_large_obj_count > 0).float()

                active_w_sum_a = raw_w_s * has_small_a + raw_w_m * has_medium_a + raw_w_l * has_large_a
                # clamp hanya relevan untuk gambar tanpa objek sama sekali (pembilang juga 0),
                # jadi tidak ada amplifikasi: bobot aktif minimum yang mungkin adalah 0.1.
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

            # gate_logit SENGAJA tidak di-detach: inilah satu-satunya jalur gradien
            # gate_value_loss. Gradien berhenti di router (fitur input gate sudah
            # di-detach di block.py), jadi backbone/neck tidak terpengaruh.
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
                # (dipakai nilai lama supaya sampel batch ini tidak ikut menggeser
                # acuan yang dipakai untuk menilai dirinya sendiri).
                offset = self.running_diff_mean.clone()
                self._update_offset((loss_A_v - loss_B_v).median().detach())

                if self.use_bce_gate_loss:
                    # jalur lama (BCE) — kini juga menghormati valid_mask
                    # target_gate > 0.5 bila loss_B > loss_A (Branch A lebih baik -> P2 aktif).
                    diff = (loss_B_v - loss_A_v)
                    target_gate = torch.sigmoid(diff / self.gate_value_temperature)
                    gate_value_loss = F.binary_cross_entropy_with_logits(gate_logit_v, target_gate)
                    mean_p = torch.sigmoid(gate_logit_v).mean().detach()
                else:
                    # OPSI C: soft-mixture dengan EMA offset
                    # [AUDIT-FIX 2026-10 #3] Versi lama:
                    #   (1-p)*(loss_B + off/2) + p*(loss_A - off/2)
                    # -> dL/dp = loss_A - loss_B - off, TIDAK terbatas (clip hanya di offset).
                    # Sekarang advantage di-clip -> gradien sama di dalam clip, terbatas di luar.
                    # Catatan: NILAI gate_value_loss (kolom GV_Loss) berubah makna -> tidak
                    # bisa dibandingkan langsung dgn run sebelum fix ini.
                    # Arah gradien: adv < 0 (A lebih baik dari biasanya) -> minimisasi p*adv
                    # mendorong p naik (P2 aktif); adv > 0 -> p turun.
                    clip_adv = getattr(self, 'gate_adv_clip', getattr(self, 'offset_clip_range', 1.0))
                    p = torch.sigmoid(gate_logit_v)
                    adv = (loss_A_v - loss_B_v - offset).clamp(-clip_adv, clip_adv).detach()
                    gate_value_loss = (p * adv).mean()
                    mean_p = p.mean().detach()

            # scaling pakai jumlah sampel VALID (0 kalau tidak ada -> gate loss mati)
            # Dikali jumlah sampel agar setara konvensi Ultralytics (loss deteksi juga
            # dikali batch_size); mean * n_valid = jumlah kontribusi per sampel valid.
            batch_size = combined_items.new_tensor(float(n_valid))
            total_loss = total_loss + current_gate_value_weight * gate_value_loss * batch_size

            # Simpan ke ROUTER (bukan cuma self) supaya compute_router_loss di loss.py
            # bisa menuliskannya ke CSV router_buffer.
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
            # Gate loss tidak aktif (eval / warmup / bobot 0): reset statistik logging ke 0
            # agar CSV tidak menampilkan nilai basi dari step sebelumnya.
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
