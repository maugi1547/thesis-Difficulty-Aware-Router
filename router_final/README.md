# router_final — arsitektur final Difficulty-Aware Router untuk YOLOv8n-P2

Kode bersih untuk sistem final tesis. Folder ini berdiri sendiri dan berjalan di atas **Ultralytics stok 8.3.240
(PyPI)**. Folder ini **tidak** memakai fork `ultralytics/` di root repo, karena fork itu masih memuat router lama.

## Sistem

```
gambar ─► Stage 1: trunk (layer 0–15) ─► p3_neck, p2_backbone, p4_neck, p5_sppf
                     │
                     └─► gate post-hoc: [GAP‖GMP] 4 peta fitur (960) · w + b = skor
                                         │
              B2: thr = kuantil (1 − ρ) dari 200 skor sebelumnya (ρ = 0,5)
                                         │
                     skor > thr ? ──ya──► Branch A: expert P2 (16–18) + bottom-up (19–27) + Detect_A (P2–P5)
                                └─tidak─► Branch B: bottom-up (28–33) + Detect_B (P3–P5)
deployment A3: satu engine TensorRT (ONNX If); thr = input engine, skor = output engine
```

| Tahap | Kode | Isi |
|---|---|---|
| 1. Detektor | `configs/yolov8n-dualbranch-p2.yaml`, `rfinal/model.py`, `rfinal/loss.py`, `rfinal/train.py` | Dual-branch, hanya modul stok. Loss = L_A + 0,7·L_B (box/cls/dfl stok per head). |
| 2. Gate post-hoc | `rfinal/gate.py` | Detektor beku. Target 1[Δ soft_small > 0] pada train, StandardScaler + LogReg C = 0,001, lalu dilipat menjadi satu Gemm. |
| 3. Anggaran B2 | `rfinal/budget.py` | Threshold kuantil bergulir; threshold awal = kuantil skor val. |
| 4. Deployment A3 | `rfinal/export.py`, `rfinal/trt.py` | ONNX If FP16 (gate FP32) lalu TensorRT. |
| Evaluasi | `rfinal/evaluate.py`, `rfinal/data.py` | pycocotools resolusi asli: A paksa, B paksa, router, acak dengan aktivasi sama. |

### Komponen versi lama yang dibuang

| Dibuang | Alasan |
|---|---|
| `UltraLightWeightDifficultyAwareRouter`: `squeeze_p3/p2`, `classifier`, `tau_param`, Gumbel-softmax | Gate terlatih tidak dipakai sistem final; routing dilakukan gate post-hoc. |
| Cabang proxy (`proxy_stem/cls/reg_dist`) + `compute_proxy_supervision_loss` | Hanya memodulasi sparsity penalty. |
| `gate_value_loss` (BCE / Opsi C + offset EMA), `compute_router_loss` (sparsity, `difficulty_weight`) | Kendali aktivasi dipindah ke pasca-training (B2). |
| Kalibrasi `calib_T/calib_b`, `compute_gate_only`, 3 engine terpisah | Diganti skor + threshold B2 di engine A3. |
| `gate_signal_mode`, `normalize_bg_anchors`, loss per strata ukuran, `router_buffer`/CSV router | Hanya melayani sinyal gate lama. |

## Bukti kesetaraan dengan versi lama

Semua uji ini dijalankan di CPU dan semuanya lolos. Perintahnya: `pytest tests`, ditambah `RFINAL_M0_REF=… pytest tests/test_m0_equivalence.py`.

1. **Graf final = jaringan deteksi M0.** Bobot M0 seed 42 (`last.pt`) dipetakan lewat `rfinal/compat.py`, yaitu dengan membuang 30 kunci router dan menggeser indeks layer +2. Hasilnya:
   - fitur trunk dan keluaran Branch A/B **identik sampai level bit** dengan model lama (`torch.equal`, 8 gambar test KITTI);
   - item loss box/cls/dfl versi stok **identik sampai level bit** dengan item `[0:3]` loss lama.

   Artinya, komponen yang dibuang memang tidak ikut menghitung deteksi.
2. **Gate dan B2 di-port tanpa perubahan semantik.** Pada bobot M0 dan subset KITTI, nilai berikut identik sampai level bit dengan `eksperimen/tahap4_final.py`:
   - Δ soft_small per gambar,
   - bobot gate (w, b),
   - thr0 dan skor val.

   `tests/test_budget.py` membandingkan B2 dengan salinan apa adanya implementasi Tahap 4.
3. **ONNX A3 = PyTorch.** Untuk ketiga mode dan kedua arah threshold, error relatifnya < 1e-5 (`tests/test_export.py`).

## Reproduksi

Konfigurasi terkunci ada di `configs/kitti.yaml`, dan semua angka diambil dari file ini. Setiap run menulis `provenance.json` yang berisi:

- commit `router_final`,
- versi library, GPU, dan driver,
- sha256 konfigurasi, data, dan bobot.

**Jaminan saat ini:**

- **CPU:** training dengan seed sama, termasuk augmentasi dan 2 worker dataloader, menghasilkan bobot **identik sampai level bit** (`state_hash` sama). Ini sudah diuji.
- **GPU (T4):** belum diuji. Jalankan `scripts/check_reproducibility.py` sebelum training 100 epoch. Skrip ini menjalankan 2 epoch dua kali lalu membandingkan `state_hash.json`.
  - Kalau identik, hasil ulangan dijamin sama asalkan GPU, driver, torch/CUDA, batch, dan jumlah worker sama (semuanya tercatat di provenance).
  - Kalau berbeda, ulangi dengan `--strict-determinism` untuk menemukan operasi non-deterministik.
- **Penjaga lingkungan:** `require_stock_ultralytics()` memeriksa sha256 8 file inti Ultralytics. Penjaga AMP menghentikan training bila `check_amp` diam-diam mematikan AMP, karena resepnya jadi berubah tanpa terlihat.

**Bukan jaminan:** bobot hasil resep ini **berbeda dari M0**. Penyebabnya:

- loss router tidak lagi ikut ke gradient clipping dan GradScaler;
- inisialisasi router tidak lagi mengambil angka acak sebelum layer Branch A/B dibuat.

Karena itu, semua angka Tahap 2–4 harus dihasilkan ulang dari model ini.

## Menjalankan

Jalankan dari folder `router_final/` (bukan root repo):

```bash
pip install ultralytics==8.3.240 onnx onnxruntime pycocotools scikit-learn   # + tensorrt untuk deployment
python -m pytest -q tests                                                    # termasuk cek ultralytics stok

python scripts/check_reproducibility.py --config configs/kitti.yaml --seed 42 --override epochs=2
python scripts/train.py     --config configs/kitti.yaml --seed 42                       # -> RUN = runs_final/kitti_seed42
python scripts/fit_gate.py  --config configs/kitti.yaml --weights RUN/weights/last.pt --out RUN/gate
python scripts/export.py    --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json --out RUN/deploy --trt
python scripts/evaluate.py  --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json \
                            --out RUN/eval_test --backend trt --engine RUN/deploy/a3_fp16.engine
```

Untuk Kaggle, gunakan `notebooks/kaggle_router_final.ipynb`. Notebook ini memuat urutan yang sama, ditambah cek commit yang dipin.

### Memvalidasi kode pada bobot M0 (opsional)

Kalau `fit_gate` dan `evaluate` dijalankan pada bobot M0, hasilnya harus mereproduksi angka PyTorch FP32 Tahap 4.

```bash
# di root repo (fork lama):
python router_final/tools/m0_state_dict.py --ckpt M0/weights/last.pt --out m0_sd.pt
python router_final/tools/m0_reference.py  --ckpt M0/weights/last.pt --data kitti_2class.yaml --out m0_ref.pt
# di router_final/ (ultralytics stok):
python tools/convert_m0.py --sd m0_sd.pt --out m0_final.pt
RFINAL_M0_REF=m0_ref.pt python -m pytest -q tests/test_m0_equivalence.py
```

## Yang tetap di `eksperimen/`

Analisis statistik tesis tetap di `eksperimen/`, yaitu bootstrap berpasangan, H1–H7, kurva Pareto, baseline Vanilla/P2-Static, latensi per komponen, dan heuristik kepadatan. Kode di sana memakai fork lama dan menjadi arsip yang menghasilkan angka versi M0.
