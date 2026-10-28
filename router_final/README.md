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
| Evaluasi final terkunci | `rfinal/suite.py`, `rfinal/cocomix.py`, `rfinal/baselines.py`, `scripts/suite.py` | Port Tahap 4/4B: Vanilla/P2-Static, H1–H7, bootstrap berpasangan, campuran nyata, heuristik tanpa label, kaskade, latensi per komponen dan **ujung-ke-ujung**, bootstrap klaster (VisDrone), gabungan + pooling 3 seed. |
| Profil latensi per segmen (diagnostik) | `rfinal/profile.py`, `scripts/profile_latency.py` | Salin citra / engine / salin keluaran / host untuk Vanilla, P2-Static, A3 rute A/B, rantai dua engine tanpa If, rute B salin 8.400; clock GPU dicatat (notebook sel 7c). |
| Ablasi pasca-training | `rfinal/ablation.py`, `scripts/ablate.py` | G1 target, G3 fitur, G4 classifier, G5 biaya label, B anggaran (dinilai di val) + uji berpasangan bootstrap dengan Holm (`ablasi_uji.csv`); cache `cache_ablasi.pkl` untuk analisis ulang tanpa GPU (`--reuse-cache`). |

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
python scripts/train.py     --config configs/visdrone.yaml --seed 42 --model vanilla    # baseline stok, resep & GPU sama
python scripts/train.py     --config configs/visdrone.yaml --seed 42 --model p2static
python scripts/check_checkpoint.py --config configs/kitti.yaml --run RUN --out RUN/cek_checkpoint   # last vs best, A & B (val)
python scripts/fit_gate.py  --config configs/kitti.yaml --weights RUN/weights/last.pt --out RUN/gate
python scripts/export.py    --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json --out RUN/deploy --trt
python scripts/evaluate.py  --config configs/kitti.yaml --weights RUN/weights/last.pt --gate RUN/gate/gate.json \
                            --out RUN/eval_test --backend trt --engine RUN/deploy/a3_fp16.engine
```

Semua tahap memakai `last.pt`. `best.pt` Ultralytics dipilih dari fitness Branch A saja, sehingga epoch-nya belum tentu terbaik untuk Branch B dan bisa membesarkan jarak A − B. `check_checkpoint.py` hanya melaporkan selisih last − best (A, B, jarak A − B) dengan bootstrap berpasangan di val; pilihan checkpoint tidak berubah.

Untuk Kaggle, gunakan `notebooks/kaggle_router_final.ipynb`. Notebook ini memuat urutan yang sama, ditambah cek commit yang dipin.

Dua hal di notebook yang perlu diketahui:

- **Tiga sesi untuk VisDrone.** Sesi 1 seed 42 di T4 terukur: training 6,2 jam, cek checkpoint + gate + ekspor ±2,7 jam, evaluasi TRT ±2 jam; evaluasi PyTorch FP32 (±2 jam lagi) terpotong batas 12 jam. Scene VisDrone padat (hingga 30.000 kandidat NMS per gambar pada conf 0,001), sehingga NMS dan COCOeval jauh lebih lambat daripada KITTI. `SESSION = 1` = uji reproduksi, training, cek checkpoint, gate, ekspor, evaluasi TRT, dan arsip (evaluasi PyTorch FP32 terpisah dilewati: suite sudah memeriksa FP32 pada seluruh test, `fp32_check`). `SESSION = 2` (dengan `RESUME_FROM`) = cek NMS cepat + suite; `SESSION = 3` = sensitivitas `best.pt` + ablasi. `SESSION = "baselines"` hanya melatih Vanilla + P2-Static. `SESSION = None` = kendali manual lewat `STEPS`.
- **NMS cepat (`fast_nms_k`, hanya `configs/visdrone.yaml`).** `rfinal.data.nms_one` menjalankan NMS Ultralytics pada K kandidat berskor tertinggi dan memakai hasilnya hanya bila ada ≥ `max_det` kotak selamat dengan skor kotak ke-`max_det` lebih besar dari skor kandidat ke-K. NMS greedy hanya menekan kotak dengan kotak berskor lebih tinggi, jadi dalam kondisi itu keluarannya identik dengan NMS penuh; selain itu NMS penuh dijalankan. Celah satu-satunya adalah skor yang persis sama (tie) di antara kotak yang bertumpuk. `scripts/check_fast_nms.py` mengukurnya di val (FP32 dan FP16) dan sel 6b menghentikan notebook bila ada satu deteksi pun yang berbeda. Tanpa kunci ini perilaku NMS sama persis dengan sebelumnya (KITTI tidak berubah).
- **Fit ulang gate di suite memakai `max_det` gate.** Suite VisDrone memakai NMS `max_det` 500, sedangkan gate di-fit dengan 300 (`gate.nms`). Fit ulang gate (pemeriksaan silang terhadap `gate.json`, toleransi 1e-4) kini memakai 300 deteksi teratas dari keluaran yang sama, yang identik dengan NMS ber-`max_det` 300. Sebelumnya target soft_small ikut dipengaruhi deteksi peringkat 301–500 dan asersi itu akan gagal. Di KITTI keduanya 300, jadi tidak ada perubahan.
- **Log training ke file.** Progress bar Ultralytics tidak lagi masuk ke output notebook (membuat tahap simpan versi Kaggle sangat lama). Stdout/stderr masuk ke `/kaggle/working/logs/<nama>.log.gz`; notebook mencetak epoch dan mAP50 dari `results.csv` tiap 10 menit serta 25 baris terakhir log. Perintah training yang gagal menghentikan notebook. Hanya uji reproduksi yang tidak fatal (hasil BERBEDA dicetak sebagai peringatan).

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

## Evaluasi final, latensi ujung-ke-ujung, dan ablasi (tanpa fork lama)

Semua analisis tesis kini berjalan di lingkungan stok yang sama dengan training:

```bash
python scripts/suite.py run --config configs/kitti.yaml --seed 42 --weights RUN/weights/last.pt --gate RUN/gate/gate.json \
    --vanilla VAN/weights/last.pt --p2static P2S/weights/last.pt --out /kaggle/working/suite --zip /kaggle/working
python scripts/ablate.py   --config configs/kitti.yaml --weights RUN/weights/last.pt --out RUN/ablasi
python scripts/suite.py merge --out /kaggle/working/suite_gabung --glob "/kaggle/input/**/suite_*" \
    --models kitti_seed42 kitti_seed7 kitti_seed2026 visdrone_seed42
```

Protokol = Tahap 4/4B (konstanta di blok `analysis:` config), dengan satu tambahan yang dikunci sebelum run ulang:
**H6 utama** = router (B2 efisien) − campuran acak Vanilla/P2-Static dengan fraksi campuran dari latensi
ujung-ke-ujung. B2 efisien (`rfinal/budget.py`, jendela terurut) memberi threshold **bit-identik** dengan
`np.quantile` (`tests/test_budget.py`); H6 Tahap 4 (komposit, B2 `np.quantile`) tetap dilaporkan.
`suite.py` memakai `gate.json` dari `fit_gate.py` dan mem-fit ulang gate dari cache train hanya sebagai cek silang.

**Bukti kesetaraan port** (bukan klaim):

1. `tests/test_cocomix.py`: CocoMix, UnionBoot, FastUnionBoot == COCOeval langsung (data sintetis).
2. `tests/test_suite_replay.py` (`RFINAL_SUITE_REF=<intermediate Tahap 4>`): `analyze` mereproduksi estimasi titik,
   sampel bootstrap per resample, keputusan B2, H3 dan B2 per urutan Tahap 4 VisDrone s42 secara eksak.
3. Pipeline penuh pada bobot M0 seed 42 (`tools/convert_m0.py`), data mini, backend ONNX Runtime:
   `tools/compare_suite_runs.py OLD NEW` -> cache train/val/test, deteksi Vanilla/P2-Static/A/B, skor gate &
   heuristik, gate kaskade, keputusan job, GFLOPs, dan semua uji bootstrap yang tidak bergantung latensi **identik**
   dengan `eksperimen/tahap4b.py` di fork lama.
4. Bagian TensorRT (build, latensi, pass ujung-ke-ujung) adalah salinan kode Tahap 4B yang sudah berjalan di T4,
   tetapi baru teruji di lingkungan stok pada run Kaggle pertama (sesi 0: `--override n_boot=50` cukup).

## Yang tetap di `eksperimen/`

`eksperimen/` adalah arsip yang menghasilkan angka versi M0 (fork lama). Sistem final tidak membutuhkannya lagi; kodenya hanya dipakai sebagai acuan uji kesetaraan di atas.
