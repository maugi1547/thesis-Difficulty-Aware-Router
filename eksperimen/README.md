# Eksperimen tesis: dual-branch YOLOv8n-P2 + gate post-hoc

Folder ini memuat seluruh kode eksperimen yang menghasilkan angka di tesis. Kode detektor dan loss ada di paket `ultralytics/` repo ini. Semua run Kaggle meng-clone commit `5416543`, lalu menerapkan `patch_local.py` (2 patch: tanda `+conf` pada skor kesulitan dan pencatatan keputusan gate per gambar). Hasil `git diff` setelah patch dicatat di setiap `provenance.json` (sha256 `ff1b15d…`). Isi `block.py` dan `loss.py` hasil patch itu sama dengan commit `26b2715`.

## Sistem final

- Detektor dual-branch M0 (`ultralytics/cfg/models/v8/yolov8n-p2-router.yaml`), dilatih bersama 100 epoch, lalu dibekukan (`last.pt`).
- Gate post-hoc: GAP‖GMP dari p3, p2, p4, p5 (960 fitur) → StandardScaler + LogisticRegression C = 0,001, target `soft_small` biner, dilipat menjadi satu Gemm.
- Kendali anggaran B2: threshold = kuantil (1 − ρ) dari 200 skor sebelumnya, ρ = 0,5 (< 20 riwayat → kuantil val).
- Deployment A3: satu engine TensorRT FP16 dengan ONNX `If`; threshold menjadi input engine, skor menjadi output engine.

## Tahap, kode, dan arsip hasil

| Tahap | Kode | Arsip hasil (folder `Hasil Eksperimen/File zip`) | Isi |
|---|---|---|---|
| Dataset | `akses-dataset-kitti.ipynb`, `akses-dataset-visdrone.ipynb` | — | Konversi ke 2 kelas dan split |
| 0 | `tahap0_kaggle_smoke_test.ipynb`, `tahap0_smoke_test.py` | `tahap0.zip` | Uji asap pipeline |
| 1 | `tahap1_train_m0.ipynb`, `visdrone_train_m0_colab.ipynb`, `script_training.py` (repo root), `router_train_utils.py` | `archive_M0_lam002_seed{42,7,2026}.zip`, `archive_M0_lam002_visdrone_seed42.zip` | Training M0 |
| 2 | `tahap2_eval.ipynb` / `.py`, `operating_point_sweep.py` | `tahap2_seed{42,7,2026}.zip` | Router terlatih vs acak, oracle selisih loss |
| 2b | `tahap2b_plafon.ipynb` / `.py` | `tahap2b.zip` | Plafon oracle, seleksi gate post-hoc di val |
| 2c | `tahap2c_gate.ipynb` / `.py`, `tahap2c_visdrone.ipynb` | `tahap2c.zip`, `tahap2c_visdrone.zip` | Gate post-hoc pertama, TensorRT FP32 |
| 3a | `tahap3a.ipynb` / `.py` | `tahap3a.zip`, `tahap3a_ringkasan.zip` | D1 (fitur z), A1 (profil overhead), B (anggaran) |
| 3a2 | `tahap3a2.ipynb` | `tahap3a2_kitti_s42*.zip` | A2 CUDA Graph, A3 engine `If`, A4 FP16 |
| 3c | `tahap3c1.ipynb`, `tahap3m0s.ipynb`, `tahap3c23.ipynb` | `C1_…zip`, `archive_M0s_…zip`, `archive_C2/C3_…zip` | C1, M0-sempit, C2, C3 |
| 3d | `tahap3d2.ipynb`, `tahap3d4.ipynb` | `tahap3d2*.zip`, `tahap3d4_kitti_s42.zip` | D2/D2b kurva label, D4 fitur kesulitan |
| 4 | `tahap4_final.ipynb`, `tahap4_final.py` | `tahap4.zip`, `tahap4_ringkasan.zip` | Evaluasi final terkunci (H1–H7) |
| 4B | `tahap4b_sensitivitas.ipynb`, `tahap4b.py` (dibangkitkan `buat_tahap4b.py`), uji `test_tahap4b_e2e.py` | `tahap4b_ringkasan_{best,last}.zip` | Sensitivitas pasca-hasil: latensi ujung-ke-ujung B2 + A3, baseline `last.pt` |
| Baseline | `verifikasi_baseline.ipynb` | — | Verifikasi loss baseline dan identitas dataset |
| Versi pertama | `versi_pertama/uji-akurasi-latensi-model-dualbranch*.ipynb` | — | Evaluasi router versi pertama (log mentah tidak lengkap) |

Notebook dijalankan di Kaggle (Tesla T4), kecuali training M0 VisDrone (Google Colab, A100). Token GitHub dibaca dari Kaggle Secrets dan tidak tersimpan di notebook.
