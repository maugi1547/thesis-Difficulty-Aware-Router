# Berkas A6 (split KITTI per rekaman)

| Berkas | Isi | Sumber | sha256 |
|---|---|---|---|
| `chen_train.txt` | 3.712 ID citra latih split Chen dkk. (3DOP, NeurIPS 2015) | `kitti/image_sets/train.txt`, github.com/charlesq34/frustum-pointnets, commit 2ffdd345e1fce4775ecb508d207e0ad465bcca80 | b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb |
| `chen_val.txt` | 3.769 ID citra validasi split Chen (= data uji A6) | `kitti/image_sets/val.txt`, repo dan commit sama | 657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86 |
| `kitti_peta_rekaman.csv` | ID citra -> tanggal, rekaman, frame mentah, ukuran | dibuat dari `train_mapping.txt` + `train_rand.txt` devkit objek KITTI resmi (verifikasi B3; sha256 devkit ca9a8052…196a54 dan 0976b8b4…f305f) | 52007b6b47bdc80e74745ff4c2be5590e9fbb2cc5b36b589902465f9ca920665 (LF; salinan Windows berakhiran CRLF: 09e3de1f…3d9efe) |
| `kitti_label_stat.csv` | jumlah objek dan w×h ternormalisasi per citra | label `kitti_yolo_2class.zip` (dataset 2 kelas yang dipakai run utama) | — |
| `kitti_split_rekaman.csv` / `.json` | split A6 terkunci + ringkasan + pemilihan val' | `tools/a6_split_rekaman.py generate` (seed 20261112, 20.000 undian), 12 Nov 2026 | — |

Catatan:
- Paper Chen dkk. hanya menyatakan pembagian separuh-separuh tanpa berbagi rekaman; angka 3.712/3.769 dan daftar ID
  berasal dari daftar yang beredar (di sini: Frustum-PointNets). Irisan rekaman Chen-train vs Chen-val menurut peta devkit = 0.
- Data uji A6 memuat tanggal 2011_10_03 (1 rekaman, 296 citra) yang tidak ada di data latih, dan proporsi citra berobjek
  kecil sedikit lebih tinggi (0,581 vs 0,524 di latih).
