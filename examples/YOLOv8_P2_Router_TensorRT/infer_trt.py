"""
Skrip Inferensi TensorRT dengan Pascapemrosesan YOLOv8 (P2-P5)
Lokasi: src/inference/infer_trt.py
"""

import os
import cv2
import time
import torch
import numpy as np
import tensorrt as trt
from ultralytics.utils.ops import non_max_suppression

# =============================================================================
# 1. KONFIGURASI PATH
# =============================================================================
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENGINE_PATH = os.path.join(BASE_DIR, "weights/router_ifconditional.engine")
IMG_DIR     = os.path.join(BASE_DIR, "data/kitti-2class/images/test")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_trt")

# Daftar Kelas (Sesuaikan dengan dataset kitti-2class Anda)
CLASSES = ["Car", "Pedestrian"]
COLORS  = [(0, 255, 0), (255, 0, 0)] # Hijau untuk mobil, Biru untuk pejalan kaki

# =============================================================================
# 2. FUNGSI UTILITAS PASCAPEMROSESAN (GPU)
# =============================================================================
def generate_anchors_and_strides(image_size=(640, 640), strides=[4, 8, 16, 32], device="cuda"):
    """
    Membuat anchor points dan stride tensor untuk 4 skala (P2, P3, P4, P5).
    Total anchors untuk 640x640 = 25600 + 6400 + 1600 + 400 = 34000.
    """
    anchor_points = []
    stride_tensor = []
    img_h, img_w = image_size

    for s in strides:
        h, w = img_h // s, img_w // s
        # Gunakan indexing='ij' agar sesuai standar YOLOv8
        sy, sx = torch.meshgrid(torch.arange(h, device=device), 
                                torch.arange(w, device=device), indexing='ij')
        # Format anchor (x, y) ditambah 0.5 agar jatuh tepat di tengah grid
        anchors = torch.stack((sx, sy), dim=-1).view(-1, 2) + 0.5
        anchor_points.append(anchors)
        stride_tensor.append(torch.full((h * w, 1), s, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Mengubah tensor jarak (dl, dt, dr, db) menjadi koordinat kotak (x, y, w, h).
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh
    return torch.cat((x1y1, x2y2), dim)    # xyxy

# =============================================================================
# 3. KELAS INFERENSI TENSORRT
# =============================================================================
class TRTInferencer:
    def __init__(self, engine_path):
        print(f"[INFO] Memuat TensorRT Engine dari: {engine_path}")
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream  = torch.cuda.current_stream().cuda_stream
        
        # Alokasi memori tensor di GPU (menghindari transfer H2D/D2H terus-menerus)
        self.d_in = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device="cuda")
        self.d_dt = torch.zeros((1, 6, 34000), dtype=torch.float32, device="cuda")
        
        self.context.set_tensor_address("images", self.d_in.data_ptr())
        self.context.set_tensor_address("detections", self.d_dt.data_ptr())

        # Siapkan konstan Anchor & Stride sekali saja saat inisialisasi
        self.anchors, self.strides = generate_anchors_and_strides()

    def infer(self, img_tensor):
        """Mengeksekusi model."""
        self.d_in.copy_(img_tensor)
        self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        return self.d_dt

    def post_process(self, preds, conf_thres=0.25, iou_thres=0.45):
        """
        Mengolah output mentah TRT [1, 6, 34000] menjadi deteksi bersih.
        """
        # Transpose untuk kemudahan indexing: [1, 34000, 6]
        preds_trans = preds.transpose(1, 2)
        
        box_preds = preds_trans[..., :4] # [1, 34000, 4] -> Jarak dl, dt, dr, db
        cls_preds = preds_trans[..., 4:] # [1, 34000, 2] -> Probabilitas kelas

        # 1. Dekode DFL distance ke bounding box aktual (cx, cy, w, h)
        box_decoded = dist2bbox(box_preds, self.anchors, xywh=True)
        box_decoded = box_decoded * self.strides # Kalikan dengan stride agar sesuai ukuran gambar asli
        
        # 2. Gabungkan kembali: [1, 34000, 6] -> Transpose kembali ke [1, 6, 34000] untuk fungsi NMS
        preds_ready = torch.cat((box_decoded, cls_preds), dim=-1).transpose(1, 2)
        
        # 3. Jalankan Non-Maximum Suppression (NMS)
        results = non_max_suppression(preds_ready, conf_thres=conf_thres, iou_thres=iou_thres, nc=2)
        return results[0] # Mengembalikan tensor deteksi untuk gambar pertama (batch index 0)

# =============================================================================
# 4. LOOP UTAMA
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inferencer = TRTInferencer(ENGINE_PATH)
    
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg'))]
    
    print(f"\n[INFO] Menjalankan inferensi pada {len(image_paths)} gambar...")
    
    for path in image_paths[:50]: # Coba jalankan untuk 50 gambar pertama
        # 1. Pre-processing
        img_bgr = cv2.imread(path)
        orig_h, orig_w = img_bgr.shape[:2]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0
        
        # 2. Model Execution (TensorRT)
        t1 = time.time()
        raw_preds = inferencer.infer(img_tensor)
        
        # 3. Post-processing
        detections = inferencer.post_process(raw_preds, conf_thres=0.3, iou_thres=0.45)
        t2 = time.time()
        
        # 4. Visualisasi (Gambar bounding box)
        if detections is not None and len(detections):
            # Tarik ke CPU untuk digambar menggunakan OpenCV
            detections = detections.cpu().numpy()
            
            # Skala koordinat box kembali dari 640x640 ke ukuran gambar asli
            r_w, r_h = orig_w / 640.0, orig_h / 640.0
            
            for *xyxy, conf, cls_id in detections:
                x1, y1 = int(xyxy[0] * r_w), int(xyxy[1] * r_h)
                x2, y2 = int(xyxy[2] * r_w), int(xyxy[3] * r_h)
                class_idx = int(cls_id)
                label = f"{CLASSES[class_idx]} {conf:.2f}"
                
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLORS[class_idx], 2)
                cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_idx], 2)

        # Simpan hasil
        filename = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, img_bgr)
        print(f"Selesai: {filename} | Waktu Total: {(t2-t1)*1000:.1f} ms | Deteksi: {len(detections)}")

if __name__ == "__main__":
    main()
