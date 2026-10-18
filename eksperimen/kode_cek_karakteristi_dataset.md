import os
import glob
from tqdm import tqdm

def analyze_dataset_difficulty(dataset_name, label_folder, imgsz=640):
    """
    Menganalisis tingkat kesulitan dataset secara komprehensif untuk Tesis S2.
    Menggunakan definisi MS COCO:
    - Small:  Area < 32^2 (1024 piksel)
    - Medium: Area 32^2 s.d 96^2 (9216 piksel)
    - Large:  Area > 96^2
    """

    txt_files = glob.glob(os.path.join(label_folder, "*.txt"))

    if len(txt_files) == 0:
        print(f"❌ Error: Tidak ada file .txt ditemukan di {label_folder}!")
        return

    total_images = len(txt_files)
    
    # Statistik Frame (Gambar)
    easy_images = 0  
    hard_images = 0  
    empty_images = 0 
    
    # Statistik Populasi Objek (Bounding Box)
    total_objects = 0
    small_objects = 0
    medium_objects = 0
    large_objects = 0

    SMALL_THRESHOLD = 32 * 32
    MEDIUM_THRESHOLD = 96 * 96

    print(f"\n--- Mengaudit Dataset {dataset_name} ({total_images} file label) ---")
    print(f"Asumsi resolusi input model: {imgsz}x{imgsz}")

    for txt_file in tqdm(txt_files, desc="Menganalisis Label"):
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_images += 1
            easy_images += 1
            continue

        has_small_object = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            total_objects += 1

            # Format YOLO: class x_center y_center width height
            w = float(parts[3])
            h = float(parts[4])

            # Konversi ke piksel real (berdasarkan input model)
            area_px = (w * imgsz) * (h * imgsz)

            # Kategorisasi Objek
            if area_px < SMALL_THRESHOLD:
                small_objects += 1
                has_small_object = True
            elif area_px < MEDIUM_THRESHOLD:
                medium_objects += 1
            else:
                large_objects += 1

        # Kategorisasi Frame
        if has_small_object:
            hard_images += 1
        else:
            easy_images += 1

    # --- PERHITUNGAN STATISTIK ---
    easy_pct = (easy_images / total_images) * 100
    hard_pct = (hard_images / total_images) * 100
    
    small_obj_pct = (small_objects / total_objects) * 100 if total_objects > 0 else 0
    med_obj_pct = (medium_objects / total_objects) * 100 if total_objects > 0 else 0
    large_obj_pct = (large_objects / total_objects) * 100 if total_objects > 0 else 0
    
    avg_obj_per_image = total_objects / total_images

    # --- TAMPILKAN HASIL ---
    print(f"\n==================================================")
    print(f" 📊 LAPORAN AUDIT KESULITAN: {dataset_name.upper()}")
    print(f"==================================================")
    print(f"Total Citra Dievaluasi : {total_images} frames")
    print(f"Kepadatan Rata-rata    : {avg_obj_per_image:.1f} objek / frame")
    
    print(f"\n[1] DISTRIBUSI KEPUTUSAN ROUTER (FRAMES)")
    print("-" * 50)
    print(f"🛑 HARD Frames (P2 ON)  : {hard_images} ({hard_pct:.2f}%) -> Ada objek kecil")
    print(f"✅ EASY Frames (P2 OFF) : {easy_images} ({easy_pct:.2f}%) -> Hanya M/L atau kosong")
    print(f"   *Catatan: Terdapat {empty_images} citra tanpa objek sama sekali.")
    
    print(f"\n[2] DISTRIBUSI SKALA OBJEK (COCO METRICS)")
    print("-" * 50)
    print(f"Total Objek            : {total_objects}")
    print(f"🔸 Small Objects (<32) : {small_objects} ({small_obj_pct:.2f}%)")
    print(f"🔸 Medium Objects      : {medium_objects} ({med_obj_pct:.2f}%)")
    print(f"🔸 Large Objects (>96) : {large_objects} ({large_obj_pct:.2f}%)")
    print(f"==================================================")
