from ultralytics import YOLO

# Load model dari YAML yang baru kita buat
model = YOLO("ultralytics/cfg/models/v8/yolov8-moe-p2.yaml")

# Coba print info model
model.info()

print("BERHASIL! Model MoE-P2 berhasil dibangun.")

# ngecek summary model yolov8-P2
model1 = YOLO("ultralytics/cfg/models/v8/yolov8-p2.yaml")
# Coba print info model
model1.info()
print("BERHASIL! Model yolov8-P2 berhasil dibangun.")

# ngecek summary model yolov8 biasa
model2 = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
# Coba print info model
model2.info()
print("BERHASIL! Model yolov8 biasa berhasil dibangun.")