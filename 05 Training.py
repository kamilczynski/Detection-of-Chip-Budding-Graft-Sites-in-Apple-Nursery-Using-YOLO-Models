!pip install -U ultralytics opencv-python



import ultralytics
print(ultralytics.__version__)



import sys, torch
print("PY:", sys.executable)
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



import sys
import torch
import platform
import jupyterlab
# --- Python ---
print("🐍 Python version:", sys.version)
# --- PyTorch ---
print("🔥 PyTorch version:", torch.__version__)
# --- CUDA (jeśli dostępna) ---
print("⚙️ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  CUDA version (z PyTorch):", torch.version.cuda)
    print("  GPU name:", torch.cuda.get_device_name(0))
else:
    print("  (CUDA niedostępna w tym środowisku)")
# --- cuDNN ---
print("🧠 cuDNN version:", torch.backends.cudnn.version())
# --- JupyterLab ---
print("📓 JupyterLab version:", jupyterlab.__version__)
# --- Dodatkowo (opcjonalnie) ---
print("💻 System info:", platform.platform())



from pathlib import Path
# Ścieżka do Twojego datasetu CUCU
DATASET_RGB = Path(r"")
# Sprawdzenie struktury YOLO
for p in [
    DATASET_RGB,
    DATASET_RGB / "images/train",
    DATASET_RGB / "images/val",      # walidacja (wcześniej był "test")
    DATASET_RGB / "images/test", 
    DATASET_RGB / "labels/train",
    DATASET_RGB / "labels/val",
    DATASET_RGB / "labels/test"
]:
    print(p, "→", "OK ✅" if p.exists() else "❌ NIE ISTNIEJE")


data_yaml = """
path: C:/
train: images/train
val: images/val
test: images/test
names:
  0: shrubs
"""
with open("data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)
print("✔ Zapisano data.yaml")



from ultralytics import YOLO
# Wczytanie modelu bazowego
model = YOLO("yolos.pt")
# Trening
results = model.train(
    data="data.yaml",         # 🔹 zmień na swój poprawny plik YAML
    epochs=200,
    batch=32,
    imgsz=640,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    lrf=0.01,
    seed=0,
    augment=True,
    workers=8,  # zoptymalizowane do Ryzen 9 9950X
    patience=0,
    device='cuda',
    project="runs/train_S",     # folder w którym zapisywane będą wyniki
    name="yolos_S",           # nazwa sesji
)
