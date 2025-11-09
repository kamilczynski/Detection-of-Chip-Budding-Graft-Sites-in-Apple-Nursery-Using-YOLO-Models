from ultralytics import YOLO
import os, time

# === DISABLE SAFETY RUNTIME MODE ===
os.environ["TENSORRT_SAFE_MODE"] = "0"

# === CONFIGURATION ===
source_dir = "/home/best.pt"
export_dir = "/home/"
os.makedirs(export_dir, exist_ok=True)

# === COLLECT MODELS ===
if os.path.isfile(source_dir) and source_dir.endswith(".pt"):
    pt_models = [source_dir]
else:
    pt_models = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith("best.pt"):
                pt_models.append(os.path.join(root, f))

if not pt_models:
    print("❌ No .pt files found in:", source_dir)
    exit(1)

print(f"🔍 Found {len(pt_models)} models for export:")
for m in pt_models:
    print("   •", m)

# === EXPORT ===
for model_path in pt_models:
    parts = model_path.split('/')
    try:
        data_type = next(p for p in parts if p.startswith("train_")).replace("train_", "")
    except StopIteration:
        data_type = "UNKNOWN"
    model_folder = next((p for p in parts if p.startswith("yolo")), "model")

    export_name = f"{data_type}_{model_folder}.engine"
    export_path = os.path.join(export_dir, export_name)

    print(f"\n🚀 TensorRT Export: {model_path}")
    print(f"📦 Output file: {export_path}")

    try:
        start = time.time()
        model = YOLO(model_path)
        model.export(
            format="engine",
            device=0,
            half=True,          # FP16 precision
            workspace=2048,     # MB
            imgsz=640,
            dynamic=False,
            simplify=True
        )
        end = time.time()
        print(f"✅ Completed ({end - start:.1f}s): {export_name}")
    except Exception as e:
        print(f"💣 Export error for {model_path}: {e}")

print("\n🎯 All models processed.")
print(f"💾 Results saved to: {export_dir}")
