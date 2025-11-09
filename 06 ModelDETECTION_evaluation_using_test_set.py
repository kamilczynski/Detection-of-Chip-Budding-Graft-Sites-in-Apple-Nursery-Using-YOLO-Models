from ultralytics import YOLO
import torch
import os
import csv
import yaml
import json

# === CONFIGURATION ===
model_path = "C:/Users/"
data_yaml = "C:/Users/"
save_dir = "C:/Users/"

# === CUDA DETECTION ===
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("❌ No GPU detected! Validation requires a CUDA-enabled device (e.g., RTX 5080 or Jetson Orin NX).")

print(f"\n🔥 Starting evaluation on TEST SET using device: {torch.cuda.get_device_name(0)}")

os.makedirs(save_dir, exist_ok=True)

# === 1️⃣ Load the model ===
model = YOLO(model_path)

# === 2️⃣ Evaluate the model on TEST SET ===
print("\n🚀 Starting model evaluation on TEST SET...")
results = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    device=device,
    save_json=True,      # 👈 forces saving of predictions.json
    save_hybrid=False,
    verbose=True,
    plots=True
)

# === 3️⃣ Collect and print metrics ===
precision = results.box.p.mean()
recall = results.box.r.mean()
f1 = results.box.f1.mean()
map50 = results.box.map50
map5095 = results.box.map

print("\n=== TEST SET METRICS ===")
print(f"Precision:    {precision:.3f}")
print(f"Recall:       {recall:.3f}")
print(f"F1-score:     {f1:.3f}")
print(f"mAP@0.5:      {map50:.3f}")
print(f"mAP@[.5:.95]: {map5095:.3f}")

# === 4️⃣ Save metrics to CSV ===
csv_path = os.path.join(save_dir, "metrics_summary_testset.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    for name, val in [
        ("Precision", precision),
        ("Recall", recall),
        ("F1-score", f1),
        ("mAP@0.5", map50),
        ("mAP@[.5:.95]", map5095)
    ]:
        writer.writerow([name, f"{val:.4f}"])
print(f"\n📁 TEST SET metrics saved to: {csv_path}")

# === 5️⃣ Load TEST SET path from data.yaml ===
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)
base_path = data_config.get('path', '')
test_split = data_config.get('test', None)
if not test_split:
    raise FileNotFoundError("❌ 'test:' section not defined in data.yaml.")

if os.path.isabs(test_split):
    test_dir = test_split
else:
    test_dir = os.path.join(base_path, test_split)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ TEST SET folder not found: {test_dir}")
print(f"\n📂 TEST SET detected: {test_dir}")

# === 6️⃣ Run detection and save TEST SET images + JSON with predictions ===
print("\n📸 Running detection on TEST SET...")
pred_results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    save_txt=False,
    project=save_dir,
    name="predicted_testset",
    exist_ok=True,
    verbose=True
)

# === 7️⃣ Save all predictions to JSON ===
json_path = os.path.join(save_dir, "predictions_testset.json")
json_data = []
for result in pred_results:
    boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else []
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes else []
    classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes else []
    json_data.append({
        "image": result.path,
        "boxes": boxes,
        "confidences": confs,
        "classes": classes
    })

with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"\n💾 Full TEST SET detection results saved to: {json_path}")
print(f"✅ TEST SET images with bounding boxes saved in: {os.path.join(save_dir, 'predicted_testset')}")
print("\n=== TEST SET evaluation completed successfully ===")
