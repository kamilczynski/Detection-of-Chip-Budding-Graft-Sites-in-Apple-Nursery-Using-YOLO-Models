from ultralytics import YOLO
import torch
import os
import csv
import yaml
import json
import cv2
import numpy as np

# === CONFIGURATION ===
model_path = "C:/Users/HARDPC/runs/train_oczkosegmentacja11S/yolov11s_oczkosegmentacja11S/weights/best.pt"
data_yaml = "C:/Users/HARDPC/data_oczkosegmentacja.yaml"
save_dir = "C:/Users/HARDPC/Desktop/PROJEKTY CNN/WYNIKI/oczkosegmentacja/METRYKI/11S_SEG"

# === CUDA DETECTION ===
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("❌ No GPU detected! Validation requires a CUDA device (e.g., RTX 5080 or Jetson Orin NX).")

print(f"\n🔥 Starting SEGMENTATION evaluation on TEST SET using device: {torch.cuda.get_device_name(0)}")

os.makedirs(save_dir, exist_ok=True)

# === 1️⃣ Load YOLOv8 SEGMENTATION model ===
model = YOLO(model_path)

# === 2️⃣ Evaluate model on TEST SET ===
print("\n🚀 Starting model evaluation on TEST SET...")
results = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    device=device,
    save_json=True,      # save predictions.json (COCO style)
    verbose=True,
    plots=True
)

# === 3️⃣ Collect and print SEGMENTATION metrics ===
precision = results.seg.p.mean()
recall = results.seg.r.mean()
f1 = results.seg.f1.mean()
map50 = results.seg.map50
map5095 = results.seg.map

print("\n=== 📊 TEST SET METRICS (SEGMENTATION) ===")
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
    raise FileNotFoundError("❌ 'test:' section not defined in data.yaml file.")

if os.path.isabs(test_split):
    test_dir = test_split
else:
    test_dir = os.path.join(base_path, test_split)
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ TEST SET folder not found: {test_dir}")
print(f"\n📂 TEST SET detected: {test_dir}")

# === 6️⃣ Run segmentation and save TEST SET results + JSON ===
print("\n🖼️ Running segmentation on TEST SET...")
pred_results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.25,
    device=device,
    save=False,              # do not save default box images
    save_txt=False,
    verbose=True
)

# === 7️⃣ Save all predictions (masks + boxes) to JSON ===
json_path = os.path.join(save_dir, "predictions_testset.json")
json_data = []

for result in pred_results:
    boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else []
    confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes else []
    classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes else []
    masks = result.masks.data.cpu().numpy().tolist() if result.masks is not None else []

    json_data.append({
        "image": result.path,
        "boxes": boxes,
        "confidences": confs,
        "classes": classes,
        "masks": masks
    })

with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"\n💾 Full SEGMENTATION TEST SET results saved to: {json_path}")

# === 8️⃣ Visualization: semi-transparent masks without boxes ===
pretty_dir = os.path.join(save_dir, "predicted_masks_pretty")
mask_dir = os.path.join(save_dir, "binary_masks")
os.makedirs(pretty_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

print(f"\n🎨 Creating mask visualizations without boxes in: {pretty_dir}")
print(f"🧩 Saving individual binary masks to: {mask_dir}")

# Function to generate a random (pastel) color
def random_color(seed):
    np.random.seed(seed)
    return tuple(np.random.randint(60, 230, size=3).tolist())

for i, result in enumerate(pred_results):
    img = result.orig_img.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # [N, H, W]
        for j, mask in enumerate(masks):
            color = random_color(j)
            alpha = 0.45  # transparency

            # Create a colored semi-transparent overlay
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[mask > 0.5] = color
            img = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)

            # Draw a thin white contour
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 255), 1)

            # Save individual binary mask
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            mask_name = f"{os.path.splitext(os.path.basename(result.path))[0]}_mask_{j}.png"
            cv2.imwrite(os.path.join(mask_dir, mask_name), mask_uint8)

    # Save final image with semi-transparent masks
    save_path = os.path.join(pretty_dir, os.path.basename(result.path))
    cv2.imwrite(save_path, img)

print(f"\n✅ Saved {len(pred_results)} images with colored masks in: {pretty_dir}")
print(f"✅ All binary masks saved in: {mask_dir}")
print("\n=== 🟢 SEGMENTATION TEST SET evaluation completed successfully ===")
