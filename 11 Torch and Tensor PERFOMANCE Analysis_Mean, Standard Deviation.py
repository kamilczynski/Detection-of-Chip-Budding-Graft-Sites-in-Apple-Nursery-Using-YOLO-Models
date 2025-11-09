import pandas as pd
import numpy as np
import os

# STATISTICS OF OBTAINED RESULTS — FINAL VERSION FOR GITHUB
# === Paths ===
input_path = r"D:\.xlsx"
output_dir = r"D:\"
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
# For .xlsx files, use the openpyxl engine
df = pd.read_excel(input_path, engine="openpyxl")

# === Convert columns to numeric values (safely) ===
for col in ["fps", "latency_ms", "map50_95_over_fps"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with NaN values (if any appear)
df = df.dropna(subset=["fps", "latency_ms", "map50_95_over_fps"], how="any")

# === Optionally recompute map50_95_over_fps from FPS if mAP values are stored separately ===
# (Comment out if you don’t want to overwrite)
# if "map50_95" in df.columns:
#     df["map50_95_over_fps"] = df["map50_95"] / df["fps"]

# === Compute per-model statistics ===
summary = (
    df.groupby("model")
      .agg({
          "fps": ["mean", "std", "median"],
          "latency_ms": ["mean", "std", "median"],
          "map50_95_over_fps": ["mean", "std", "median"]
      })
)

# === Flatten column headers ===
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)

# === Coefficient of Variation (CV%) ===
summary["fps_cv_percent"] = (summary["fps_std"] / summary["fps_mean"]) * 100
summary["latency_cv_percent"] = (summary["latency_ms_std"] / summary["latency_ms_mean"]) * 100
summary["map_cv_percent"] = (summary["map50_95_over_fps_std"] / summary["map50_95_over_fps_mean"]) * 100

# === Rounding ===
summary = summary.round(6)  # 6 decimals to preserve precision

# === Save results ===
output_path = os.path.join(output_dir, "FINAL_RTX_STATISTICS_FIX.xlsx")
summary.to_excel(output_path, index=False)
print(f"✅ Results saved to: {output_path}")
