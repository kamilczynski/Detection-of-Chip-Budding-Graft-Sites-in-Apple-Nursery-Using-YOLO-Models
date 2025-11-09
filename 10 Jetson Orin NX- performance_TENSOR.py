# -*- coding: utf-8 -*-
"""
YOLO TensorRT Multi-Model Benchmark — Jetson Orin NX Edition (FP16, CUDA)
------------------------------------------------------------------------
Methodology 1:1 consistent with the YOLO Torch benchmark:
✅ FP16 inference
✅ GPU stabilization: 10 s
✅ Warmup: 20 iterations
✅ Measurement: 60 iterations
✅ Trimming between 20–80 percentile
✅ 5 repetitions per model
✅ 150 s cooldown between sessions (3 models)
✅ Per-session save + master Excel file
"""

import os
import time
import datetime
import statistics as st
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
import traceback
from itertools import islice

# ===================== CONFIGURATION =====================
MODELS = [
    {"name": "YOLOv8s_oczkodetekcja8S",   "path": "/weights/best.engine",  "map5095": 0.000},
    
]

SAVE_DIR = "/workspace/benchmarks/yolo_perfoczkafinalengine/"
INPUT_RES = (640, 640)
WARMUP_IT = 20
TIMED_IT = 60
NUM_RUNS = 5
SESSION_COOLDOWN = 150
STABILIZATION_TIME = 10
TRIM_LOWER = 20
TRIM_UPPER = 80
# =========================================================


def ensure_cuda_or_exit():
    if not torch.cuda.is_available():
        raise SystemExit("❌ No CUDA GPU detected.")
    torch.cuda.init()


def sanity_check_gpu():
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    used = torch.cuda.memory_allocated()
    if used / total > 0.9:
        print("⚠️ GPU memory almost full — clearing cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)


def make_dummy(res):
    return torch.zeros((1, 3, res[0], res[1]), dtype=torch.float16, device="cuda")


def stabilize_gpu(model, dummy, seconds=STABILIZATION_TIME):
    print(f"⚙️ Stabilizing GPU clocks ({seconds}s)...")
    end = time.time() + seconds
    with torch.inference_mode():
        while time.time() < end:
            _ = model(dummy)
    torch.cuda.synchronize()


def summarize(vals):
    s = list(map(float, vals))
    q1, q3 = pd.Series(s).quantile([0.25, 0.75])
    return {
        "mean": float(st.mean(s)),
        "std": float(st.pstdev(s)) if len(s) > 1 else 0.0,
        "median": float(st.median(s)),
        "q1": float(q1), "q3": float(q3),
        "iqr": float(q3 - q1), "min": float(min(s)), "max": float(max(s)),
    }


def run_yolo_engine(model, dummy):
    fps_list = []
    with torch.inference_mode():
        # Warmup
        for _ in range(WARMUP_IT):
            _ = model(dummy)
        torch.cuda.synchronize()

        # Measurement
        for _ in range(TIMED_IT):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            fps_list.append(1000.0 / total_ms)

    fps_array = np.array(fps_list)
    trimmed = fps_array[
        (fps_array > np.percentile(fps_array, TRIM_LOWER)) &
        (fps_array < np.percentile(fps_array, TRIM_UPPER))
    ]
    fps_mean = np.mean(trimmed)
    latency_ms = 1000.0 / fps_mean
    jitter = np.std(trimmed) / fps_mean * 100
    return fps_mean, latency_ms, jitter


def benchmark_model(model_info):
    name, path, mapv = model_info["name"], model_info["path"], model_info["map5095"]
    print(f"\n🚀 Benchmarking {name} ({os.path.basename(path)})")

    sanity_check_gpu()

    # ✅ Detect model type based on path
    path_lower = path.lower()
    if "segment" in path_lower or "seg" in path_lower or "mask" in path_lower:
        model = YOLO(path, task="segment")
    else:
        model = YOLO(path, task="detect")

    dummy = make_dummy(INPUT_RES)
    stabilize_gpu(model, dummy)

    runs = []
    for i in range(1, NUM_RUNS + 1):
        fps, lat, jitter = run_yolo_engine(model, dummy)
        ratio = mapv / fps if fps > 0 else float("nan")
        runs.append({
            "run": i, "model": name,
            "fps": fps, "latency_ms": lat,
            "jitter_percent": jitter, "map50_95_over_fps": ratio
        })
        print(f"Run {i:02d}: FPS={fps:.2f} | Lat={lat:.2f} ms | Jitter={jitter:.2f}%")

    df_runs = pd.DataFrame(runs)
    df_summary = pd.DataFrame([
        {"metric": "FPS", **summarize(df_runs["fps"])},
        {"metric": "Latency (ms)", **summarize(df_runs["latency_ms"])},
        {"metric": "Jitter (%)", **summarize(df_runs["jitter_percent"])},
    ])
    df_summary.insert(0, "model", name)
    return df_runs, df_summary


def chunk_models(models, n):
    it = iter(models)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def run_session(session_id, models):
    print(f"\n🧪 --- Session {session_id} --- ({len(models)} models) ---")
    all_runs, all_summaries = [], []
    for m in models:
        try:
            df_r, df_s = benchmark_model(m)
            all_runs.append(df_r)
            all_summaries.append(df_s)
        except Exception as e:
            print(f"❗ Skipped {m['name']}: {e}")
            traceback.print_exc()

    df_all_runs = pd.concat(all_runs, ignore_index=True)
    df_all_summaries = pd.concat(all_summaries, ignore_index=True)
    save_path = os.path.join(SAVE_DIR, f"session_{session_id}_{datetime.datetime.now():%Y%m%d_%H%M}.xlsx")
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_all_runs.to_excel(writer, sheet_name="runs", index=False)
        df_all_summaries.to_excel(writer, sheet_name="summary", index=False)
    print(f"✅ Session {session_id} saved: {save_path}")
    return df_all_runs, df_all_summaries


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_cuda_or_exit()

    all_runs_global, all_summaries_global = [], []
    session_id = 1

    for models_chunk in chunk_models(MODELS, 3):
        df_r, df_s = run_session(session_id, models_chunk)
        all_runs_global.append(df_r)
        all_summaries_global.append(df_s)
        print(f"🕒 Cooldown {SESSION_COOLDOWN}s before the next session...")
        time.sleep(SESSION_COOLDOWN)
        session_id += 1

    df_global_runs = pd.concat(all_runs_global, ignore_index=True)
    df_global_summaries = pd.concat(all_summaries_global, ignore_index=True)
    master_path = os.path.join(SAVE_DIR, f"benchmark_master_{datetime.datetime.now():%Y%m%d_%H%M}.xlsx")
    with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
        df_global_runs.to_excel(writer, sheet_name="runs", index=False)
        df_global_summaries.to_excel(writer, sheet_name="summary", index=False)
    print(f"\n🏁 All sessions completed. Master file saved: {master_path}")


if __name__ == "__main__":
    main()
