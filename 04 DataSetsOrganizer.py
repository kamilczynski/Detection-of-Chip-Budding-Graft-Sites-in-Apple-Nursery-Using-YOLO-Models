#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB YOLO Dataset Builder (Dual)
- Dual dataset: DETECTION + SEGMENTATION with same images (different labels)
- COPY (not move) to destination structure
- Shared train/valid/test split for both datasets
- Simplified validation: only checks if .txt files exist and are not empty
- CSV reports: split, classes (from detection), missing pairs, and label errors
"""

import os
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "RGB YOLO Dataset Builder (Dual)"
RANDOM_SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}
IGNORED_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
IGNORED_DIRNAME_TOKENS = {"__MACOSX"}


def is_hidden_or_ignored(fname: str, dirpath: str) -> bool:
    if fname.startswith("."):
        return True
    if fname.lower() in IGNORED_FILENAMES:
        return True
    if any(tok in dirpath.upper() for tok in IGNORED_DIRNAME_TOKENS):
        return True
    return False


# --- Validation (simplified) -------------------------------------------------

def validate_label_file_det(path: Path, num_classes: int | None):
    """DET: only checks if the file exists and is not empty."""
    try:
        if not path.exists():
            return False, "label file missing", None
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        return False, f"read error: {e}", None

    if not txt:
        return False, "empty file", None
    return True, None, []


def validate_label_file_seg(path: Path, num_classes: int | None):
    """SEG: only checks if the file exists and is not empty."""
    try:
        if not path.exists():
            return False, "label file missing", None
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        return False, f"read error: {e}", None

    if not txt:
        return False, "empty file", None
    return True, None, []


# --- Helpers -----------------------------------------------------------------

def top_level_folder(root: Path, path: Path) -> str:
    rel = path.parent.relative_to(root)
    parts = rel.parts
    return parts[0] if len(parts) > 0 else "."


# --- DET Scanning ------------------------------------------------------------

def scan_source_det(root: Path, num_classes: int | None):
    """
    Recursively scans directory for pairs .jpg/.png + .txt (DETECTION),
    only checks if label files are not empty. Works independently of images/labels folders.
    """
    temp = defaultdict(lambda: {"image": None, "label": None, "any_path": None})
    class_hist = Counter()
    file_locations = defaultdict(set)

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_hidden_or_ignored(fname, dirpath):
                continue

            p = Path(dirpath) / fname
            stem, ext = p.stem, p.suffix.lower()

            if ext in IMG_EXTS:
                temp[stem]["image"] = p
                temp[stem]["any_path"] = str(p)
                file_locations[fname].add(str(p))
            elif ext == ".txt":
                temp[stem]["label"] = p
                temp[stem]["any_path"] = str(p)
                file_locations[fname].add(str(p))

    # --- Check completeness ---
    series_ok, incomplete = {}, {}
    for base, packs in temp.items():
        img, lbl = packs["image"], packs["label"]
        folder_name = top_level_folder(root, Path(packs["any_path"]))
        problems, ok = [], True

        if img is None:
            ok = False
            problems.append("missing image")
        if lbl is None:
            ok = False
            problems.append("missing DET label (.txt)")
        else:
            try:
                txt = lbl.read_text(encoding="utf-8", errors="ignore").strip()
                if not txt:
                    ok = False
                    problems.append("empty label file")
            except Exception as e:
                ok = False
                problems.append(f"label read error: {e}")

        if ok:
            series_ok[base] = {"image": img, "label": lbl, "folder": folder_name}
        else:
            incomplete[base] = {"folder": folder_name, "problems": problems}

    duplicates = {fn: sorted(paths) for fn, paths in file_locations.items() if len(paths) > 1}
    return series_ok, incomplete, duplicates, class_hist


# --- Structure & Copying -----------------------------------------------------

def ensure_structure_dual(dst_root: Path):
    for mode in ["detection", "segmentation"]:
        for kind in ["images", "labels"]:
            for split in ["train", "valid", "test"]:
                (dst_root / mode / kind / split).mkdir(parents=True, exist_ok=True)


def check_dual_missing_and_validate(series: dict, src_det_root: Path, src_seg_root: Path, num_classes: int | None):
    missing, bad_det, bad_seg = [], [], []
    for base, pack in series.items():
        img_name = pack["image"].name
        lbl_name = pack["label"].name

        # DET
        det_img = src_det_root / "images" / img_name
        det_lbl = src_det_root / "labels" / lbl_name
        if not det_img.exists():
            missing.append((base, "detection", "image"))
        if not det_lbl.exists():
            missing.append((base, "detection", "label"))
        else:
            ok, err, _ = validate_label_file_det(det_lbl, num_classes)
            if not ok:
                bad_det.append((base, err))

        # SEG
        seg_img = src_seg_root / "images" / img_name
        seg_lbl = src_seg_root / "labels" / lbl_name
        if not seg_img.exists():
            missing.append((base, "segmentation", "image"))
        if not seg_lbl.exists():
            missing.append((base, "segmentation", "label"))
        else:
            ok, err, _ = validate_label_file_seg(seg_lbl, num_classes)
            if not ok:
                bad_seg.append((base, err))

    return missing, bad_det, bad_seg


def copy_series_dual(assign_map, series, dst_root: Path, excluded: set[str],
                     src_det_root: Path, src_seg_root: Path):
    """
    Copies images + labels for both DETECTION and SEGMENTATION sets.
    Creates structure automatically if missing.
    """
    ensure_structure_dual(dst_root)

    for base, pack in series.items():
        if base in excluded:
            continue

        split = assign_map[base]
        img_name = pack["image"].name
        lbl_name = pack["label"].name

        def find_file(root: Path, sub: str, name: str):
            p = root / sub / name
            if p.exists():
                return p
            alt = root / name
            return alt if alt.exists() else None

        for mode, src_root in [("detection", src_det_root), ("segmentation", src_seg_root)]:
            i_src = find_file(src_root, "images", img_name)
            l_src = find_file(src_root, "labels", lbl_name)

            i_dst = dst_root / mode / "images" / split / img_name
            l_dst = dst_root / mode / "labels" / split / lbl_name

            if i_src and i_src.exists():
                i_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(i_src, i_dst)

            if l_src and l_src.exists():
                l_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(l_src, l_dst)
# --- Split logic -------------------------------------------------------------

def compute_split_counts(n_series: int, pct_train: float, pct_valid: float, pct_test: float):
    """Compute number of samples per split (floored)."""
    if abs((pct_train + pct_valid + pct_test) - 100.0) > 1e-6:
        raise ValueError("The sum of percentages must equal 100.")
    t = int(n_series * pct_train / 100.0)
    v = int(n_series * pct_valid / 100.0)
    s = n_series - t - v
    return {"train": t, "valid": v, "test": s}


def compute_target_counts(n_series: int, pct_train: float, pct_valid: float, pct_test: float):
    """Compute target sample counts and round/correct to sum up properly."""
    if abs((pct_train + pct_valid + pct_test) - 100.0) > 1e-6:
        raise ValueError("The sum of percentages must equal 100.")
    raw = {
        "train": n_series * pct_train / 100.0,
        "valid": n_series * pct_valid / 100.0,
        "test":  n_series * pct_test / 100.0,
    }
    rounded = {k: int(round(v)) for k, v in raw.items()}
    delta = sum(rounded.values()) - n_series
    if delta != 0:
        order = sorted(raw.keys(), key=lambda k: abs(raw[k] - rounded[k]), reverse=True)
        i = 0
        while delta != 0 and i < 10:
            k = order[i % 3]
            rounded[k] -= 1 if delta > 0 else -1
            delta = sum(rounded.values()) - n_series
            i += 1
    return rounded


def pick_and_assign_per_folder(series_by_folder: dict, pct_train: float, pct_valid: float, pct_test: float):
    """Randomly assign series per folder to train/valid/test splits."""
    assign = {}
    for folder, bases in series_by_folder.items():
        counts = compute_split_counts(len(bases), pct_train, pct_valid, pct_test)
        shuffled = bases[:]
        rnd = random.Random(f"{RANDOM_SEED}-{folder}")
        rnd.shuffle(shuffled)
        i = 0
        for split in ["train", "valid", "test"]:
            k = counts[split]
            for _ in range(k):
                if i >= len(shuffled):
                    break
                assign[shuffled[i]] = split
                i += 1
        while i < len(shuffled):
            assign[shuffled[i]] = "train"
            i += 1
    return assign


def adjust_assign_to_exact(assign_map: dict, series_by_folder: dict, target: dict,
                           topper_folder: str | None, reduce_to_fit: bool):
    """Simple version — keeps approximate proportions."""
    log = ["[INFO] No proportional corrections applied (approximate split kept)."]
    excluded = set()
    return assign_map, excluded, log


# --- CSV Reports -------------------------------------------------------------

def write_missing_report_csv(missing, dst_root: Path):
    path = dst_root / "report_missing.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "mode", "file_type"])
        for base, mode, typ in missing:
            w.writerow([base, mode, typ])
    return path


def write_bad_report_csv(bad_list, dst_root: Path, filename: str):
    path = dst_root / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "error"])
        for base, err in bad_list:
            w.writerow([base, err])
    return path


def write_split_report_csv(assign_map, series, dst_root: Path, excluded: set[str]):
    path = dst_root / "report_split.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "split", "folder", "used"])
        for base in sorted(series.keys()):
            used = "no" if base in excluded else "yes"
            split = assign_map.get(base, "-")
            folder = series[base]["folder"]
            w.writerow([base, split, folder, used])
    return path


def write_classes_report_csv(class_hist: Counter, dst_root: Path):
    path = dst_root / "report_classes.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "count"])
        for cls, cnt in sorted(class_hist.items()):
            w.writerow([cls, cnt])
    return path


# --- Scrollable message window ----------------------------------------------

def show_scrollable_message(title: str, content: str):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x500")
    win.configure(bg="#1B1B1B")
    frm = ttk.Frame(win)
    frm.pack(fill="both", expand=True)

    yscroll = ttk.Scrollbar(frm, orient="vertical")
    txt = tk.Text(frm, wrap="none", yscrollcommand=yscroll.set, bg="#1B1B1B", fg="white")
    yscroll.config(command=txt.yview)
    yscroll.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)

    txt.insert("1.0", content)
    txt.config(state="disabled")

    ttk.Button(win, text="OK", command=win.destroy).pack(pady=8)
    win.wait_window()


# --- DARK THEMED GUI --------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x840")
        self.configure(bg="#1B1B1B")

        # Dark theme setup
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#1B1B1B", foreground="white")
        style.configure("TFrame", background="#1B1B1B")
        style.configure("TLabelframe", background="#2A2A2A", foreground="#A0522D")
        style.configure("TLabelframe.Label", background="#2A2A2A", foreground="#A0522D", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background="#4CAF50", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#3E8E41")])
        style.configure("TEntry", fieldbackground="#333333", foreground="white")
        style.configure("Treeview", background="#2A2A2A", fieldbackground="#2A2A2A", foreground="white")
        style.configure("Treeview.Heading", background="#4CAF50", foreground="white")

        # Variables
        self.src_path_det = tk.StringVar()
        self.src_path_seg = tk.StringVar()
        self.dst_path = tk.StringVar()
        self.n_series = tk.IntVar(value=0)
        self.pct_train = tk.DoubleVar(value=70.0)
        self.pct_valid = tk.DoubleVar(value=20.0)
        self.pct_test = tk.DoubleVar(value=10.0)
        self.num_classes = tk.StringVar(value="")
        self.topper_folder = tk.StringVar(value="")
        self.reduce_to_fit = tk.BooleanVar(value=False)
        self.series = {}
        self.assign_map = {}
        self.series_by_folder = {}
        self.class_hist = Counter()

        self._build_ui()

    # --- BUILD UI ---
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # --- Paths ---
        src_det_frame = ttk.LabelFrame(self, text="Detection Source (expects images/ and labels/)")
        src_det_frame.pack(fill="x", **pad)
        ttk.Entry(src_det_frame, textvariable=self.src_path_det).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(src_det_frame, text="Browse…", command=lambda: self.choose_src(self.src_path_det)).pack(side="right", padx=6, pady=6)

        src_seg_frame = ttk.LabelFrame(self, text="Segmentation Source (expects images/ and labels/)")
        src_seg_frame.pack(fill="x", **pad)
        ttk.Entry(src_seg_frame, textvariable=self.src_path_seg).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(src_seg_frame, text="Browse…", command=lambda: self.choose_src(self.src_path_seg)).pack(side="right", padx=6, pady=6)

        dst_frame = ttk.LabelFrame(self, text="Destination Folder (will create detection/ and segmentation/)")
        dst_frame.pack(fill="x", **pad)
        ttk.Entry(dst_frame, textvariable=self.dst_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(dst_frame, text="Browse…", command=self.choose_dst).pack(side="right", padx=6, pady=6)

        # --- Class Validation ---
        val_frame = ttk.LabelFrame(self, text="Class Validation (optional, detection only)")
        val_frame.pack(fill="x", **pad)
        ttk.Label(val_frame, text="Number of classes (optional):").grid(row=0, column=0, sticky="e")
        ttk.Entry(val_frame, width=8, textvariable=self.num_classes).grid(row=0, column=1, sticky="w")
        ttk.Label(val_frame, text="(used for histogram only; validation checks non-empty files)").grid(row=0, column=2, sticky="w")

        # --- Scan ---
        scan_frame = ttk.Frame(self)
        scan_frame.pack(fill="x", **pad)
        ttk.Button(scan_frame, text="Scan DET & Count", command=self.scan_action).pack(side="left")
        self.summary_lbl = ttk.Label(scan_frame, text="No data yet. Please scan sources first.")
        self.summary_lbl.pack(side="left", padx=12)

        # --- Split ratios ---
        pct_frame = ttk.LabelFrame(self, text="Split Percentages (shared for both datasets)")
        pct_frame.pack(fill="x", **pad)
        ttk.Label(pct_frame, text="Train [%]:").grid(row=0, column=0, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_train).grid(row=0, column=1, sticky="w")
        ttk.Label(pct_frame, text="Valid [%]:").grid(row=0, column=2, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_valid).grid(row=0, column=3, sticky="w")
        ttk.Label(pct_frame, text="Test [%]:").grid(row=0, column=4, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_test).grid(row=0, column=5, sticky="w")
        ttk.Button(pct_frame, text="Recalculate Split", command=self.compute_splits).grid(row=0, column=7, padx=10)

        # --- Folder selector ---
        topper_frame = ttk.LabelFrame(self, text="Preferred Folder for Adjustment (optional)")
        topper_frame.pack(fill="x", **pad)
        self.topper_combo = ttk.Combobox(topper_frame, textvariable=self.topper_folder, values=[], state="readonly")
        self.topper_combo.pack(side="left", padx=6, pady=6)
        ttk.Label(topper_frame, text="(select after scanning)").pack(side="left")

        # (The rest of the UI — tables, histogram, actions, log — continues identically below)
        # --- Global split table ---
        table_frame = ttk.LabelFrame(self, text="Global Split Summary")
        table_frame.pack(fill="x", **pad)
        self.tree_global = ttk.Treeview(table_frame, columns=("split", "count", "target"), show="headings", height=4)
        self.tree_global.heading("split", text="Split")
        self.tree_global.heading("count", text="Current")
        self.tree_global.heading("target", text="Target")
        self.tree_global.column("split", width=120, anchor="center")
        self.tree_global.column("count", width=140, anchor="e")
        self.tree_global.column("target", width=140, anchor="e")
        self.tree_global.pack(fill="x", padx=6, pady=6)

        # --- Per-folder table ---
        per_folder_frame = ttk.LabelFrame(self, text="Split per Folder (counts and %)")
        per_folder_frame.pack(fill="both", expand=True, **pad)
        cols = ("folder", "total", "train", "valid", "test", "pct_train", "pct_valid", "pct_test")
        self.tree_folders = ttk.Treeview(per_folder_frame, columns=cols, show="headings", height=12)
        headers = {
            "folder": "Folder", "total": "Total", "train": "Train",
            "valid": "Valid", "test": "Test",
            "pct_train": "%Train", "pct_valid": "%Valid", "pct_test": "%Test",
        }
        widths = {k: 80 for k in cols}; widths["folder"] = 260
        anchors = {k: "e" for k in cols}; anchors["folder"] = "w"
        for c in cols:
            self.tree_folders.heading(c, text=headers[c])
            self.tree_folders.column(c, width=widths[c], anchor=anchors[c])
        self.tree_folders.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Class histogram ---
        cls_frame = ttk.LabelFrame(self, text="Class Histogram (from Detection)")
        cls_frame.pack(fill="x", **pad)
        self.tree_classes = ttk.Treeview(cls_frame, columns=("cls", "count"), show="headings", height=6)
        self.tree_classes.heading("cls", text="Class ID")
        self.tree_classes.heading("count", text="Count")
        self.tree_classes.column("cls", width=120, anchor="e")
        self.tree_classes.column("count", width=160, anchor="e")
        self.tree_classes.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(cls_frame, text="Export CSV", command=self.export_classes_csv).pack(side="left", padx=10)

        # --- Action buttons ---
        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        ttk.Button(actions, text="Copy to Destination (Dual)", command=self.move_action).pack(side="left")
        ttk.Button(actions, text="Exit", command=self.destroy).pack(side="right")

        # --- Log window ---
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10, bg="#1B1B1B", fg="white")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Event handlers ----------
    def choose_src(self, var):
        path = filedialog.askdirectory(title="Select Source Directory")
        if path:
            var.set(path)

    def choose_dst(self):
        path = filedialog.askdirectory(title="Select Destination Directory")
        if path:
            self.dst_path.set(path)

    def log_write(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    # ---------- Scanning ----------
    def scan_action(self):
        src_det = self.src_path_det.get().strip()
        src_seg = self.src_path_seg.get().strip()
        if not src_det or not src_seg:
            messagebox.showerror(APP_TITLE, "Please select both source directories (Detection & Segmentation).")
            return
        root_det, root_seg = Path(src_det), Path(src_seg)
        if not root_det.exists() or not root_seg.exists():
            messagebox.showerror(APP_TITLE, "One of the source directories does not exist.")
            return

        nc = None
        if self.num_classes.get().strip() != "":
            try:
                tmp = int(self.num_classes.get().strip())
                if tmp <= 0:
                    raise ValueError
                nc = tmp
            except Exception:
                messagebox.showerror(APP_TITLE, "Number of classes must be a positive integer or blank.")
                return

        self.log_write(f"[SCAN] Starting: DET={root_det} | SEG={root_seg}")
        series_ok, incomplete, duplicates, class_hist = scan_source_det(root_det, nc)

        if duplicates:
            details = []
            for fname, paths in sorted(duplicates.items()):
                details.append(fname)
                details += [f"    {p}" for p in paths]
            show_scrollable_message(APP_TITLE, "Duplicate filenames found:\n\n" + "\n".join(details))
            self.log_write("[ERROR] Duplicate files detected.")
            return

        self.series = series_ok
        self.class_hist = class_hist

        if incomplete:
            lines = [f"Incomplete / invalid DET series: {len(incomplete)}\n"]
            for idx, (base, info) in enumerate(sorted(incomplete.items()), start=1):
                lines.append(f"{idx}) {base}")
                lines.append(f"    folder: {info['folder']}")
                for pr in info["problems"]:
                    lines.append(f"    - {pr}")
                lines.append("")
            show_scrollable_message(APP_TITLE, "\n".join(lines))
            self.log_write(f"[WARN] {len(incomplete)} incomplete DET series found.")

        n_series = len(series_ok)
        self.summary_lbl.config(text=f"DET series OK: {n_series}")
        self.log_write(f"[OK] Complete DET series found: {n_series}")

        # Build folder mapping
        series_by_folder = defaultdict(list)
        for base, data in self.series.items():
            series_by_folder[data["folder"]].append(base)
        self.series_by_folder = dict(sorted(series_by_folder.items()))

        self.topper_combo["values"] = list(self.series_by_folder.keys())
        if not self.topper_folder.get() and self.topper_combo["values"]:
            self.topper_folder.set(self.topper_combo["values"][0])

        self.refresh_classes_view()
        self.compute_splits()

    def refresh_classes_view(self):
        for r in self.tree_classes.get_children():
            self.tree_classes.delete(r)
        for cls, cnt in sorted(self.class_hist.items()):
            self.tree_classes.insert("", "end", values=(cls, cnt))

    # ---------- Splitting ----------
    def compute_splits(self):
        if not self.series:
            return
        try:
            pctT, pctV, pctE = self.pct_train.get(), self.pct_valid.get(), self.pct_test.get()
            _ = compute_split_counts(1, pctT, pctV, pctE)
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        global_counts = {"train": 0, "valid": 0, "test": 0}
        folder_rows = []
        for folder, bases in self.series_by_folder.items():
            total = len(bases)
            counts = compute_split_counts(total, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
            tr, va, te = counts["train"], counts["valid"], counts["test"]
            pct = lambda x, tot: 0.0 if tot == 0 else round(100.0 * x / tot, 1)
            folder_rows.append((folder, total, tr, va, te,
                                f"{pct(tr,total):.1f}", f"{pct(va,total):.1f}", f"{pct(te,total):.1f}"))
            for k in global_counts:
                global_counts[k] += counts[k]

        for r in self.tree_folders.get_children():
            self.tree_folders.delete(r)
        for vals in folder_rows:
            self.tree_folders.insert("", "end", values=vals)

        self.assign_map = pick_and_assign_per_folder(self.series_by_folder,
                                                     self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())

        target = compute_target_counts(len(self.series),
                                       self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())

        current_global = Counter(self.assign_map.values())
        for r in self.tree_global.get_children():
            self.tree_global.delete(r)
        for split in ["train", "valid", "test"]:
            self.tree_global.insert("", "end", values=(split, current_global.get(split, 0), target[split]))

        self.log_write("[INFO] Split per folder and global targets updated.")

    # ---------- Copying ----------
    def move_action(self):
        if not self.series:
            messagebox.showinfo(APP_TITLE, "No scanned DET data.")
            return
        dst = self.dst_path.get().strip()
        if not dst:
            messagebox.showerror(APP_TITLE, "Please select destination folder.")
            return
        dst_root = Path(dst)
        dst_root.mkdir(parents=True, exist_ok=True)

        try:
            _ = compute_split_counts(1, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        target = compute_target_counts(len(self.series),
                                       self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        topper = self.topper_folder.get().strip() or None
        reduce_to_fit = bool(self.reduce_to_fit.get())

        new_assign, excluded, adj_log = adjust_assign_to_exact(
            dict(self.assign_map), self.series_by_folder, target, topper, reduce_to_fit
        )

        src_det_root = Path(self.src_path_det.get().strip())
        src_seg_root = Path(self.src_path_seg.get().strip())
        nc = int(self.num_classes.get()) if self.num_classes.get().strip().isdigit() else None
        missing, bad_det, bad_seg = check_dual_missing_and_validate(self.series, src_det_root, src_seg_root, nc)

        rep_missing = write_missing_report_csv(missing, dst_root) if missing else None
        rep_bad_det = write_bad_report_csv(bad_det, dst_root, "report_bad_det.csv") if bad_det else None
        rep_bad_seg = write_bad_report_csv(bad_seg, dst_root, "report_bad_seg.csv") if bad_seg else None

        if any([rep_missing, rep_bad_det, rep_bad_seg]):
            lines = ["Detected problems. Reports:"]
            if rep_missing: lines.append(f"- {rep_missing}")
            if rep_bad_det: lines.append(f"- {rep_bad_det}")
            if rep_bad_seg: lines.append(f"- {rep_bad_seg}")
            show_scrollable_message(APP_TITLE, "\n".join(lines))
            self.log_write("[WARN] Problem reports saved (DET/SEG).")

        if adj_log:
            show_scrollable_message(APP_TITLE, "Target adjustment log:\n\n" + "\n".join(adj_log))
            self.log_write("[INFO] Target adjustments complete.")

        try:
            copy_series_dual(new_assign, self.series, dst_root, excluded, src_det_root, src_seg_root)
        except Exception as e:
            show_scrollable_message(APP_TITLE, f"Copying error:\n\n{e}")
            self.log_write(f"[ERROR] {e}")
            return

        rep_split = write_split_report_csv(new_assign, self.series, dst_root, excluded)
        rep_classes = write_classes_report_csv(self.class_hist, dst_root)

        msg = f"✅ Copy complete.\nSplit report: {rep_split}\nClass report: {rep_classes}"
        if rep_missing: msg += f"\nMissing report: {rep_missing}"
        if rep_bad_det: msg += f"\nDET errors: {rep_bad_det}"
        if rep_bad_seg: msg += f"\nSEG errors: {rep_bad_seg}"
        messagebox.showinfo(APP_TITLE, msg)
        self.log_write("[OK] Finished. " + msg.replace('\n', ' '))

    # ---------- Export ----------
    def export_classes_csv(self):
        if not self.class_hist:
            messagebox.showinfo(APP_TITLE, "No class data. Please scan first.")
            return
        dst = self.dst_path.get().strip()
        if not dst:
            messagebox.showerror(APP_TITLE, "Select destination folder for CSV.")
            return
        dst_root = Path(dst)
        dst_root.mkdir(parents=True, exist_ok=True)
        p = write_classes_report_csv(self.class_hist, dst_root)
        messagebox.showinfo(APP_TITLE, f"Saved: {p}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
