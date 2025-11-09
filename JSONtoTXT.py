import json
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_via_entries(json_path: Path):
    """
    Loads annotations from VIA 2.x or 3.x JSON files.
    Returns: dict filename -> list(region), where each region has:
        - "shape_attributes": {...}
        - "region_attributes": {...}
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # VIA 3.x "project" format: has fields "file" and "metadata"
    if isinstance(data, dict) and "file" in data and "metadata" in data:
        files = data.get("file") or {}
        meta = data.get("metadata") or {}
        fname_by_id = {str(k): (v.get("fname") or v.get("filename")) for k, v in files.items()}
        image_map = defaultdict(list)
        for m in meta.values():
            img_id = str(m.get("vid") or m.get("iid") or m.get("image_id") or "")
            fname = fname_by_id.get(img_id)
            if not fname:
                continue
            shape = m.get("xy") or m.get("shape_attributes") or m.get("shape") or {}
            attrs = m.get("av") or m.get("region_attributes") or {}
            image_map[fname].append({"shape_attributes": shape, "region_attributes": attrs})
        return image_map

    # VIA 2.x: dict with entries {filename, regions, ...}
    if isinstance(data, dict):
        image_map = defaultdict(list)
        for v in data.values():
            if not isinstance(v, dict):
                continue
            fname = v.get("filename")
            regs = v.get("regions", []) or []
            if fname:
                for r in regs:
                    image_map[fname].append({
                        "shape_attributes": r.get("shape_attributes", {}) or {},
                        "region_attributes": r.get("region_attributes", {}) or {}
                    })
        return image_map

    raise ValueError(f"Unknown VIA JSON format: {json_path}")


def region_to_bbox(shape):
    """
    Returns (x, y, w, h) in pixels for different VIA shape variants:
      - rect: x,y,width,height
      - VIA 3: x,y,w,h
      - two corners: x1,y1,x2,y2
      - polygon: all_points_x, all_points_y -> bbox
      - circle: cx,cy,r -> bbox
      - ellipse: cx,cy,rx,ry -> bbox
    Returns None if a bounding box cannot be determined.
    """
    keys = {k.lower(): k for k in shape.keys()}

    def g(k, default=None):
        v = shape.get(keys.get(k, k), default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    name = (shape.get(keys.get("name", "name")) or
            shape.get(keys.get("type", "type")) or "").lower()

    # 1) Standard rect: x, y, width, height
    if all(x in keys for x in ("x", "y", "width", "height")) or name == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        if None not in (x, y, w, h):
            return x, y, w, h

    # 2) VIA 3.x: x, y, w, h
    if all(x in keys for x in ("x", "y", "w", "h")):
        x, y, w, h = g("x"), g("y"), g("w"), g("h")
        if None not in (x, y, w, h):
            return x, y, w, h

    # 3) Two corners: x1, y1, x2, y2
    if all(x in keys for x in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
        if None not in (x1, y1, x2, y2):
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            return x, y, w, h

    # 4) Polygon → bounding box
    if "all_points_x" in keys and "all_points_y" in keys:
        xs = shape.get(keys["all_points_x"]) or []
        ys = shape.get(keys["all_points_y"]) or []
        if xs and ys and len(xs) == len(ys):
            try:
                xs = [float(v) for v in xs]
                ys = [float(v) for v in ys]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                return xmin, ymin, xmax - xmin, ymax - ymin
            except (TypeError, ValueError):
                pass

    # 5) Circle: cx, cy, r
    if all(x in keys for x in ("cx", "cy", "r")):
        cx, cy, r = g("cx"), g("cy"), g("r")
        if None not in (cx, cy, r):
            return cx - r, cy - r, 2 * r, 2 * r

    # 6) Ellipse: cx, cy, rx, ry
    if all(x in keys for x in ("cx", "cy", "rx", "ry")):
        cx, cy, rx, ry = g("cx"), g("cy"), g("rx"), g("ry")
        if None not in (cx, cy, rx, ry):
            return cx - rx, cy - ry, 2 * rx, 2 * ry

    return None


def clamp01(v):
    return max(0.0, min(1.0, v))


def index_images(images_dir: Path):
    """Indexes all image files in directory tree for fast matching."""
    by_name = {}
    by_stem = defaultdict(list)
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            by_name[p.name] = p
            by_stem[p.stem].append(p)
    return by_name, by_stem


def find_image_path(fname: str, images_dir: Path, by_name, by_stem):
    """Match by full filename first, then by stem (different extension), finally by relative path."""
    p = by_name.get(Path(fname).name)
    if p and p.exists():
        return p
    cands = by_stem.get(Path(fname).stem, [])
    if cands:
        return cands[0]
    direct = images_dir / fname
    return direct if direct.exists() else None


def main():
    root = tk.Tk()
    root.withdraw()

    # 1) Select one or more JSON files
    json_files = filedialog.askopenfilenames(
        title="Select one or more VIA JSON files (train/val/test)",
        filetypes=[("JSON files", "*.json")]
    )
    if not json_files:
        messagebox.showerror("Error", "No JSON files selected.")
        return

    # 2) Select the image folder
    images_dir = filedialog.askdirectory(title="Select folder containing images")
    if not images_dir:
        messagebox.showerror("Error", "No image folder selected.")
        return
    images_dir = Path(images_dir)

    # 3) Select output folder
    out_dir = filedialog.askdirectory(title="Select output folder for YOLO labels")
    if not out_dir:
        messagebox.showerror("Error", "No output folder selected.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge annotations from all JSONs
    merged = defaultdict(list)  # filename -> list(regions)
    total_regions = 0
    for jf in json_files:
        try:
            image_map = load_via_entries(Path(jf))
        except Exception as e:
            messagebox.showerror("JSON Error", f"{jf}\n{e}")
            return
        for fname, regs in image_map.items():
            merged[fname].extend(regs)
            total_regions += len(regs)

    # Build image index
    by_name, by_stem = index_images(images_dir)

    created_txt = 0
    imgs_with_anns = 0
    skipped_regions = 0
    not_found = 0
    processed = set()
    shape_counter = Counter()

    # 4) Write YOLO labels for annotated images
    for fname, regions in merged.items():
        img_path = find_image_path(fname, images_dir, by_name, by_stem)
        if not img_path:
            print(f"[WARN] Image not found: {fname}")
            not_found += 1
            continue

        processed.add(img_path.resolve())
        try:
            with Image.open(img_path) as im:
                w_img, h_img = im.size
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            continue

        yolo_lines = []
        for r in regions:
            sa = r.get("shape_attributes", {}) or {}
            stype = (sa.get("name") or sa.get("type") or
                     ("polygon" if ("all_points_x" in sa and "all_points_y" in sa) else "unknown"))
            shape_counter[str(stype).lower()] += 1

            bbox = region_to_bbox(sa)
            if bbox is None:
                skipped_regions += 1
                continue

            x, y, w, h = bbox
            xc = (x + w / 2.0) / w_img
            yc = (y + h / 2.0) / h_img
            wn = w / w_img
            hn = h / h_img
            xc, yc, wn, hn = map(clamp01, (xc, yc, wn, hn))

            # Single class 'apple' -> id = 0
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_txt = out_dir / (img_path.stem + ".txt")
        out_txt.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        created_txt += 1
        if yolo_lines:
            imgs_with_anns += 1

    # 5) Create empty .txt for remaining images (so YOLO has one per image)
    for p in by_name.values():
        if p.resolve() in processed:
            continue
        out_txt = out_dir / (p.stem + ".txt")
        if not out_txt.exists():
            out_txt.write_text("", encoding="utf-8")
            created_txt += 1

    # 6) classes.txt
    (out_dir / "classes.txt").write_text("apple\n", encoding="utf-8")

    # Summary
    total_imgs_on_disk = len(by_name)
    annotated_unique = len(merged)
    shapes_info = ", ".join(f"{k}:{v}" for k, v in sorted(shape_counter.items()))
    msg = (
        "✅ Conversion complete\n\n"
        f"Selected JSON files: {len(json_files)}\n"
        f"Annotated (unique) images in JSON: {annotated_unique}\n"
        f"Total regions (bbox) in JSON: {total_regions}\n"
        f"Shapes found in JSON: {shapes_info}\n\n"
        f"Images found in folder: {total_imgs_on_disk}\n"
        f"— with annotations saved: {imgs_with_anns}\n"
        f"— skipped (unsupported) regions: {skipped_regions}\n"
        f"— missing images: {not_found}\n\n"
        f"Total .txt files created: {created_txt}\n"
        f"Labels and classes.txt saved in: {out_dir}\n"
        f"Classes: apple (id=0)"
    )
    print(msg)
    messagebox.showinfo("YOLO conversion (1 class)", msg)


if __name__ == "__main__":
    main()
