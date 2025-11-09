import os
import cv2
import glob
import numpy as np
import random
from tkinter import *
from tkinter import filedialog, messagebox

# ====== FUNCTIONS ======
def load_classes(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [c.strip() for c in f.readlines() if c.strip()]
    return []


def visualize_yolo(images_dir, labels_dir, classes_file):
    class_names = load_classes(classes_file)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_names]

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    if not image_files:
        messagebox.showerror("Error", "No images found in the selected folder.")
        return

    index = 0
    cv2.namedWindow("YOLO Segmentation Preview", cv2.WINDOW_NORMAL)

    while True:
        image_path = image_files[index]
        img = cv2.imread(image_path)
        if img is None:
            index = (index + 1) % len(image_files)
            continue

        h, w = img.shape[:2]
        label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line in lines:
                parts = line.split()
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                pts = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    pts.append([x, y])

                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                color = colors[cls_id % len(colors)] if class_names else (0, 255, 0)

                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

                label_name = class_names[cls_id] if cls_id < len(class_names) else f"id {cls_id}"
                cv2.putText(img, label_name, (pts[0][0][0], pts[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "No label file found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        filename = os.path.basename(image_path)
        cv2.setWindowTitle("YOLO Segmentation Preview", f"{filename}  ({index + 1}/{len(image_files)})")
        cv2.imshow("YOLO Segmentation Preview", img)

        key = cv2.waitKeyEx(0)

        if key in [27, ord('q')]:  # ESC or Q = exit
            break
        elif key in [2424832, ord('a')]:  # ← or A
            index = (index - 1) % len(image_files)
        elif key in [2555904, ord('d')]:  # → or D
            index = (index + 1) % len(image_files)

        if cv2.getWindowProperty("YOLO Segmentation Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# ====== GUI ======
def browse_images_folder():
    folder = filedialog.askdirectory(title="Select folder with images (images/)")
    if folder:
        images_path_var.set(folder)

def browse_labels_folder():
    folder = filedialog.askdirectory(title="Select folder with labels (labels/)")
    if folder:
        labels_path_var.set(folder)

def browse_classes_file():
    file_path = filedialog.askopenfilename(title="Select classes.txt file", filetypes=[("Text files", "*.txt")])
    if file_path:
        classes_path_var.set(file_path)

def start_viewer():
    images_dir = images_path_var.get()
    labels_dir = labels_path_var.get()
    classes_file = classes_path_var.get()

    if not images_dir or not labels_dir or not classes_file:
        messagebox.showerror("Error", "Please select all three paths!")
        return

    visualize_yolo(images_dir, labels_dir, classes_file)


# ====== MAIN WINDOW ======
root = Tk()
root.title("YOLO Segmentation – Label Preview")
root.geometry("550x400")
root.resizable(False, False)

# --- Colors ---
BG_MAIN = "#1B1B1B"       # dark background
BG_ENTRY = "#2A2A2A"      # entry box background
FG_TEXT = "#FFFFFF"       # white text
FG_LABEL = "#A0522D"      # brown labels
BTN_GREEN = "#4CAF50"     # green button
BTN_GREEN_HOVER = "#3E8E41"

root.configure(bg=BG_MAIN)
font_main = ("Segoe UI", 11)

def make_button(master, text, cmd, color=BTN_GREEN):
    b = Button(master, text=text, command=cmd,
               bg=color, fg=FG_TEXT,
               activebackground=BTN_GREEN_HOVER,
               activeforeground=FG_TEXT,
               relief="flat", padx=10, pady=5)
    return b

# --- GUI Elements ---
Label(root, text="📂 Images folder (images/):", font=font_main, bg=BG_MAIN, fg=FG_LABEL).pack(pady=(15, 5))
images_path_var = StringVar()
Entry(root, textvariable=images_path_var, width=60, bg=BG_ENTRY, fg=FG_TEXT, insertbackground=FG_TEXT,
      relief="flat").pack()
make_button(root, "Browse folder", browse_images_folder).pack(pady=5)

Label(root, text="🏷️ Labels folder (labels/):", font=font_main, bg=BG_MAIN, fg=FG_LABEL).pack(pady=(15, 5))
labels_path_var = StringVar()
Entry(root, textvariable=labels_path_var, width=60, bg=BG_ENTRY, fg=FG_TEXT, insertbackground=FG_TEXT,
      relief="flat").pack()
make_button(root, "Browse folder", browse_labels_folder).pack(pady=5)

Label(root, text="📘 Classes file (classes.txt):", font=font_main, bg=BG_MAIN, fg=FG_LABEL).pack(pady=(15, 5))
classes_path_var = StringVar()
Entry(root, textvariable=classes_path_var, width=60, bg=BG_ENTRY, fg=FG_TEXT, insertbackground=FG_TEXT,
      relief="flat").pack()
make_button(root, "Browse file", browse_classes_file).pack(pady=5)

make_button(root, "👁️  Show preview", start_viewer, color=BTN_GREEN).pack(pady=25)

Label(root, text="Controls: ← previous | → next | ESC or Q to exit",
      font=("Segoe UI", 9), bg=BG_MAIN, fg="#C0C0C0").pack()

root.mainloop()
