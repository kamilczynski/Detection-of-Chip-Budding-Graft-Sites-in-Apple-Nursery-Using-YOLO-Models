import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ---------------- IMAGE RESIZING FUNCTION ----------------
def resize_images(input_folder, output_folder, width, height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image = cv2.imread(input_path)
            if image is None:
                continue
            resized_image = cv2.resize(image, (width, height))
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)
            print(f"Rescaled: {filename} -> {output_path}")

    messagebox.showinfo("Ready", "✅ Image scaling completed!")

# ---------------- GUI FUNCTIONS ----------------
def select_input_folder():
    folder = filedialog.askdirectory(title="Select input folder")
    if folder:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, folder)

def select_output_folder():
    folder = filedialog.askdirectory(title="Select output folder")
    if folder:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, folder)

def start_processing():
    input_folder = input_entry.get()
    output_folder = output_entry.get()
    width = width_entry.get()
    height = height_entry.get()

    if not input_folder or not output_folder:
        messagebox.showerror("Error", "Please select input and output folders.")
        return

    if not width or not height:
        messagebox.showerror("Error", "Please specify both width and height.")
        return

    try:
        width = int(width)
        height = int(height)
        resize_images(input_folder, output_folder, width, height)
    except ValueError:
        messagebox.showerror("Error", "Width and height must be whole numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# ---------------- APP CONFIGURATION ----------------
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("ChipBuddingPix")
app.geometry("600x400")
app.resizable(False, False)

# 🔹 Color palette (black, white, brown, green)
BLACK = "#1A1A1A"
WHITE = "#FAFAFA"
BROWN = "#4E342E"
DARK_BROWN = "#3E2723"
GREEN = "#4CAF50"
LIGHT_GREEN = "#81C784"
CREAM = "#E8E6E1"

# 🔹 Full background
full_bg = ctk.CTkFrame(app, fg_color=BLACK, corner_radius=0)
full_bg.pack(fill="both", expand=True)

# 🔹 App title
title_label = ctk.CTkLabel(
    full_bg,
    text="ChipBudding",
    text_color=LIGHT_GREEN,
    font=("Orbitron", 26, "bold")
)
title_label.place(relx=0.5, rely=0.18, anchor="center")

# 🔹 Central panel
panel = ctk.CTkFrame(full_bg, fg_color=DARK_BROWN, corner_radius=20)
panel.place(relx=0.5, rely=0.55, anchor="center")

font = ("Orbitron", 13)

# ---------------- INPUTS & BUTTONS ----------------
input_entry = ctk.CTkEntry(
    master=panel,
    width=250,
    placeholder_text="Input folder",
    fg_color=BLACK,
    text_color=WHITE,
    border_color=GREEN,
    font=font
)
input_entry.grid(row=0, column=0, padx=10, pady=5)

input_button = ctk.CTkButton(
    master=panel,
    text="Select",
    command=select_input_folder,
    fg_color=GREEN,
    text_color=BLACK,
    hover_color=LIGHT_GREEN,
    corner_radius=8,
    border_width=2,
    border_color=LIGHT_GREEN,
    bg_color=DARK_BROWN,
    font=("Orbitron", 13, "bold"),
    width=80
)
input_button.grid(row=0, column=1, padx=5, pady=5)

output_entry = ctk.CTkEntry(
    master=panel,
    width=250,
    placeholder_text="Output folder",
    fg_color=BLACK,
    text_color=WHITE,
    border_color=GREEN,
    font=font
)
output_entry.grid(row=1, column=0, padx=10, pady=5)

output_button = ctk.CTkButton(
    master=panel,
    text="Select",
    command=select_output_folder,
    fg_color=GREEN,
    text_color=BLACK,
    hover_color=LIGHT_GREEN,
    corner_radius=8,
    border_width=2,
    border_color=LIGHT_GREEN,
    bg_color=DARK_BROWN,
    font=("Orbitron", 13, "bold"),
    width=80
)
output_button.grid(row=1, column=1, padx=5, pady=5)

# 🔸 Dimension fields (brown + green accent)
size_frame = ctk.CTkFrame(panel, fg_color=DARK_BROWN)
size_frame.grid(row=2, column=0, columnspan=2, pady=10)

width_entry = ctk.CTkEntry(
    master=size_frame,
    width=90,
    placeholder_text="Width",
    fg_color=BLACK,
    text_color=WHITE,
    border_color=LIGHT_GREEN,
    font=font
)
width_entry.grid(row=0, column=0, padx=10)

height_entry = ctk.CTkEntry(
    master=size_frame,
    width=90,
    placeholder_text="Height",
    fg_color=BLACK,
    text_color=WHITE,
    border_color=LIGHT_GREEN,
    font=font
)
height_entry.grid(row=0, column=1, padx=10)

# 🔹 Start button (green with cream accent)
start_button = ctk.CTkButton(
    master=panel,
    text="Start scaling",
    command=start_processing,
    fg_color=GREEN,
    text_color=WHITE,
    hover_color=LIGHT_GREEN,
    corner_radius=8,
    border_width=2,
    border_color=CREAM,
    bg_color=DARK_BROWN,
    font=("Orbitron", 13, "bold"),
    width=160
)
start_button.grid(row=3, column=0, columnspan=2, pady=10)

# ---------------- RUN APP ----------------
app.mainloop()

