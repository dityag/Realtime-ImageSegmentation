import cv2
import numpy as np
import os

# === Mapping grayscale class value to RGB colors ===
# Ubah sesuai dengan jumlah class Anda dan preferensi warna
color_map = {
    0: (0, 0, 0),         # background - black
    1: (255, 0, 0),       # class 1 - red
    2: (0, 255, 0),       # class 2 - green
    3: (0, 0, 255),       # class 3 - blue
    4: (255, 255, 0),     # class 4 - cyan
    5: (255, 0, 255),     # class 5 - magenta
    6: (0, 255, 255),     # class 6 - yellow
    7: (128, 0, 128),     # class 7 - purple
    8: (128, 128, 128)    # class 8 - gray
}

# Folder annotasi
annot_folders = ["trainannot", "validannot", "testannot"]

for folder in annot_folders:
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder_path):
        print(f"Folder '{folder}' tidak ditemukan, dilewati.")
        continue

    out_folder = os.path.join(folder_path + "_rgb")
    os.makedirs(out_folder, exist_ok=True)

    print(f"Memproses: {folder}")

    for fn in os.listdir(folder_path):
        if not fn.endswith(".png"):
            continue

        gray_path = os.path.join(folder_path, fn)
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_val, color in color_map.items():
            rgb[gray == class_val] = color

        rgb_path = os.path.join(out_folder, fn)
        cv2.imwrite(rgb_path, rgb)
        print(f" -> {rgb_path}")

print("Selesai mengonversi semua grayscale ke RGB.")

