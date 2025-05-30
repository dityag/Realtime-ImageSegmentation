import json
import numpy as np
import cv2
import os

tmp_dir = os.getcwd()

subsets = {
    "train": "trainannot",
    "valid": "validannot",
    "test":  "testannot",
}

# Prioritas class (semakin tinggi, semakin dominan)
priority_order = {
    "water": 1,
    "land": 2,
    "tree": 3,
    "sky": 4,
    "building": 5,
    "obstacle": 6,
    "ship": 7,
    "objects-VPsl": 8
}

for subset_name, annot_folder in subsets.items():
    img_folder = os.path.join(tmp_dir, subset_name)
    out_folder = os.path.join(tmp_dir, annot_folder)
    json_path = os.path.join(img_folder, "_annotations.coco.json")

    if not os.path.isdir(img_folder) or not os.path.exists(json_path):
        print(f"Melewati {subset_name} (folder/json tidak ditemukan).")
        continue

    os.makedirs(out_folder, exist_ok=True)
    print(f"Memproses: {subset_name}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    category_mapping = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Buat indeks anotasi berdasarkan image_id
    annotations_per_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        cat_name = category_names.get(cat_id, "objects-VPsl")
        ann['priority'] = priority_order.get(cat_name, 0)
        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []
        annotations_per_image[img_id].append(ann)

    for img_info in coco_data['images']:
        fn = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        anns = annotations_per_image.get(img_info['id'], [])

        # Urutkan anotasi berdasarkan prioritas kelas (low → high)
        anns.sort(key=lambda x: x['priority'])

        for ann in anns:
            mask_val = category_mapping.get(ann['category_id'], 0)
            for seg in ann['segmentation']:
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], mask_val)

        out_fn = os.path.splitext(fn)[0] + ".png"
        cv2.imwrite(os.path.join(out_folder, out_fn), mask)
        print(f" -> {subset_name}: {out_fn}")

print("Selesai semua.")

