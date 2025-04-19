import json
import numpy as np
import cv2
import os

# Load COCO JSON
with open("_annotations.coco.json", "r") as f:
    coco_data = json.load(f)

# Create output directory for segmentation masks
os.makedirs("segmentation_masks", exist_ok=True)

# Get category mapping (ID -> grayscale value)
category_mapping = {category["id"]: idx + 1 for idx, category in enumerate(coco_data["categories"])}

# Process each image
for image in coco_data["images"]:
    img_width, img_height = image["width"], image["height"]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Find annotations for the image
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] == image["id"]:
            category_id = annotation["category_id"]
            segmentation = annotation["segmentation"]
            mask_value = category_mapping.get(category_id, 0)  # Assign grayscale value

            # Draw segmentation on mask
            for segment in segmentation:
                pts = np.array(segment, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], mask_value)

    # Save mask as PNG
    mask_filename = os.path.splitext(image["file_name"])[0] + ".png"
    mask_path = os.path.join("segmentation_masks", mask_filename)
    cv2.imwrite(mask_path, mask)

print("Segmentation masks saved in 'segmentation_masks' directory.")
