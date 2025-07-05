import cv2
import numpy as np
import os

# === Step 1: Load Image ===
img_path = "../data/raw/image1.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError("❌ Image not found at: " + img_path)

# === Step 2: Convert to HSV ===
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_channel = hsv[:, :, 2]

# === Step 3: Detect Shadows (dark pixels) ===
shadow_mask = v_channel < 60  # threshold on brightness
cv2.imwrite("../outputs/shadow_masks/image1_shadow_mask.png", shadow_mask.astype(np.uint8) * 255)

# === Step 4: Brightness Normalization using CLAHE ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
bright_fixed = clahe.apply(gray)
cv2.imwrite("../outputs/brightness_fixed/image1_brightness_fixed.png", bright_fixed)

print("✅ Shadow mask and brightness-normalized image saved successfully.")
