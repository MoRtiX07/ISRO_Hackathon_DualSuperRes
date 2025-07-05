import cv2
import numpy as np
import os
import tensorflow_hub as hub
import tensorflow as tf

# Paths
input_path = "../data/raw/image1.jpg"
shadow_output = "../outputs/shadow_masks/image1_shadow_mask.png"
brightness_output = "../outputs/brightness_fixed/image1_brightness_equalized.png"
sr_output = "../outputs/super_resolved/image1_pipeline_edsr.png"

# 1. Load image
img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Shadow Detection
shadow_mask = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15, 10
)
os.makedirs("../outputs/shadow_masks", exist_ok=True)
cv2.imwrite(shadow_output, shadow_mask)

# 3. Brightness Normalization
equalized = cv2.equalizeHist(gray)
equalized = cv2.resize(equalized, (256, 256))  # Prevent GPU OOM
os.makedirs("../outputs/brightness_fixed", exist_ok=True)
cv2.imwrite(brightness_output, equalized)

# 4. EDSR Super-Resolution
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
input_img = tf.convert_to_tensor(equalized[..., np.newaxis], dtype=tf.float32) / 255.0  # normalize to [0, 1]
input_img = tf.expand_dims(input_img, axis=0)  # [1, H, W, 1]
input_img = tf.image.grayscale_to_rgb(input_img)  # [1, H, W, 3]


sr_img = model(input_img)
sr_img = tf.squeeze(sr_img).numpy()
sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)

os.makedirs("../outputs/super_resolved", exist_ok=True)
cv2.imwrite(sr_output, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

print("✅ Full pipeline completed and all outputs saved!")
