import cv2
import os

# Load image
img = cv2.imread("../data/raw/image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding to detect shadows
shadow_mask = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15, 10
)

# Save shadow mask
os.makedirs("../outputs/shadow_masks", exist_ok=True)
cv2.imwrite("../outputs/shadow_masks/image1_shadow_mask.png", shadow_mask)

print("✅ Shadow mask saved at: outputs/shadow_masks/image1_shadow_mask.png")
