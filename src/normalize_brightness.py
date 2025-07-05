import cv2
import os

# Load and convert image to grayscale
img = cv2.imread("../data/raw/image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram equalization for brightness normalization
equalized = cv2.equalizeHist(gray)

# Save output
os.makedirs("../outputs/brightness_fixed", exist_ok=True)
cv2.imwrite("../outputs/brightness_fixed/image1_brightness_equalized.png", equalized)

print("✅ Brightness-normalized image saved at: outputs/brightness_fixed/image1_brightness_equalized.png")
