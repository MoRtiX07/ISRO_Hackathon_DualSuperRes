import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import os

def build_srcnn():
    model = Sequential()
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (5, 5), activation='linear', padding='same'))
    return model

# === Download pretrained weights ===
#weights_url = "https://github.com/Arhamshah/SRCNN/releases/download/v1.0/srcnn_weights.h5"
weights_path = "srcnn_weights.h5"

'''if not os.path.exists(weights_path):
    import urllib.request
    print("📥 Downloading pretrained weights...")
    urllib.request.urlretrieve(weights_url, weights_path)
    print("✅ Download complete!")'''

# === Load and preprocess image ===
img = cv2.imread("../data/raw/image1.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("⚠️ Image not found at ../data/raw/image1.jpg")

# Simulate low-res by downsampling then upsampling
low_res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
upscaled = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

# Normalize and reshape
input_img = upscaled.astype(np.float32) / 255.0
input_img = np.expand_dims(np.expand_dims(input_img, axis=0), axis=-1)

# Load model and weights
model = build_srcnn()
model.load_weights(weights_path)

# Run prediction
predicted = model.predict(input_img)
predicted_img = np.squeeze(predicted, axis=0)
predicted_img = np.squeeze(predicted_img, axis=-1)
predicted_img = np.clip(predicted_img * 255.0, 0, 255).astype(np.uint8)

# Save output
output_dir = "../outputs/super_resolved"
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "image1_srcnn.png"), predicted_img)

print("✅ Super-resolved image saved at:", os.path.join(output_dir, "image1_srcnn.png"))
