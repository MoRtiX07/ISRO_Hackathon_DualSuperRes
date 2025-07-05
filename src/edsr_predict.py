import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained EDSR model from TensorFlow Hub
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

# Load and preprocess image
img = cv2.imread("../data/raw/image1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))  # small input

lr = img.astype(np.float32)
lr = lr / 255.0
lr = tf.convert_to_tensor(lr)
lr = tf.expand_dims(lr, axis=0)

# Super resolve
sr = model(lr)
sr = tf.clip_by_value(sr, 0.0, 1.0)
sr_img = tf.squeeze(sr).numpy()
sr_img = (sr_img * 255.0).astype(np.uint8)

# Show and save
plt.imsave("../outputs/super_resolved/image1_edsr.png", sr_img)
print("✅ Super-resolved image saved as: image1_edsr.png")
