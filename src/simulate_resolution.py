from PIL import Image
import matplotlib.pyplot as plt

# Load high-resolution image
img_path = "../data/raw/image1.jpg"
img = Image.open(img_path)

# Step 1: Downscale to simulate low-resolution
low_res = img.resize((64, 64), Image.BICUBIC)

# Step 2: Upscale back to original size
upscaled = low_res.resize(img.size, Image.BICUBIC)

# Step 3: Display comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original (High-Res)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(low_res)
plt.title("Low-Res (64x64)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(upscaled)
plt.title("Upscaled (Blurry)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the upscaled (blurry) image
upscaled.save("../outputs/super_resolved/image1_bicubic.png")
