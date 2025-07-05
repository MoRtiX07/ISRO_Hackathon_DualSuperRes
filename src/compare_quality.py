from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

print("📊 Quality Comparison with Original Image")
print("----------------------------------------")

original = cv2.imread("data/raw/image1.jpg", cv2.IMREAD_GRAYSCALE)
bicubic = cv2.imread("outputs/super_resolved/image1_bicubic.png", cv2.IMREAD_GRAYSCALE)
edsr = cv2.imread("outputs/super_resolved/image1_edsr.png", cv2.IMREAD_GRAYSCALE)
pipeline = cv2.imread("outputs/super_resolved/image1_pipeline_edsr.png", cv2.IMREAD_GRAYSCALE)

# Validate
for name, img in [("original", original), ("bicubic", bicubic), ("edsr", edsr), ("pipeline", pipeline)]:
    if img is None:
        raise FileNotFoundError(f"❌ Could not load {name} image.")

# Resize all to match original size
target_size = (original.shape[1], original.shape[0])
bicubic = cv2.resize(bicubic, target_size)
edsr = cv2.resize(edsr, target_size)
pipeline = cv2.resize(pipeline, target_size)

print(f"🌀 Bicubic PSNR:  {psnr(original, bicubic):.2f} dB")
print(f"🌀 Bicubic SSIM:  {ssim(original, bicubic):.4f}")

print(f"⚡ EDSR PSNR:     {psnr(original, edsr):.2f} dB")
print(f"⚡ EDSR SSIM:     {ssim(original, edsr):.4f}")

print(f"🛰️ Full Pipeline PSNR:  {psnr(original, pipeline):.2f} dB")
print(f"🛰️ Full Pipeline SSIM:  {ssim(original, pipeline):.4f}")
