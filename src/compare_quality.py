import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load images
original = cv2.imread("../data/raw/image1.jpg")
edsr = cv2.imread("../outputs/super_resolved/image1_edsr.png")
bicubic = cv2.resize(cv2.resize(original, (64, 64), interpolation=cv2.INTER_CUBIC), (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)

# Convert BGR to RGB
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
edsr = cv2.cvtColor(edsr, cv2.COLOR_BGR2RGB)
bicubic = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)

edsr = cv2.resize(edsr, (original.shape[1], original.shape[0]))


# Compute PSNR and SSIM
print("📊 Quality Comparison with Original Image")
print("----------------------------------------")
print(f"🌀 Bicubic PSNR: {psnr(original, bicubic):.2f} dB")
print(f"🌀 Bicubic SSIM: {ssim(original, bicubic, channel_axis=2):.4f}")
print(f"⚡ EDSR PSNR:    {psnr(original, edsr):.2f} dB")
print(f"⚡ EDSR SSIM:    {ssim(original, edsr, channel_axis=2):.4f}")
