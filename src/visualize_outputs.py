import cv2
import matplotlib.pyplot as plt

bicubic = cv2.imread("../outputs/super_resolved/image1_bicubic.png")
edsr = cv2.imread("../outputs/super_resolved/image1_edsr.png")
pipeline = cv2.imread("../outputs/super_resolved/image1_pipeline_edsr.png")

# Convert BGR to RGB for display
bicubic = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
edsr = cv2.cvtColor(edsr, cv2.COLOR_BGR2RGB)
pipeline = cv2.cvtColor(pipeline, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(bicubic)
plt.title("🌀 Bicubic")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edsr)
plt.title("⚡ EDSR")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pipeline)
plt.title("🛠️ Full Pipeline")
plt.axis("off")

plt.tight_layout()
plt.savefig("../outputs/comparison_grid.png")
plt.show()
