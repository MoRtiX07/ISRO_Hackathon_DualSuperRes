import matplotlib.pyplot as plt
import cv2

# Use full or adjusted paths
bicubic = cv2.imread("outputs/super_resolved/image1_bicubic.png")
edsr = cv2.imread("outputs/super_resolved/image1_edsr.png")
pipeline = cv2.imread("outputs/super_resolved/image1_pipeline_edsr.png")

titles = ['Bicubic', 'EDSR', 'Pipeline (Final)']

plt.figure(figsize=(12, 4))
for i, img in enumerate([bicubic, edsr, pipeline]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/super_resolved/comparison_grid.png")
plt.show()
