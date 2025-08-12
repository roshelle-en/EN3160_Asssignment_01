import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('daisy.jpg')
if img is None:
    raise ValueError("Image not found. Please provide the correct path.")

# Define a bounding rectangle for GrabCut (x, y, width, height)

rect = (50, 50, 550, 500)  # Tune these coordinates to enclose the flower


mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Run GrabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
segmentation_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') #0=background, 1=foreground
foreground = img * segmentation_mask[:, :, np.newaxis]  # Keep foreground pixels
background = img * (1 - segmentation_mask[:, :, np.newaxis])  # Keep background pixels

blurred_bg = cv2.GaussianBlur(img, (19, 19), 0) #Apply Gaussian blur


enhanced = blurred_bg.copy()

enhanced[segmentation_mask == 1] = img[segmentation_mask == 1]

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title('Segmentation Mask')
plt.imshow(segmentation_mask * 255, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('Foreground Image')
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(133)
plt.title('Background Image')
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.title('Enhanced Image (Blurred Background)')
plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

