import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

spider = cv.imread('spider.png')
if spider is None:
    raise FileNotFoundError("Error: Could not load 'spider.png'. Check the file path and ensure the image exists.")

spider_rgb = cv.cvtColor(spider, cv.COLOR_BGR2RGB)
Hue, Saturation, Value = cv.split(cv.cvtColor(spider, cv.COLOR_BGR2HSV))

fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax[0].imshow(Hue, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Hue Plane')
ax[0].axis("off")
ax[1].imshow(Saturation, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Saturation Plane')
ax[1].axis("off")
ax[2].imshow(Value, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Value Plane')
ax[2].axis("off")
plt.tight_layout()
plt.show()

# Transformation function for Saturation channel
a = 0.65
sigma = 70.0 
x = np.arange(0, 256)
function = np.minimum(x + a * 128 * np.exp(-((x - 128)**2) / (2 * sigma**2)), 255).astype('uint8')
S_transformed = cv.LUT(Saturation, function)
spider_transformed = cv.cvtColor(cv.merge([Hue, S_transformed, Value]), cv.COLOR_HSV2RGB)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(x, function)
axs[0].set_title('Intensity Transformation curve with a = 0.65')
axs[0].set_xlabel('Input Intensity')
axs[0].set_ylabel('Output Intensity')
axs[0].grid(True)
axs[0].set_xlim([0, 255])
axs[0].set_ylim([0, 255])

# Original image
axs[1].imshow(spider_rgb)
axs[1].set_title('Original')
axs[1].axis('off')

# Vibrance-modified image
axs[2].imshow(spider_transformed)
axs[2].set_title('Vibrance Modified')
axs[2].axis('off')

plt.tight_layout()
plt.show()