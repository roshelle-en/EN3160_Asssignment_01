import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(f):

    M, N = f.shape
    L = 256
    t = np.zeros(256, dtype=np.uint8)
    hist, bins = np.histogram(f.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    t = np.array([(L-1)/(M*N)*cdf[i] for i in range(256)], dtype=np.uint8)
    g = t[f]
    return g

f = cv.imread('shells.tif', cv.IMREAD_GRAYSCALE)


g = histogram_equalization(f) # Apply histogram equalization

# Display results
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(g, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Histogram Equalization')
ax[1].axis('off')
plt.show()

hist_original = cv.calcHist([f], [0], None, [256], [0, 256])
hist_equalized = cv.calcHist([g], [0], None, [256], [0, 256])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(hist_original)
ax[0].set_title('Histogram of Original')
ax[0].set_xlabel('Intensity')
ax[0].set_ylabel('Frequency')
ax[1].plot(hist_equalized)
ax[1].set_title('Histogram of Equalized Image')
ax[1].set_xlabel('Intensity')
ax[1].set_ylabel('Frequency')
plt.show()