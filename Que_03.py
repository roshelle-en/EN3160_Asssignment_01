import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_orig = cv.imread('highlights_and_shadows.jpg', cv.IMREAD_COLOR)

if img_orig is None:
    raise FileNotFoundError("Error: Could not load 'highlights_and_shadows.jpg'. Check the file path and ensure the image exists.")

img_color = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
img_lab = cv.cvtColor(img_orig, cv.COLOR_BGR2Lab)# Convert to RGB and Lab color spaces
l, a, b = cv.split(img_lab)
gamma = 0.75
table = np.array([(i/255.0)**(gamma)*255.0 for i in np.arange(0, 256)]).astype('uint8')# Apply gamma correction to L channel
l_gamma_corrected = cv.LUT(l, table)
img_gamma_merge = cv.merge((l_gamma_corrected, a, b)) # Merge corrected L channel with original a and b channels
img_corrected_color = cv.cvtColor(img_gamma_merge, cv.COLOR_Lab2RGB)

# Display original image, L channel, and gamma-corrected image in a row
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.imshow(img_color)
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(l, cmap="gray")
plt.title('L Channel of the Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_corrected_color)
plt.title('Gamma Corrected Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot histograms for L, a, b channels of original and gamma-corrected images 
plt.figure(figsize=(12, 4))
space = ('l', 'a', 'b')
color = ('orange', 'purple', 'green')  # L=orange, a=purple, b=green

plt.subplot(121)
for i, c in enumerate(space):
    hist_orig = cv.calcHist([img_lab], [i], None, [256], [0, 256])
    plt.plot(hist_orig, color=color[i], label=f'{c} channel')
plt.title('Histogram of the Original Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(122)
for i, c in enumerate(space):
    hist_gamma_merge = cv.calcHist([img_gamma_merge], [i], None, [256], [0, 256])
    plt.plot(hist_gamma_merge, color=color[i], label=f'{c} channel')
plt.title('Histogram of the Gamma Corrected Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()