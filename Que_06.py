import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Read the image
image = cv.imread('jeniffer.jpg', cv.IMREAD_COLOR)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Convert the image to HSV color space and split into channels
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)       
hue_channel, saturation_channel, value_channel = cv.split(hsv_image)     

# Display the individual HSV channels
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(hue_channel, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(132)
plt.imshow(saturation_channel, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(133)
plt.imshow(value_channel, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.tight_layout()
plt.show()

# Apply thresholding on the value channel to extract foreground
value_threshold = 80
_, foreground_mask = cv.threshold(value_channel, value_threshold, 255, cv.THRESH_BINARY)

# Extract the foreground from the original image using the mask
foreground_image = cv.bitwise_and(image, image, mask=foreground_mask)

# Convert the extracted foreground to grayscale and compute its histogram
foreground_gray = cv.cvtColor(foreground_image, cv.COLOR_BGR2GRAY)
foreground_histogram = cv.calcHist([foreground_gray], [0], None, [256], [0, 256])

# Compute the cumulative histogram
cumulative_foreground_hist = np.cumsum(foreground_histogram)

# Display the foreground mask, histogram, and cumulative histogram
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask From Value Channel')
plt.axis('off')

plt.subplot(132)
plt.plot(foreground_histogram, color='black')
plt.title('Foreground Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(133)
plt.plot(cumulative_foreground_hist, color='black')
plt.title('Cumulative Foreground Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Perform histogram equalization on each color channel
r_channel_eq = cv.equalizeHist(foreground_image[:, :, 0])
g_channel_eq = cv.equalizeHist(foreground_image[:, :, 1])
b_channel_eq = cv.equalizeHist(foreground_image[:, :, 2])

# Merge the equalized channels back together
equalized_image = cv.merge((r_channel_eq, g_channel_eq, b_channel_eq))

# Extract the background by inverting the foreground mask
background_image = cv.bitwise_and(image, image, mask=cv.bitwise_not(foreground_mask))

# Combine the background and the equalized foreground
final_image = cv.add(background_image, equalized_image)
final_image_rgb = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)

# Display the original and final images
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(final_image_rgb)
plt.title("Foreground Equalized Image")
plt.axis('off')

plt.tight_layout()
plt.show()
