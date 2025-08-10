
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Define control points for piecewise linear transformation
control_points = np.array([(50, 50), (50, 100), (150, 255), (150, 150), (255, 255)])

# Create transformation segments
segment1 = np.linspace(0, control_points[0, 1], control_points[0, 0] + 1 - 0).astype('uint8')
segment2 = np.linspace(control_points[0, 1] + 1, control_points[1, 1], control_points[1, 0] - control_points[0, 0]).astype('uint8')
segment3 = np.linspace(control_points[1, 1] + 1, control_points[2, 1], control_points[2, 0] - control_points[1, 0]).astype('uint8')
segment4 = np.linspace(control_points[2, 1] + 1, control_points[3, 1], control_points[3, 0] - control_points[2, 0]).astype('uint8')
segment5 = np.linspace(control_points[3, 1] + 1, control_points[4, 1], control_points[4, 0] - control_points[3, 0]).astype('uint8')

# Build complete transformation lookup table
lut = np.concatenate((segment1, segment2), axis=0).astype('uint8')
lut = np.concatenate((lut, segment3), axis=0).astype('uint8')
lut = np.concatenate((lut, segment4), axis=0).astype('uint8')
lut = np.concatenate((lut, segment5), axis=0).astype('uint8')

plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.plot(lut)
plt.title("Transformation Curve")
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.xticks([0, 50, 100, 150, 200, 250])  # Set specific x-axis ticks
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')

original_image = cv.imread('emma.jpg', cv.IMREAD_GRAYSCALE)

plt.subplot(132)
plt.imshow(original_image, cmap="gray")
plt.title('Original')
plt.axis('off')

transformed_image = cv.LUT(original_image, lut)

plt.subplot(133)
plt.imshow(transformed_image, cmap="gray")
plt.title('After Applying Intensity Transformation')
plt.axis('off')

plt.show()