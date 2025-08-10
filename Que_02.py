import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

original_image = cv.imread('brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)

# Define control points for white matter accentuation
cp_wm = np.array([(0, 0), (180, 0), (180, 200), (255, 200)])
x = np.arange(256)
lut_wm = np.interp(x, cp_wm[:, 0], cp_wm[:, 1]).astype('uint8')
transformed_wm = cv.LUT(original_image, lut_wm)

# Define control points for grey matter accentuation
cp_gm = np.array([(0, 0), (140, 0), (140, 200), (180, 200), (180, 0), (255, 0)])
lut_gm = np.interp(x, cp_gm[:, 0], cp_gm[:, 1]).astype('uint8')
transformed_gm = cv.LUT(original_image, lut_gm)


plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x, lut_wm)
plt.title("White Matter Accentuation Curve")
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')

plt.subplot(122)
plt.imshow(transformed_wm, cmap="gray")
plt.title('Transformed Image - White Matter')
plt.axis('off')

plt.show()


plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x, lut_gm)
plt.title("Grey Matter Accentuation Curve")
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.xticks([0, 50, 100, 150, 200, 250])
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')

plt.subplot(122)
plt.imshow(transformed_gm, cmap="gray")
plt.title('Transformed Image - Grey Matter')
plt.axis('off')

plt.show()