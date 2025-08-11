import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def upscale_image(image, method, factor=4):
    """
    Function to upscale the image using either nearest neighbor or bilinear interpolation
    """
    if method == 'nearest':
        return cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_NEAREST)
    elif method == 'bilinear':
        return cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_LINEAR)
    else:
        raise ValueError("Invalid interpolation method. Use 'nearest' or 'bilinear'.")

def compute_norm_SSD(image1, image2):
    """
    Computes the Normalized Sum of Squared Differences (SSD) between two images
    """
    if image1.shape != image2.shape:
        raise ValueError("The images must have the same dimensions.")
    return np.sum((image1 - image2)**2) / float(image1.size)


original_image = cv.imread('im02.png')
small_image = cv.imread('im02small.png')


assert original_image is not None, "Original image not found"
assert small_image is not None, "Small image not found"


upscaled_nn = upscale_image(small_image, method='nearest')
upscaled_bilinear = upscale_image(small_image, method='bilinear')


ssd_nn = compute_norm_SSD(original_image, upscaled_nn)
ssd_bilinear = compute_norm_SSD(original_image, upscaled_bilinear)

# Print SSD results
print(f"Normalized SSD for Nearest Neighbor interpolation: {ssd_nn:.3f}")
print(f"Normalized SSD for Bilinear interpolation: {ssd_bilinear:.3f}")


fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].imshow(cv.cvtColor(upscaled_nn, cv.COLOR_BGR2RGB))
axes[0].set_title('Nearest Neighbor')
axes[0].axis('off') 

axes[1].imshow(cv.cvtColor(upscaled_bilinear, cv.COLOR_BGR2RGB))
axes[1].set_title('Bilinear Interpolation')
axes[1].axis('off')

plt.tight_layout()
plt.show()
