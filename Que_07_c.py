import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

input_image = cv.imread('einstein.png', cv.IMREAD_GRAYSCALE)

vertical_kernel = np.array([(1), (2), (1)], dtype='float')   
horizontal_kernel = np.array([(1, 0, -1)], dtype='float')

vertical_convolution = cv.filter2D(input_image, -1, vertical_kernel)    
edge_detected_image = cv.filter2D(vertical_convolution, -1, horizontal_kernel)    

fig, axes  = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(12, 12))

axes[0].imshow(input_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])

axes[1].imshow(edge_detected_image, cmap='gray')
axes[1].set_title('Sobel Filtered Image')
axes[1].set_xticks([]), axes[1].set_yticks([])

plt.show()
