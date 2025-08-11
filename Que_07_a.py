import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_with_filter2d(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Apply Sobel filters using filter2D
    grad_x = cv2.filter2D(img, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, sobel_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grad_x, cmap='gray')
    plt.title('Sobel X (Vertical Edges)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(grad_y, cmap='gray')
    plt.title('Sobel Y (Horizontal Edges)')
    plt.axis('off')
    

    plt.tight_layout()
    plt.show()
    
    return gradient_magnitude


result = sobel_with_filter2d('einstein.png')