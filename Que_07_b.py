import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_sobel_filter(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float64)
    
    
    height, width = img.shape
    
  
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
   
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighborhood = img[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(neighborhood * sobel_x)
            grad_y[i, j] = np.sum(neighborhood * sobel_y)
    
   
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 range
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    grad_x = np.clip(np.abs(grad_x), 0, 255).astype(np.uint8)
    grad_y = np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)
   
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grad_x, cmap='gray')
    plt.title('Sobel X (Custom)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(grad_y, cmap='gray')
    plt.title('Sobel Y (Custom)')
    plt.axis('off')
    
        
    plt.tight_layout()
    plt.show()
    
    return gradient_magnitude


result = custom_sobel_filter('einstein.png')