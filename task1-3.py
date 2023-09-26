import cv2
import matplotlib.pyplot as plt

image = cv2.imread('green.jpg')
resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Display all four images using Matplotlib
plt.figure(figsize=(12, 8))  

# Original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized image 
plt.subplot(2, 2, 2)
plt.title('Resized Image')
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Grayscale image
plt.subplot(2, 2, 3)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# Binary image
plt.subplot(2, 2, 4)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()  
plt.show()
