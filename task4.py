import cv2
import numpy as np

# segmentation
image = cv2.imread('coin.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and count coins based on size
min_coin_area = 100  
coin_count = 0
for contour in contours:
    if cv2.contourArea(contour) > min_coin_area:
        coin_count += 1
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

print(f"Total number of coins detected: {coin_count}")

# Display the result
cv2.imshow("Segmented Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
