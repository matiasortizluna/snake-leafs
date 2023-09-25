import cv2
import numpy as np

image = cv2.imread('plantvillage/color/Tomato___Target_Spot/0a610d40-f5b8-4580-8e32-cc90b3620017___Com.G_TgS_FL 8191.JPG', -1)
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, roi_corners, ignore_mask_color)

masked_image = cv2.bitwise_and(image, mask)

cv2.imwrite('image_masked.png', masked_image)