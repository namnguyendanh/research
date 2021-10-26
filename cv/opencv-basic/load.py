import numpy as np
import time

import cv2


# cv2.namedWindow("frame")


img = cv2.imread("data/kmk.jpg")

print(img.shape[0], img.shape[1])
gray_img = np.zeros((img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        print(img[i][j])

cv2.imwrite("data/kmk_transform.jpg", gray_img)
