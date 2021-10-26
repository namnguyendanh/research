import time

import cv2
import matplotlib.pyplot as plt
import os


# # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# img = cv2.imread("data/training_data/3/1.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# print(img)
#
# plt.imshow(img)
# plt.show()
#
#
for file in os.listdir("data/training_data/1")[:10]:
    file_url = "data/training_data/1/" + file
    print(file_url)
    img = cv2.imread(file_url)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.imshow(img)
    time.sleep(1)
plt.show()
