import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


path = "image_processing_files\house.jpg"
img = cv.imread(path)
cv.imshow("original_image",img)

# convert BGR to Grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Grayscale",gray)

# convert BGR to HSV
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow('HSV',hsv)

# convert BGR to LAB
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
# cv.imshow("Lab",lab)


# convert BGR to RGB
rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow("RGB",rgb)

# lets read the image from Opencv and Matplotlib
plt.imshow(img)
plt.show()



cv.waitKey(0)