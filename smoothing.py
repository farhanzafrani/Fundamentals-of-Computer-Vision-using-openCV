import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image
path = "image_processing_files\house.jpg"
img = cv.imread(path)
cv.imshow("Original image",img)

# Average blurr
average_blur = cv.blur(img,(5,5),cv.BORDER_DEFAULT)
cv.imshow("Average Blur",average_blur)

#Gaussian blur
gauss_blur = cv.GaussianBlur(img,(5,5),5,cv.BORDER_DEFAULT)
cv.imshow("Gaussian Blur",gauss_blur)

# Median blurr
median_blur = cv.medianBlur(img,3,cv.BORDER_DEFAULT)
cv.imshow("Median Blur",median_blur)

# bilaterail filter (bilateral blur)
bilater_blur = cv.bilateralFilter(img,15,15,15,cv.BORDER_DEFAULT)
cv.imshow("Bilateral Blur",bilater_blur)


cv.waitKey(0)