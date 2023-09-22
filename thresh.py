import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\house.jpg"

img = cv.imread(path)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY,cv.BORDER_DEFAULT)
cv.imshow("original image",img)
cv.imshow("grayscale image",gray)


# simple threshoding
threshold, thresh = cv.threshold(gray,155,255,cv.THRESH_BINARY)
cv.imshow("thresholded",thresh)

# print(threshold)

# cv.threshold_binary is the type of the thresholding type based on which the thresholdin is performed.
# There are two types of the thresholding.
# 1. simple thresholding
# 2. addaptive thresholding

adaptive_threshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,21,2)
cv.imshow("Addaptive Threshold",adaptive_threshold)



cv.waitKey(0)