import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = r"image_processing_files\road.jpg"
img = cv.imread(path)
cv.imshow('Original_image',img)

# change into grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale',gray)

# Blur the image
blur = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
# cv.imshow('Blur',blur)

# edge detection canny
canny = cv.Canny(img,200,205)
# cv.imshow('Canny',canny)

# erode the image
eroded = cv.erode(canny,(3,3),iterations=2)
# cv.imshow("Eroded",eroded)

# dilation of the image
dilated = cv.dilate(canny,(5,5),iterations=2)
# cv.imshow("dilated_image",dilated)
# print(img.shape[:2])

cv.waitKey(0)