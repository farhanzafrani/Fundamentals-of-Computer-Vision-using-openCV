import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image
path = r"image_processing_files\road.jpg"
img = cv.imread(path)
cv.imshow("original image",img)

# lets create a blank image
blank = np.zeros(img.shape[:],dtype='uint8')

# lets split the image into r,g,b
(b,g,r) = cv.split(img)

# cv.imshow("blue",b)
# cv.imshow("green",g)
# cv.imshow("red",r)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow("Blue",blue)
cv.imshow("Green",green)
cv.imshow("Red",red)

cv.waitKey(0)