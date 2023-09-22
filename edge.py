import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\house.jpg"
img = cv.imread(path)
cv.imshow("original image",img)

grayScale = cv.cvtColor(img,cv.COLOR_BGR2GRAY,cv.BORDER_DEFAULT)
cv.imshow("grayScale",grayScale)

# laplacian 
lap = cv.Laplacian(grayScale,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("laplacian",lap)

# sobel
sobelx = cv.Sobel(grayScale,cv.CV_64F,1,0)
# sobelx = np.uint8(np.absolute(sobelx))
sobely = cv.Sobel(grayScale,cv.CV_64F,0,1)
# sobely = np.uint8(np.absolute(sobely))
combine_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow("sobelx",sobelx)
cv.imshow("sobely",sobely)
cv.imshow("combine",combine_sobel)


# Canny
canny = cv.Canny(grayScale,150,175)
cv.imshow("Canny image",canny)
cv.waitKey(0)