import cv2 as cv
import numpy as np

path = "image_processing_files\house.jpg"
img = cv.imread(path)
cv.imshow("original_image",img)

# grayscale image
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("grayScale",gray)

#blurring the image
blur = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
# cv.imshow("blurr",blur)

# canny edge detection
canny = cv.Canny(blur,125,175)
# cv.imshow("canny",canny)

contours, hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours in the image.')


blank = np.zeros(img.shape,dtype='uint8')
cv.drawContours(blank,contours,-1,color=(0,0,255),thickness=1)
cv.imshow("contours",blank)




cv.waitKey(0)