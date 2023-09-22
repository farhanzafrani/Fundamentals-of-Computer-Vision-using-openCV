import cv2 as cv
import numpy as np



path = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\road.jpg"

img = cv.imread(path)
cv.imshow("original image",img)

print(img.shape)
blank = np.zeros((img.shape[:2]),dtype='uint8')
cv.imshow("blank image",blank)

circle = cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),100,255,thickness=-1)
cv.imshow("mask",circle)

masked_image = cv.bitwise_and(img,img,mask=circle)

cv.imshow("masked image",masked_image)

cv.bitwise_and()
cv.waitKey(0)