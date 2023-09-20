import cv2 as cv
import numpy as np


blank = np.zeros((400,400),dtype='uint8')

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),color=255,thickness=-1)
circle = cv.circle(blank.copy(),(200,200),180,color=255,thickness=-1)

# applying bitwise operation on the rectangle and circle object
intersection = cv.bitwise_and(rectangle,circle)
union = cv.bitwise_or(rectangle,circle)

cv.imshow("rectangle",rectangle)
cv.imshow("circle",circle)
cv.imshow("intersection",intersection)
cv.imshow("union",union)

cv.waitKey(0)