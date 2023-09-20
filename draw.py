import cv2 as cv
import numpy as np

# make black image
blank = np.zeros((500,500,3),dtype='uint8')

# accessing the red channel
# blank[200:250,200:250] = (0,0,255)

# drawing rectangle
# cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]),color=(0,255,0),thickness=cv.FILLED)

# drawing circle
# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),60,color=(255,0,0),thickness=cv.FILLED)

# drawing line 
# cv.line(blank,(0,250),(250,250),color=(0,0,255),thickness=4)
# cv.line(blank,(250,0),(250,250),color=(0,0,255),thickness=4)
# cv.line(blank,(500,250),(250,250),color=(0,0,255),thickness=4)
# cv.line(blank,(250,500),(250,250),color=(0,0,255),thickness=4)

# drawing Text
cv.putText(blank,"Muhammad Farhan",(70,250),cv.FONT_HERSHEY_TRIPLEX,1.0,color=(200,250,0),thickness=2)
# show the blank image
cv.imshow('image',blank)


cv.waitKey(0)