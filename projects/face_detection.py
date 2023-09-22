import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the path of the image
path = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\my-passport-photo.jpg"
img = cv.imread(path)
cv.imshow("original image",img)

# convert into the grayScale image
grayScale = cv.cvtColor(img,cv.COLOR_BGR2GRAY,cv.BORDER_DEFAULT)
cv.imshow("GrayScale",grayScale)

# Reading the haar_cascade path
cascade_path = "Fundamentals-of-Computer-Vision-using-openCV\haarcascade.xml"
haar_cascade = cv.CascadeClassifier(cascade_path)

# face_rectangle coordinates
face_rect = haar_cascade.detectMultiScale(grayScale,1.1,3)
print(f'The # of faces found:{len(face_rect)}')

# lets draw the rectangle over the face
for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# show the image result
cv.imshow("Detected image",img)


cv.waitKey(0)