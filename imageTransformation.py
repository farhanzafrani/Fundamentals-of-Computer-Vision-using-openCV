import cv2 as cv
import numpy as np

path = "image_processing_files\house.jpg"
img = cv.imread(path)
cv.imshow("original_image",img)

# translation of the image
def translation(image,X,y):
    '''X+ --> right
        y+ --> down
        X- --> left
        y- --> upward
    '''
    transMat = np.float32([[1,0,X],
                           [0,1,y]])
    dimensions = (image.shape[1],image.shape[0])
    return cv.warpAffine(image,transMat,dimensions)

translated_image = translation(img,100,100)
cv.imshow("translated",translated_image)


# Rotating the image
def rotate_image(image,angle,rotPoint:None):
    (height,width) = image.shape[:2]
    if rotPoint == None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height) 

    return cv.warpAffine(image,rotMat,dimensions)


rotated = rotate_image(img,30,rotPoint=None)
cv.imshow('Rotated',rotated)

# resizing the image
resized = cv.resize(img,(200,300),cv.BORDER_DEFAULT)
cv.imshow("Resized_image",resized)



cv.waitKey(0)