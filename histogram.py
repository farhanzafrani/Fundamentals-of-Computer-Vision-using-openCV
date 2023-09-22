import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


path1 = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\house.jpg"
path2 = r"Fundamentals-of-Computer-Vision-using-openCV\image_processing_files\road.jpg"

img = cv.imread(path1)
image = cv.imread(path2)

cv.imshow("original image",image)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY,cv.BORDER_DEFAULT)
# cv.imshow("grayScale image",gray)
# (b,g,r) =  cv.split(img)

colors = ['r','g','b']
for i, col in enumerate(colors):
    hist = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.title('Histogram')
    plt.xlabel('hist --> (bins)')
    plt.ylabel('# of pixals')
    plt.xlim([0,256])


plt.show()

# histogram = cv.calcHist([img],[0,1,2],None,[256],[0,256])

# plt.figure()
# plt.plot(histogram)
# plt.title('Histogram')
# plt.xlabel('hist --> (bins)')
# plt.ylabel('# of pixals')
# plt.xlim([0,256])
# plt.show()