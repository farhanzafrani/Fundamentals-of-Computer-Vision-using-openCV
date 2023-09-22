import cv2 as cv
import numpy as np


def image_prediction(path):
    # Reading the haar_cascade path
    cascade_path = r"C:\Users\FARHAN\Desktop\Projects\OpenCV\Fundamentals-of-Computer-Vision-using-openCV\projects\haarcascade.xml"
    haar_cascade = cv.CascadeClassifier(cascade_path)

    # Reading the features and labels
    peoples = ['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
    # features = np.load('features.npy')
    # labels = np.load('labels.npy')

    # Reading the face recognizer file
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_recognizer.yml')

    img = cv.imread(path)


    grayScale = cv.cvtColor(img,cv.COLOR_BGR2GRAY,cv.BORDER_DEFAULT)
    cv.imshow("grayScale",grayScale)

    face_rect = haar_cascade.detectMultiScale(grayScale,1.1,4)

    for (x,y,w,h) in face_rect:
        face_roi = grayScale[y:y+h,x:x+w]
        
        label , confidence = face_recognizer.predict(face_roi)

        print(f'Label is {label} with a confidence of {confidence}')
        print(f'The person will be {peoples[label]}')
        cv.putText(img,str(peoples[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,color=(0,255,0),thickness=2)


    cv.imshow("Original image",img)

    cv.waitKey(0)


image_prediction(r"Fundamentals-of-Computer-Vision-using-openCV\projects\Faces\test\mindy_kaling\4.jpg")