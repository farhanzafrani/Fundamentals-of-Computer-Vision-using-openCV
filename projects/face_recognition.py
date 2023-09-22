import os
import shutil
import cv2 as cv
import numpy as np

# Define the directory where subdirectories with images are located
directory = r"C:\Users\FARHAN\Desktop\Projects\OpenCV\Fundamentals-of-Computer-Vision-using-openCV\projects\Faces"

# List subdirectories in the main directory
sub_directories = os.listdir(directory)

# Create a directory named "com_images" inside the main directory if it doesn't exist
# com_images = os.path.join(directory, 'com_images')
# os.makedirs(com_images, exist_ok=True)
# # print(os.listdir(directory))

# def combine_images():
#     for sub_dir in sub_directories:
#         if sub_dir != 'com_images':
#             # Get the path of the subdirectory
#             sub_dir_path = os.path.join(directory, sub_dir)
            
#             # List the files (images) in the subdirectory
#             image_files = os.listdir(sub_dir_path)
            
#             for image_file in image_files:
#                 # Get the full path of the image file
#                 image_path = os.path.join(sub_dir_path, image_file)
                
#                 # Use shutil to move the image file to the "com_images" directory
#                 shutil.move(image_path, com_images)
#         else:
#             break

# # Call the combine_images function to perform the operation
# # combine_images()
            

# Reading the haar_cascade path
cascade_path = r"C:\Users\FARHAN\Desktop\Projects\OpenCV\Fundamentals-of-Computer-Vision-using-openCV\projects\haarcascade.xml"
haar_cascade = cv.CascadeClassifier(cascade_path)

dataset_directory = os.listdir(directory)
# print(dataset_directory)

def train_images(): 
    labels = []
    features = []
    for dir in os.listdir(directory):
        if dir == "train":
            persons_path = os.path.join(directory, dir)
            peoples = os.listdir(persons_path)
            for label, people in enumerate(peoples):
                path = os.path.join(persons_path, people)
                for img in os.listdir(path):
                    img_path = os.path.join(path, img)
                    img = cv.imread(img_path)
                    grayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                    face_rect = haar_cascade.detectMultiScale(grayScale, 1.1, 4)

                    for (x, y, w, h) in face_rect:
                        face_roi = grayScale[y:y+h, x:x+w]
                        features.append(face_roi)
                        labels.append(label)
        else:
            continue
    
    return features, labels


features, labels = train_images()
print(f'The length of the features:{len(features)}')
print(f'The length of the labels: {len(labels)}')

#converting into numpy array
features =  np.array(features,dtype='object')
labels = np.array(labels)
# save the features and labels
np.save('features.npy',features)
np.save('labels.npy',labels)

# model face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

# save the model
face_recognizer.save('face_recognizer.yml')


