import os
from os import walk
from PIL import Image
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

current_id = 0
label_id = {}
x_train =[]
y_labels = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).lower()
            #print(label,'---',path)
            if not label in label_id:
                label_id[label] = current_id
                current_id+=1

            id = label_id[label]
            #print(label_id)
            pil_image = Image.open(path).convert("L")
            img_array = np.array(pil_image,np.uint8)
            #print(img_array)
            faces = face_cascade.detectMultiScale(img_array)

            for(x,y,w,h) in faces:
                roi = img_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id)

#print("*"*50)
#print(y_labels)
#print(x_train)

with open("labels.pickle","wb") as f:
    pickle.dump(label_id,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("clf.yml")