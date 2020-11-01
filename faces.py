import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("clf.yml")

with open("labels.pickle","rb") as f:
    labels = pickle.load(f)
    labels_final = {v:k for k,v in labels.items()}

#cap = cv2.VideoCapture(0)

#while(True):
    #ret,frame = cap.read()
def facr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        id,conf = recognizer.predict(roi_gray)
        if(conf>=45):
            print(labels_final[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels_final[id]
            cv2.putText(frame,name,(x,y),font,1,(0,0,0),1,cv2.LINE_AA)
        #img_item = 'myimage.png'
        #cv2.imwrite(img_item,roi_gray)

        #drawing bounding box
        color = (255,0,0)
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()