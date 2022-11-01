import cv2
from deepface import DeepFace
from flask import Flask
import matplotlib.pyplot as plt

app= Flask(__name__)
camera = cv2.VideoCapture(0)
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')

while True:
    ret,frame=camera.read()
    answer= DeepFace.analyze(frame,actions=['emotion'])
    sanswer= DeepFace.analyze(frame,actions=['age'])

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiscale(gray,1.1,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(103, 14, 212),2)

        font=cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,answer['dominant_emotion'],(x,y-20),font,1,(103, 14, 212),2,cv2.LINE_4)
        cv2.putText(frame,sanswer['dominant_emotion'],(x,y-20),font,1,(103, 14, 212),2,cv2.LINE_4)
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF==ord('k'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ =="__main__":
    app.run()
    