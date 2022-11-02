from flask import Flask,render_template,Response
from deepface import DeepFace
import cv2
 
app=Flask(__name__)
camera = cv2.VideoCapture(0)
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')


def generate_frame():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
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
                print(answer)
                print(sanswer)
            
            

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__=="__main__":
    app.run(debug=True)