from flask import Flask
import cv2
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf

global capture, grey, switch, neg, face, out 
capture=0
grey=0
neg=0
face=0
# -1: initialisation, 0: stopped, 1: running 
switch=-1


#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained model    

#net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')
model = tf.keras.models.load_model('./models/model4/my_model_4')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

# def detect_face(frame):
#     global net
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#         (300, 300), (104.0, 177.0, 123.0))   
#     net.setInput(blob)
#     detections = net.forward()
#     confidence = detections[0, 0, 0, 2]

#     if confidence < 0.5:            
#             return frame           

#     box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
#     (startX, startY, endX, endY) = box.astype("int")
#     try:
#         frame=frame[startY:endY, startX:endX]
#         (h, w) = frame.shape[:2]
#         r = 480 / float(h)
#         dim = ( int(w * r), 480)
#         frame=cv2.resize(frame,dim)
#     except Exception as e:
#         pass
#     return frame

# Load the CNN model

def getLetter(result):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','k','L','m','n','o','p','q','r','s','t','u','v','w','x','y']
    return alphabet[int(result)]

def gen_frames():  # generate frame by frame from camera
    global out, capture
    while True:
        success, frame = camera.read() 
        if success:
            frame = cv2.flip(frame,1)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                
            try:
                # define region of interest
                roi= frame[100:320, 100:320]
                cv2.rectangle(frame, (100,100), (320,320), (255,0,0), 5)

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(28,28), interpolation = cv2.INTER_AREA)              
                roi= roi.reshape(1,28,28,1)
                roi = roi/255

                model_predict = model.predict(roi, 1, verbose = 0)
                model_classes = np.argmax(model_predict, axis=1)
                result = str(model_classes[0])

                cv2.putText(frame, getLetter(result), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                # Convert the frame to a JPEG image
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

       
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                          
                 
        elif request.method=='GET':
            return render_template('index.html')
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    
camera.release()
cv2.destroyAllWindows()     