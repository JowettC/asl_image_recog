from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf
from flask_socketio import SocketIO, emit
import socketio
import asyncio

global capture, grey, switch, neg, face, out, res, result, str_result, predict
predict=0
str_result = ""
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
model = tf.keras.models.load_model('./models/model4/my_model_4')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

IMG_FOLDER = os.path.join('static', 'img')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER
socketio = SocketIO(app)
camera = cv2.VideoCapture(0)
camera.release()


# to convert index to letter
def getLetter(result):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','k','L','m','n','o','p','q','r','s','t','u','v','w','x','y']
    return alphabet[int(result)]

def gen_frames():  # generate frame by frame from camera
    global out, capture, result, str_result, predict
    while True:
        success, frame = camera.read() 
        
        if success:
            frame = cv2.flip(frame,1)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                predict=1
                print("taking screenshot")
                print("predict")
                print(predict)
                # sets path & takes screenshot
                p = os.path.sep.join(['shots', 'test.png'])
                cv2.imwrite(p, frame)
            try:
                # define region of interest
                roi= frame[100:400, 100:400]
                cv2.rectangle(frame, (100,100), (400,400), (255,0,0), 5)


                # this line causes errors
                if (not grey):
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(28,28), interpolation = cv2.INTER_AREA)              
                roi= roi.reshape(1,28,28,1)
                roi = roi/255

                model_predict = model.predict(roi, 1, verbose = 0)
                model_classes = np.argmax(model_predict, axis=1)
                result = str(model_classes[0])

                cv2.putText(frame, getLetter(result), (250,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                # Convert the frame to a JPEG image
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if (predict == 1):
                    print('predict is 1')
                    str_result += str(getLetter(result))
                    print('test predict')
                    print(str_result)
                    predict=0
                    # loop = asyncio.get_event_loop()
                    # sio = socketio.AsyncClient()

                    # @sio.event
                    # async def update_str(str_result):
                    #     return sio.emit("result", str_result)

                    # async def start_server():
                    #     await sio.connect('http://localhost:5000')
                    #     await sio.wait()

                    if __name__ == '__main__':
                        loop.run_until_complete(start_server())                                       

            except Exception as e:
                # print('exception')
                # print(e)
                pass
        else:
            pass


@app.route('/')
def index():
    header_img = os.path.join(app.config['UPLOAD_FOLDER'], 'header.jpeg')
    asl_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'asl_chart.jpeg')

    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'img3.jpg')
    regenie = os.path.join(app.config['UPLOAD_FOLDER'], 'regenie.jpg')
    vicky = os.path.join(app.config['UPLOAD_FOLDER'], 'vicky.jpg')
    rheza = os.path.join(app.config['UPLOAD_FOLDER'], 'rheza.jpg')

    return render_template('index.html' , height = "800px", header_img=header_img, asl_chart=asl_chart, img1=img1, img2=img2, img3=img3, regenie=regenie, vicky=vicky, rheza=rheza, str_result=str_result)
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    print('req is running')
    if request.method == 'POST':
        print(f"status: {request.form.get('click')}")
        if request.form.get('click') == 'Capture And Predict':
            print("changing capture to 1")
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
                 
        elif request.form.get('clear') == "Clear String":
            global str_result
            str_result = ""
        return render_template('index.html', str_result=str_result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

camera.release()
cv2.destroyAllWindows()     