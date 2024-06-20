from flask import Flask, render_template, request, Response, redirect
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import threading

app = Flask(__name__)
#mail = Mail(app)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')



video = cv2.VideoCapture('')
photo = ''
MODEL_FILE="static/model2/best.pt"
COCO_FILE="static/coco.txt"
MODEL_PHOTO="static/model2/best.pt"
COCO_FILE1="static/coco.txt"

@app.route('/')
def index():
    return render_template('index.html')

VIDEO_EXTENSIONS = ['mp4']
PHOTO_EXTENSIONS = ['png','jpeg','jpg']

def fextension(filename):
    return filename.rsplit('.', 1)[1].lower()

@app.route('/upload', methods=['POST'])
def upload():
    global video
    global photo
    if 'video' not in request.files:
        return 'No video file found'
    file = request.files['video']
    if file.filename == '':
        return 'No video selected'
    if file:
        exttype=fextension(file.filename)
        print(exttype)
        if exttype in VIDEO_EXTENSIONS:
            file.save('static/input/gvideo/' + file.filename)
            print('video')
            video = cv2.VideoCapture('static/input/gvideo/' + file.filename)
            return redirect('/video_feed_signs')
        elif exttype in PHOTO_EXTENSIONS:
            file.save('static/input/gphoto/' + file.filename)
            print('photo')
            photo = 'static/input/gphoto/' + file.filename
            return redirect('/video_feed_sign')
    return 'Invalid video file'




def gen_new(video):
    # opening the file in read mode
    my_file = open(COCO_FILE, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()

    # load a pretrained YOLOv8n model
    model = YOLO(MODEL_FILE, "v8")
    while True:
        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, (720, 480))

        detect_params = model.predict(source=[frame], conf=0.45, save=False)
        #print(detect_params)
        DP = detect_params[0].cpu().numpy()
        #print(DP)
        no_faces=0
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes.cpu()
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),(150, 150, 255),3,)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame, "Status: "+ class_list[int(clsID)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.putText(frame,class_list[int(clsID)]+" "+str(conf),(int(bb[0]), int(bb[1]) - 10),font,0.5,(150, 150, 255),1,)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_signs')
def video_feed_signs():
    global video
    if not (video.isOpened()):
        return 'Could not process video'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_feed():
    global video
    video = cv2.VideoCapture(0)
    if not (video.isOpened()):
        return 'Could not connect to camera'
    return Response(gen_new(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_new2(photo):
    my_file = open(COCO_FILE1, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text | when newline ('\n') is seen.
    class_list = data.split("\n")
    my_file.close()

    model = YOLO(MODEL_PHOTO, "v8").to(device)
    print(photo)
    frame = cv2.imread(photo)
    frame=cv2.resize(frame, (720, 480))
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    print(detect_params)
    DP = detect_params[0].numpy()
    #print(DP)
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(frame,(int(bb[0]), int(bb[1])),(int(bb[2]), int(bb[3])),(255,255,255),3,)
            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame,class_list[int(clsID)],(int(bb[0]), int(bb[1]) - 10),font,0.5,(255, 255, 255),1,)
            cv2.putText(frame, "Status: "+ class_list[int(clsID)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.imwrite('static/input/photo/dskjds.jpg', frame) 
            if class_list[0] == "sign":
                cv2.putText(frame, "Emergency!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
    # cv2.putText(frame, "Status: ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    # cv2.imwrite('static/input/photo/dskjds.jpg', frame) 
    
@app.route('/video_feed_sign')
def video_feed_sign():
    global photo
    gen_new2(photo)
    return render_template('preview_photo.html', file_name='dskjds.jpg', type='image/jpg')



if __name__ == '__main__':
    app.run(debug=True)