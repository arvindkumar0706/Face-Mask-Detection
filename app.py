from flask import Flask, render_template, Response
import cv2
from detect_mask_video import detect_and_predict_mask

app = Flask(__name__)
video = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = detect_and_predict_mask(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
