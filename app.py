import cv2
import numpy as np
from RunMediapipe import MediaPipedeploy
import tensorflow as tf
from flask import Flask, render_template, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)
holistic = MediaPipedeploy()
model = tf.keras.models.load_model('model.h5')
sequence = []

def generate_frames():
    global sequence  # Mendeklarasikan sequence sebagai variabel global
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image, results = holistic.mediapipe_detection(frame)
            holistic.draw_styled_landmarks(image, results)
            
            keypoints = holistic.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-15:]
            
            if len(sequence) >= 15:  # Menambahkan pengecekan panjang sequence
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action = holistic.actions[np.argmax(res)]
                image = holistic.prob_viz(res, holistic.actions, image, holistic.colors)
                cv2.putText(image, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
