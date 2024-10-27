import cv2
from flask import Flask, Response
from filters import remove_black_bars, canny_edge_detection

# Initialize the Flask app
app = Flask(__name__)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar cascade for head (or face) detection
head_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def frame_generation():
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break

        # Convert frame to grayscale (required by the cascade classifier)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Remove black bars from the frame
        cropped_frame = remove_black_bars(frame, gray)

        # Detect heads in the frame
        heads = head_cascade.detectMultiScale(cropped_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around detected heads
        for (x, y, w, h) in heads:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box around head

        # Encode the frame with head detections to JPEG
        ret, frame_encode = cv2.imencode('.jpg', frame)
        frame_bytes = frame_encode.tobytes()
        
        # Yield the frame as part of the stream
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(frame_generation(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
