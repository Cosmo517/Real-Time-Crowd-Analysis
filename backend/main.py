import cv2
from flask import Flask, Response
from filters import remove_black_bars, canny_edge_detection

# Create a flask app so we can stream video to a website
app = Flask(__name__)

def frame_generation():
    # Capture video from camera
    # 0 is typically laptop webcam, 1 and onward could be other cameras
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Remove black bars from the frame
        cropped_frame = remove_black_bars(frame, gray)
        
        edges = canny_edge_detection(cropped_frame)
        
        ret, edges_jpg = cv2.imencode('.jpg', edges)
        edges_bytes = edges_jpg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + edges_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(frame_generation(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)