import cv2
from flask import Flask, Response
from filters import remove_black_bars

# Initialize the Flask app
app = Flask(__name__)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar cascade for head (or face) detection
head_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def text_with_background(image, text, scale, thickness, text_color, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

    text_x, text_y = 10, 30 

    # Draw rectangle background
    cv2.rectangle(image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), bg_color, -1)

    # Draw text on top of the rectangle
    cv2.putText(image, text, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)

def frame_generation():
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break

        # Convert frame to grayscale (required by the cascade classifier)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Remove black bars from the frame
        cropped_frame = remove_black_bars(frame, gray)

        # Detect heads in the frame
        heads = head_cascade.detectMultiScale(cropped_frame, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        
        # Add text to the image with a background
        text_with_background(frame, f'Humans: {len(heads)}', 1, 2, (0, 255, 0), (0, 0, 0))
    
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
