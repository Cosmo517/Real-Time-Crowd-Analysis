import cv2
from flask import Flask, Response
from filters import remove_black_bars
from ultralytics import YOLO

# Initialize the Flask app
app = Flask(__name__)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load our YOLO model. We choose to use the yolov8 model 
# currently because its lightweight and provides a good
# amount of accuracy in human identification
model = YOLO('yolov8n.pt')

# This function will allow adding text to the image, with a specified background color
def text_with_background(image, text, scale, thickness, text_x, text_y, text_color, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

    # Draw rectangle background
    cv2.rectangle(image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), bg_color, -1)

    # Draw text on top of the rectangle
    cv2.putText(image, text, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)

def frame_generation():
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break

        # Remove black bars from the frame if they exist
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = remove_black_bars(frame, gray)

        # Perform YOLOv8 detection
        results = model.predict(cropped_frame)

        # Draw bounding boxes for detected people
        num_heads = 0
        for result in results:
            for box in result.boxes:
                # Check to see if the object detected is a person
                if int(box.cls) == 0:  # Ensure cls is an integer
                    num_heads += 1
                    # Convert bounding box coordinates from tensor to integers
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Green box around detected person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        # Add text to the image with a background
        text_with_background(frame, f'Humans: {num_heads}', 1, 2, 10, 30, (0, 255, 0), (0, 0, 0))

        # Encode the frame to JPEG format
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
