import cv2
from flask import Flask, Response
from filters import remove_black_bars
from ultralytics import YOLO
from deepface import DeepFace
import time

# Initialize the Flask app
app = Flask(__name__)

video_path = 'HCI_Video_Test_2.mp4'

# Initialize the camera
cap = cv2.VideoCapture(video_path)

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

# Store human counts over time
human_count_history = []
bad_emotions = 0

def detect_surge(surge_threshold):
    global human_count_history
    current_time = time.time()
    # Remove entries older than 5 seconds
    human_count_history = [(t, count) for t, count in human_count_history if current_time - t <= 5]

    # If there isn't many data points, return early and dont
    # compare
    if len(human_count_history) < 2:
        return False

    # Compare the first and last recorded counts
    first_count = human_count_history[0][1]
    last_count = human_count_history[-1][1]

    return abs(last_count - first_count) >= surge_threshold

def frame_generation():
    frame_counter = 0  # Counter to keep track of frames
    human_bad_emotions = 0
    
    while True:
        success, frame = cap.read()  # Read a frame from the camera
        if not success:
            break

        # Increment frame counter
        frame_counter += 1

        # Remove black bars from the frame if they exist
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = remove_black_bars(frame, gray)

        # Perform YOLOv8 detection
        results = model.predict(cropped_frame)

        # Keep track of the number of humans and security rating
        num_heads = 0
        security_rating = 0
        
        # Reset bad emotion counter every 5 frames
        if frame_counter % 5 == 0:
            human_bad_emotions = 0
            
        # Draw bounding boxes for detected people
        for result in results:
            for box in result.boxes:
                # Check to see if the object detected is a person
                if int(box.cls) == 0:
                    num_heads += 1
                    # Convert bounding box coordinates from tensor to integers
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Run emotion analysis only every 5 frames
                    if frame_counter % 5 == 0:
                        # Grab image of human and resize for faster emotion detection
                        human_location = frame[y1:y2, x1:x2]
                        human_location = cv2.resize(human_location, (0, 0), fx=0.5, fy=0.5)
                        
                        # Perform emotion analysis 
                        emotion_analysis = DeepFace.analyze(human_location, actions=['emotion'], enforce_detection=False, silent=False)
                        dominant_emotion = emotion_analysis[0]['dominant_emotion']

                        # List of "good" emotions
                        emotions = ["happy", "sad", "surprise", "neutral"]
                        
                        # Set box color depending on emotion
                        box_color = (0, 255, 0) if dominant_emotion in emotions else (0, 0, 255)
                        
                        # Update the # of bad emotions detected if needed
                        human_bad_emotions = (human_bad_emotions + 1) if dominant_emotion not in emotions else human_bad_emotions
                    else:
                        box_color = (0, 255, 255)  # Default color for skipped frames

                    # Draw colored box around detected human
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Calculate the security rating of the frame
        security_rating = round((num_heads - human_bad_emotions) * 100 / num_heads, 2) if num_heads > 0 else 100.0
        
        # Record the current count and timestamp
        human_count_history.append((time.time(), num_heads))

        # Check for a surge (rate of change of 8 within 5 seconds)
        if detect_surge(8):
            print("Surge detected!")

        # Add text to the image with a background
        text_with_background(frame, f'Humans: {num_heads}', 1, 2, 10, 30, (0, 255, 0), (0, 0, 0))
        text_with_background(frame, f'Security Rating: {security_rating}%', 1, 2, 10, 70, (255, 255, 255), (0, 0, 0))

        # Encode the frame to JPG format
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
