import cv2
from filters import remove_black_bars, canny_edge_detection

# Capture video from camera
# 0 is typically laptop webcam, 1 and onward could be other cameras
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # Remove black bars from the frame
    cropped_frame = remove_black_bars(frame, gray)
    
    edges = canny_edge_detection(cropped_frame)
    
    # Show the results
    cv2.imshow("Real-Time Feed", cropped_frame)
    cv2.imshow("Edges", edges)

    
    # Press Q on one of the frames to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
