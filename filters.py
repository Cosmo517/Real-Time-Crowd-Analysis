import cv2
import numpy as np

def remove_black_bars(frame, gray_frame):
    """Crops the frame to remove horizontal and vertical black bars

    Args:
        frame (image): image frame
        gray_frame (image): grayscale of the frame

    Returns:
        array: The cropped frame without the black bars
    """
    
    # Find all the rows and columns that are not completely black
    non_black_rows = np.where(gray_frame.max(axis=1) > 10)[0]
    non_black_cols = np.where(gray_frame.max(axis=0) > 10)[0]

    # Crop the image to only include the non-black regions
    if non_black_rows.size > 0 and non_black_cols.size > 0:
        cropped_frame = frame[non_black_rows[0]:non_black_rows[-1], non_black_cols[0]:non_black_cols[-1]]
    else:
        cropped_frame = frame  # If there are no black bars, return the original frame
    
    return cropped_frame

def blur_frame(frame):
    """Blurs an image frame

    Args:
        frame (image): The frame to blur

    Returns:
        image: The blurred image
    """
    
    blurred = cv2.GaussianBlur(src=frame, ksize=(3, 5), sigmaX=0.5) 
    return blurred

def canny_edge_detection(gray_frame):
    """Performs canny edge detection on the grayscale frame

    Args:
        gray_frame (image): Grayscale frame

    Returns:
        _type_: _description_
    """
    
    # Apply Gaussian blur to reduce noise and smoothen edges 
    blurred = blur_frame(gray_frame)
    
    # Perform Canny edge detection 
    edges = cv2.Canny(blurred, 60, 145) 
    
    return edges