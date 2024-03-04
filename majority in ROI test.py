import cv2 as cv 
import numpy as np 
from PIL import Image
import imutils
from get_hsv_values import get_limits # Set masking parameters through here

capture = cv.VideoCapture(0)
maskcolour = [255,0,0]

capture.set(cv.CAP_PROP_FRAME_HEIGHT,480)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
lowerlimit , upperlimit = get_limits(maskcolour)

height = 480
width = 640

roi1_top_left = (0,0)
roi1_bottom_right = (150, 100)

roi2_top_left = (500, 300)
roi2_bottom_right = (640, 480)
while True:

    # Read a frame from the video feed
    ret, frame = capture.read()
    if not ret:
        break

    # Convert ROIs to HSV color space
    roi1 = frame[0:100, 0:150, :]
    roi2 = frame[300:480, 500:640, :]

    roi1_hsv = cv.cvtColor(roi1, cv.COLOR_BGR2HSV)
    roi2_hsv = cv.cvtColor(roi2, cv.COLOR_BGR2HSV)

    # Create binary masks for the ROIs
    roi1_mask = cv.inRange(roi1_hsv, lowerlimit, upperlimit)
    roi2_mask = cv.inRange(roi2_hsv, lowerlimit, upperlimit)

    # Calculate areas of the binary masks
    roi1_area = np.sum(roi1_mask != 0)
    roi2_area = np.sum(roi2_mask != 0)

    # Calculate the total areas of the regions
    roi1_total_area = np.prod(roi1_mask.shape)
    roi2_total_area = np.prod(roi2_mask.shape)

    # Calculate the percentage of areas filled with the specified color
    roi1_percentage = (roi1_area / roi1_total_area) * 100
    roi2_percentage = (roi2_area / roi2_total_area) * 100

    # Draw rectangles for the ROIs on the frame
    cv.rectangle(frame, roi1_top_left, roi1_bottom_right, (0, 255, 0), 2)
    cv.rectangle(frame, roi2_top_left, roi2_bottom_right, (0, 255, 0), 2)

    if roi1_percentage < 50 and roi2_percentage < 50:
        cv.putText(frame,"count : 2" , (320,240),cv.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0],1)
    elif roi1_percentage < 50 or roi2_percentage <50:
        cv.putText(frame,"count : 1" , (320,240),cv.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0],1)
    else:
        cv.putText(frame,"count : 0" , (320,240),cv.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0],1 )
        
    # Draw text on the frame indicating the results for each ROI
    cv.putText(frame, "ROI 1: {}% :".format(roi1_percentage), (50, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(frame, "ROI 2: {}% :".format(roi2_percentage), (450, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    # Display the frame
    cv.imshow('Frame', frame)
    
    # Check for exit key
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

# Release the video capture object and close all windows
capture.release()
cv.destroyAllWindows()