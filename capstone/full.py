import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Set resolution
width, height = 640, 480
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

frame_rate = 120    # Camera frame rate (maximum at 120 fps)

B = 9               # Distance between the cameras [cm]
f = 2.66            # Camera lens focal length [mm]
alpha = 56.6        # Camera field of view in the horizontal plane [degrees]

count = -1

while True:
    count += 1
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    # If cannot catch any frame, break
    if not ret_right or not ret_left:
        print("\u274c Failed to capture images")
        break

    # APPLYING HSV-FILTER:
    mask_right = hsv.add_HSV_filter(frame_right, 1)
    mask_left = hsv.add_HSV_filter(frame_left, 0)

    # Debugging: Show Masked Output
    cv2.imshow("Mask Left Debug", mask_left)
    cv2.imshow("Masked Left Frame", cv2.bitwise_and(frame_left, frame_left, mask=mask_left))

    # Result-frames after applying HSV-filter mask
    res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
    res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

    # APPLYING SHAPE RECOGNITION:
    circles_right = shape.find_circles(frame_right, mask_right)
    circles_left = shape.find_circles(frame_left, mask_left)

    # Debug prints
    print("Right circles:", circles_right)
    print("Left circles:", circles_left)

    # Validate circles before processing
    if circles_right is not None and len(circles_right) > 0:
        circle_right = circles_right[0] if isinstance(circles_right[0], tuple) else (0, 0, 0)
    else:
        circle_right = None

    if circles_left is not None and len(circles_left) > 0:
        circle_left = circles_left[0] if isinstance(circles_left[0], tuple) else (0, 0, 0)
    else:
        circle_left = None

    if circle_right is not None and circle_left is not None and len(circle_right) >= 2 and len(circle_left) >= 2:
        depth = tri.find_depth(circle_right, circle_left, frame_right, frame_left, B, f, alpha)
        
        if depth is not None:
            cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
            print("Depth:", depth)
        else:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    else:
        cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    # Show the frames
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)
    cv2.imshow("mask right", mask_right)
    cv2.imshow("mask left", mask_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
