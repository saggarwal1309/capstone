import cv2
import numpy as np
import imutils
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Set resolution
width, height = 640, 480
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Set exposure (adjust as needed)
cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap_right.set(cv2.CAP_PROP_EXPOSURE, -5)
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap_left.set(cv2.CAP_PROP_EXPOSURE, -5)

# Camera Parameters
B = 9           # Distance between the cameras [cm]
f = 2.66        # Camera lens focal length [mm]
alpha = 56.6    # Camera field of view in the horizontal plane [degrees]

while True:
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    if not ret_right or not ret_left:
        print("\u274c Failed to capture images")
        break

    # Debug: Show raw camera frames
    cv2.imshow("Raw Left Frame", frame_left)
    cv2.imshow("Raw Right Frame", frame_right)

    # APPLYING HSV FILTER:
    mask_right = hsv.add_HSV_filter(frame_right, 1)
    mask_left = hsv.add_HSV_filter(frame_left, 0)

    # Debug: Show filtered mask outputs
    cv2.imshow("Mask Left", mask_left)
    cv2.imshow("Mask Right", mask_right)

    # Apply mask to get result frames
    res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
    res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

    # APPLYING SHAPE RECOGNITION:
    circles_right = shape.find_circles(frame_right, mask_right)
    circles_left = shape.find_circles(frame_left, mask_left)

    # Debugging outputs
    print(f"Right circles: {circles_right}")
    print(f"Left circles: {circles_left}")

    # Ensure valid circle detection
    circle_right = circles_right[0] if circles_right and isinstance(circles_right[0], tuple) else None
    circle_left = circles_left[0] if circles_left and isinstance(circles_left[0], tuple) else None

    if circle_right and circle_left:
        depth = tri.find_depth(circle_right, circle_left, frame_right, frame_left, B, f, alpha)
        
        if depth is not None:
            tracking_text = f"Distance: {round(depth, 3)} cm"
            color = (124, 252, 0)  # Green for tracking
        else:
            tracking_text = "TRACKING LOST"
            color = (0, 0, 255)  # Red for lost tracking
    else:
        tracking_text = "TRACKING LOST"
        color = (0, 0, 255)

    # Display tracking text
    cv2.putText(frame_right, tracking_text, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame_left, tracking_text, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show processed frames
    cv2.imshow("Frame Right", frame_right)
    cv2.imshow("Frame Left", frame_left)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and destroy windows
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
