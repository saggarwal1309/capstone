import cv2
import numpy as np
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left = cv2.VideoCapture(2, cv2.CAP_DSHOW)

width, height = 640, 480
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap_right.set(cv2.CAP_PROP_EXPOSURE, -4)
cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap_left.set(cv2.CAP_PROP_EXPOSURE, -4)

while True:
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    if not ret_right or not ret_left:
        print("❌ Camera capture failed!")
        break  # ✅ Now correctly inside the while loop

    cv2.imshow("Raw Left Frame", frame_left)
    cv2.imshow("Raw Right Frame", frame_right)

    mask_right = hsv.add_HSV_filter(frame_right, 1)
    mask_left = hsv.add_HSV_filter(frame_left, 0)

    cv2.imshow("Mask Left Debug", cv2.bitwise_and(frame_left, frame_left, mask=mask_left))
    cv2.imshow("Mask Right Debug", cv2.bitwise_and(frame_right, frame_right, mask=mask_right))

    circles_right = shape.find_circles(frame_right, mask_right)
    circles_left = shape.find_circles(frame_left, mask_left)

    print(f"Right circles: {circles_right}")
    print(f"Left circles: {circles_left}")

    if circles_right is None or circles_left is None:
        print("⚠️ No circles detected in one or both frames.")
        continue

    circle_right = circles_right[0] if circles_right else None
    circle_left = circles_left[0] if circles_left else None

    if circle_right and circle_left:
        depth = tri.find_depth(circle_right, circle_left, frame_right, frame_left, 9, 2.66, 56.6)
        print(f"Depth: {depth}")
    
    cv2.imshow("Frame Right", frame_right)
    cv2.imshow("Frame Left", frame_left)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
