
import cv2
import numpy as np
import glob
import open3d as o3d
import os

CHESSBOARD_SIZE = (12, 8)  # Change if using a different pattern
SQUARE_SIZE = 0.025  # Square size in meters (adjust based on printed pattern)

# \ud83d\udd0d Prepare 3D points in real-world space
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points & image points
objpoints = []  # 3D points
imgpoints_left = []  # 2D points (Left camera)
imgpoints_right = []  # 2D points (Right camera)


# Load images
left_images = sorted(glob.glob(r"C:\Users\Dell\capstone\leftcamera\*.png"))
right_images = sorted(glob.glob(r"C:\Users\Dell\capstone\rightcamera\*.png"))


if len(left_images) == 0 or len(right_images) == 0:
    raise Exception("❌ No calibration images found! Ensure images are inside 'stereo_images/'.")

print(f"✅ Found {len(left_images)} left images and {len(right_images)} right images.")

# Detect chessboard corners
for left_img_path, right_img_path in zip(left_images, right_images):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    retL, cornersL = cv2.findChessboardCorners(left_img, CHESSBOARD_SIZE, None)
    retR, cornersR = cv2.findChessboardCorners(right_img, CHESSBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        cv2.drawChessboardCorners(left_img, CHESSBOARD_SIZE, cornersL, retL)
        cv2.drawChessboardCorners(right_img, CHESSBOARD_SIZE, cornersR, retR)

        cv2.imshow("Left Chessboard Detection", left_img)
        cv2.imshow("Right Chessboard Detection", right_img)
        cv2.waitKey(1000)  # Pause for 1 sec to see detections

    else:
        print(f"❌ Chessboard not found in: {left_img_path} or {right_img_path}")

cv2.destroyAllWindows()


# Check if valid chessboard corners were detected
if len(imgpoints_left) == 0 or len(imgpoints_right) == 0:
    raise Exception("❌ No valid chessboard detections! Ensure the pattern is visible in images.")

print(f"✅ Detected chessboard in {len(imgpoints_left)} image pairs.")
