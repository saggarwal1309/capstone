import cv2
import numpy as np

def find_depth(circle_right, circle_left, frame_right, frame_left, baseline, f, alpha):
    # Check input validity
    if circle_right is None or circle_left is None:
        return None
        
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    if width_right != width_left:
        print(f'Frame width mismatch: right={width_right}, left={width_left}')
        return None

    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    
    x_right = circle_right[0]
    x_left = circle_left[0]
    
    # Calculate disparity
    disparity = abs(x_left - x_right)
    if disparity == 0:
        return None
        
    # Calculate depth
    zDepth = (baseline * f_pixel) / disparity
    
    return zDepth
