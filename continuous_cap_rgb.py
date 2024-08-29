import cv2, os
import numpy as np
from calibration import *

# Create a directory to save the images
os.makedirs('captured_images', exist_ok=True)


# Define the dimensions of the checkerboard
CHECKERBOARD = (4,11)
pattern_points = pattern_gen(40)

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

cap = cv2.VideoCapture(0)

count = 0
while count < 50:
    # Read the current frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the circular grid
    ret, corners = cv2.findCirclesGrid(gray, CHECKERBOARD, None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    # If found, add object points, image points
    if ret == True:
        # Save the frame to the directory
        cv2.imwrite(f'captured_images/image_{count}.jpg', frame)
        
        objpoints.append(pattern_points)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
        count += 1
        print(count)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # If the user presses the 'q' key, exit the loop
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix: \n", mtx)
print("Distortion Coefficients: \n", dist)