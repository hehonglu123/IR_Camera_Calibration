import cv2
import numpy as np
import time

# Define the dimensions of the checkerboard
CHECKERBOARD = (4,11)

# Prepare object points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

cap = cv2.VideoCapture(0)

count = 0
while count < 100:
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
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
        count += 1
        print(count)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # If the user presses the 'q' key, exit the loop
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix: \n", mtx)
print("Distortion Coefficients: \n", dist)