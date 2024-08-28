import cv2, time, os, traceback
import numpy as np
from scipy.spatial import KDTree
from motoman_def import *

# Function to check the quality of detected corners
def is_good_corners(corners, checkerboard_dims):
    if corners is None:
        return False
    
    # Check if the number of detected corners matches the expected number
    expected_num_corners = checkerboard_dims[0] * checkerboard_dims[1]
    if len(corners) != expected_num_corners:
        return False
    
    # Use KDTree to find the nearest neighbors
    tree = KDTree(corners.reshape(-1, 2))
    distances, _ = tree.query(corners.reshape(-1, 2), k=2)  # k=2 to get the nearest neighbor
    nearest_distances = distances[:, 1]  # The first column is the distance to itself, so we take the second column
    

    if np.max(nearest_distances) > 37:  # Define some_threshold based on your requirements
        return False
    
    return True

# Define the dimensions of the checkerboard
CHECKERBOARD = (4,11)

###########################################################################################################
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

###################################################################################################

###################################################################################################

# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
###########################################################################################################


# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane


###load data

# List all image files in the "captured_images/" directory
recorded_dir='captured_data/'
associated_q2=np.loadtxt(recorded_dir+'associated_q2.csv',delimiter=',')[::1]
num_points=len(associated_q2)
images=[]
valid_indices=[]
# Loop through each image file
for i in range(num_points):
    img=cv2.imread(recorded_dir+'image_'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
    images.append(img)


    # Process the image
    keypoints = blobDetector.detect(img)  # Detect blobs
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Find the circular grid
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, CHECKERBOARD, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    
    # If found, add object points, image points
    if ret and is_good_corners(corners, CHECKERBOARD):
        objpoints.append(objp)
        imgpoints.append(corners)
        valid_indices.append(i)
    #     cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

    
    # cv2.imshow('IR', img)
    
    # if cv2.waitKey(200) & 0xFF == ord('q'):
    #     break

# Release the webcam and destroy all windows
# cv2.destroyAllWindows()
associated_q2=associated_q2[valid_indices]
print(len(objpoints),len(imgpoints),len(associated_q2))
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix: \n", mtx)
print("Distortion Coefficients: \n", dist)

config_dir='../../Welding_Motoman/config/'
robot2_no_tool=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
R_gripper2base = []
t_gripper2base = []
for i in range(len(associated_q2)):
    
    r2_pose=robot2_no_tool.fwd(associated_q2[i])
    R_gripper2base.append(r2_pose.R)
    t_gripper2base.append(r2_pose.p)
print(len(R_gripper2base),len(t_gripper2base),len(rvecs),len(tvecs))
R_target2cam = []
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, rvecs, tvecs)
# Print the results
print("Rotation matrix from camera to gripper:")
print(R_cam2gripper)
print("Translation vector from camera to gripper:")
print(t_cam2gripper)