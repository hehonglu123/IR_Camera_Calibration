import cv2
import numpy as np
from scipy.spatial import KDTree
from motoman_def import *

def pattern_gen(distance=40):
    pattern_points = []
    for j in range(11):
        for i in range(4):
            x = j/2
            if j%2 == 0:
                y = i
            else:
                y = i + 0.5
            pattern_points.append((x,y))

            
    pattern_points = np.hstack((pattern_points,np.zeros((11*4,1)))).astype(np.float32)
    return distance*pattern_points

def blobDetector_initialize():

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

    return cv2.SimpleBlobDetector_create(blobParams)

    
###########################################################################################################
config_dir='../../Welding_Motoman/config/'
robot2_no_tool=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
robot2_flir=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
# Arrays to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

blobDetector=blobDetector_initialize()
pattern_points = pattern_gen(40)
###load data

# List all image files in the "captured_images/" directory
recorded_dir='captured_data/'
q2_all=np.loadtxt(recorded_dir+'associated_q2.csv',delimiter=',')
indices=np.arange(0,len(q2_all),1)  #skipping for faster calculation

associated_q2=[]
images=[]
# Loop through each image file
for i in indices:
    img=cv2.imread(recorded_dir+'image_'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
    images.append(img)


    # Process the image
    keypoints = blobDetector.detect(img)  # Detect blobs
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Find the circular grid
    ret, corners = cv2.findCirclesGrid(img,(4,11),flags=cv2.CALIB_CB_ASYMMETRIC_GRID,blobDetector=blobDetector)

    # If found, add object points, image points
    if ret: # and is_good_corners(corners, CHECKERBOARD):
        objpoints.append(pattern_points)
        imgpoints.append(corners)
        associated_q2.append(q2_all[i])
#         cv2.drawChessboardCorners(img, (4,11), corners, ret)
    
#     cv2.imshow('IR', img)
    
#     if cv2.waitKey(200) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()



print(len(objpoints),len(imgpoints),len(associated_q2))

# Calibrate the camera
projection_error, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(objpoints, imgpoints, img.shape[::-1], None, None)

print("RMSE: ", projection_error)


###Print the camera matrix and distortion coefficients
print("Camera Matrix: \n", mtx)
print("Distortion Coefficients: \n", dist)


R_gripper2base = []
t_gripper2base = []
for i in range(len(associated_q2)):
    
    r2_pose=(robot2_no_tool.fwd(associated_q2[i]))
    R_gripper2base.append(r2_pose.R)
    t_gripper2base.append(r2_pose.p)


for method in [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]:
    try:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, rvecs, tvecs, method=method)
        print("Method: ", method)
        print("Rotation matrix from camera to gripper:")
        print(R_cam2gripper)
        print("Translation vector from camera to gripper:")
        print(t_cam2gripper)
    except:
        pass

# R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
# # Print the results
# print("Rotation matrix from camera to gripper:")
# print(R_cam2gripper)
# print("Translation vector from camera to gripper:")
# print(t_cam2gripper)