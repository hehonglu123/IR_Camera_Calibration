from calibration import *
import numpy as np
import cv2
from motoman_def import *

  
config_dir='../../Welding_Motoman/config/'
robot2_no_tool=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')


calibration_dir='calibration_results/'
rvecs=np.loadtxt(calibration_dir+'rvecs.csv',delimiter=',')
tvecs=np.loadtxt(calibration_dir+'tvecs.csv',delimiter=',')

recorded_dir='captured_data/'
q2_all=np.loadtxt(recorded_dir+'associated_q2.csv',delimiter=',')#[::10]
indices=np.arange(0,len(q2_all),1)  #skipping for faster calculation
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane
R_gripper2base=[]
t_gripper2base=[]
images=[]

blobDetector=blobDetector_initialize()
pattern_points = pattern_gen(40)
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
        r2_pose=(robot2_no_tool.fwd(q2_all[i]))
        R_gripper2base.append(r2_pose.R)
        t_gripper2base.append(r2_pose.p)


R_cam2gripper=[[-0.99972873,  0.0080398,   0.02185935],
                [-0.00811353, -0.99996168, -0.00328667],
                [ 0.02183209, -0.00346314,  0.99975565]]
t_cam2gripper=[[  1.37626892],
                [ -6.75729627],
                [196.83094144]]
evaluate_hand_eye_calibration(R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base, rvecs, tvecs)