import cv2, time, os
import numpy as np
from RobotRaconteur.Client import *
from motoman_def import *
from StreamingSend import *

ir_img=None
image_consts = None
image_updated = False

# Create a directory to save the images
os.makedirs('captured_data', exist_ok=True)


def main():
	global ir_img, image_consts, image_updated
	#####################################RR FLIR #####################################
	
	url='rr+tcp://localhost:60827/?service=camera'

	c1=RRN.ConnectService(url)
	c1.setf_param("focus_pos", RR.VarValue(int(2222),"int32"))
	c1.setf_param("object_distance", RR.VarValue(0.3,"double"))
	c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
	c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
	c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
	c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
	c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))

	c1.setf_param("current_case", RR.VarValue(1,"int32"))
	# c1.setf_param("ir_format", RR.VarValue("temperature_linear_100mK","string"))
	c1.setf_param("ir_format", RR.VarValue("radiometric","string"))

	c1.setf_param("object_emissivity", RR.VarValue(0.7,"double"))    
	
	c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
	c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

	# print(print(c1.getf_param('atmospheric_temperature').data[0]))
	image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

	p=c1.frame_stream.Connect(-1)

	#Set the callback for when a new pipe packet is received to the
	#new_frame function
	p.PacketReceivedEvent+=new_frame
	try:
		c1.start_streaming()
	except: pass
	
	#####################################RR Robot#####################################
	config_dir='../../Welding_Motoman/config/'
	robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	robot2_no_tool=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	streaming_rate=125.
	SS=StreamingSend(RR_robot_sub,streaming_rate=streaming_rate)

	#####################################Path Gen#####################################
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	center_of_rotation=positioner.fwd(q_positioner_home,world=True).p
	measure_distance=500
	H2010_1440=H_inv(robot2.base_H)
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],center_of_rotation)+H2010_1440[:3,3]

	total_time=10
	num_points=total_time*streaming_rate
	q2_cmd_all=[]
	for angle in np.linspace(np.pi/4,5*np.pi/4,num_points):	#define rotation range
		q_positioner=np.array([angle,np.pi/2])
		SS.send_positioner(q_positioner)
		v_z_global=-Rx(angle)[:,1]
		v_z=H2010_1440[:3,:3]@v_z_global ###pointing toward positioner's X with 15deg tiltd angle looking down
		v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
		v_x=np.cross(v_y,v_z)
		p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
		R2=np.vstack((v_x,v_y,v_z)).T
		q2_cmd_all.append(robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0])

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
	##############################################MOTION#############################################################

	SS.jog2q(np.hstack((np.zeros(6),q2_cmd_all[0]),q_positioner_home))

	# Arrays to store object points and image points from all images
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane
	count = 0
	images = []
	associated_q2=[]
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q2_cmd_all)):
		SS.position_cmd(np.hstack((np.zeros(6),q2_cmd_all[i]),q_positioner_home),now)
		now=time.perf_counter()
		if ir_img is not None and image_updated:
			image_updated = False

			ir_img_inverted = cv2.bitwise_not(ir_img)
			###########################################################################################################
			keypoints = blobDetector.detect(ir_img_inverted) # Detect blobs.

			# Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
			im_with_keypoints = cv2.drawKeypoints(ir_img_inverted, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			###########################################################################################################

			# Find the circular grid
			ret, corners = cv2.findCirclesGrid(im_with_keypoints, CHECKERBOARD, None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

			# If found, add object points, image points
			if ret == True:

				images.append(ir_img_inverted)
				objpoints.append(objp)
				imgpoints.append(corners)
				count += 1
				associated_q2.append(SS.q_cur[6:12])
	
	print("Number of images captured: ", count)
	#save data
	for i in range(len(images)):
		cv2.imwrite(f'captured_data/image_{i}.jpg', images[i])
	np.savetxt('captured_data/associated_q2.csv',associated_q2,delimiter=',')

	# Calibrate the camera
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ir_img.shape[::-1], None, None)

	# Print the camera matrix and distortion coefficients
	print("Camera Matrix: \n", mtx)
	print("Distortion Coefficients: \n", dist)

	R_gripper2base = []
	t_gripper2base = []
	for i in range(len(images)):
		
		r2_pose=robot2_no_tool.fwd(associated_q2[i])
		R_gripper2base.append(r2_pose.R)
		t_gripper2base.append(r2_pose.p)

	R_target2cam = []
	R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, rvecs, tvecs)
	# Print the results
	print("Rotation matrix from camera to gripper:")
	print(R_cam2gripper)
	print("Translation vector from camera to gripper:")
	print(t_cam2gripper)

def new_frame(pipe_ep):
	global ir_img, image_consts, image_updated

	#Loop to get the newest frame
	while (pipe_ep.Available > 0):
		#Receive the packet
		rr_img=pipe_ep.ReceivePacket()
		if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono8"]:
			# Simple uint8 image
			mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		elif rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
			data_u16 = np.array(rr_img.data.view(np.uint16))
			mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		ir_format = rr_img.image_info.extended["ir_format"].data

		if ir_format == "temperature_linear_10mK":
			display_mat = (mat * 0.01) - 273.15    
		elif ir_format == "temperature_linear_100mK":
			display_mat = (mat * 0.1) - 273.15    
		else:
			display_mat = mat
		
		#convert image to gray scale in 8 bit
		ir_img = cv2.normalize(display_mat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		image_updated = True

if __name__ == "__main__":
	main()