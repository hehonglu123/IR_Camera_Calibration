import cv2, time, os
import numpy as np
from RobotRaconteur.Client import *
from motoman_def import *
from StreamingSend import *

from calibration import *

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

	#####################################Path Gen1#####################################
	q1_default=np.zeros(6)
	q1_default[1]=-np.pi/6
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	center_of_rotation=positioner.fwd(q_positioner_home,world=True).p
	measure_distance=550
	H2010_1440=H_inv(robot2.base_H)
	center_of_rotation_in_base_frame=np.dot(H2010_1440[:3,:3],center_of_rotation)+H2010_1440[:3,3]
	total_time=20
	num_points=int(total_time*streaming_rate)
	q2_cmd_all1=[]
	for angle in np.linspace(np.pi/5,3*np.pi/5,num_points):	#define rotation range
		v_z_global=-Rx(angle)[:,1]
		v_z=H2010_1440[:3,:3]@v_z_global ###pointing toward positioner's X with 15deg tiltd angle looking down
		v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
		v_x=np.cross(v_y,v_z)
		p2_in_base_frame=center_of_rotation_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
		R2=np.vstack((v_x,v_y,v_z)).T
		q2_cmd_all1.append(robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0])

	#####################################Path Gen2#####################################
	q2_cmd_all2=[]
	for angle in np.linspace(np.pi/2,np.pi/4,num_points):	#define rotation range
		v_z_global=-Ry(-angle)[:,0]
		v_z=H2010_1440[:3,:3]@v_z_global ###pointing toward positioner's X with 15deg tiltd angle looking down
		v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
		v_x=np.cross(v_y,v_z)
		p2_in_base_frame=center_of_rotation_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
		R2=np.vstack((v_x,v_y,v_z)).T
		q2_cmd_all2.append(robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0])

	# Define the dimensions of the checkerboard
	CHECKERBOARD = (4,11)
	blobDetector=blobDetector_initialize()
	pattern_points = pattern_gen(40)
	
	##############################################MOTION#############################################################

	

	# Arrays to store object points and image points from all images
	images = []
	associated_q2=[]

	SS.jog2q(np.hstack((q1_default,q2_cmd_all1[0],q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q2_cmd_all1)):
		SS.position_cmd(np.hstack((q1_default,q2_cmd_all1[i],q_positioner_home)),now)
		now=time.perf_counter()
		if ir_img is not None and image_updated:
			associated_q2.append(SS.q_cur[6:12])
			image_updated = False	
			images.append(ir_img)

	SS.jog2q(np.hstack((q1_default,q2_cmd_all2[0],q_positioner_home)))
	now=time.perf_counter()
	###CONTINUOUS CAPTURE for CALIBRATION
	for i in range(len(q2_cmd_all2)):
		SS.position_cmd(np.hstack((q1_default,q2_cmd_all2[i],q_positioner_home)),now)
		now=time.perf_counter()
		if ir_img is not None and image_updated:
			associated_q2.append(SS.q_cur[6:12])
			image_updated = False
			images.append(ir_img)
			


	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane
	valid_indices=[]
	processed_images = []
	###filter captured images with no detected checkerboard
	for i in range(len(associated_q2)):

		ir_img_inverted = cv2.bitwise_not(images[i])
		# Normalize the image to the range 0-255
		ir_img_normalized = cv2.normalize(ir_img_inverted, None, 0, 255, cv2.NORM_MINMAX)

		# Convert to uint8
		ir_img_normalized = ir_img_normalized.astype(np.uint8)

		# Apply CLAHE to increase contrast
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		ir_img_clahe = clahe.apply(ir_img_normalized)
		###########################################################################################################

		# Find the circular grid
		ret, corners = cv2.findCirclesGrid(ir_img_clahe,CHECKERBOARD,flags=cv2.CALIB_CB_ASYMMETRIC_GRID,blobDetector=blobDetector)   # Find the circle grid

		# If found, add object points, image points
		if ret == True:

			valid_indices.append(i)
			processed_images.append(ir_img_clahe)
			objpoints.append(pattern_points)
			imgpoints.append(corners)

	
	print("Number of images captured: ", len(valid_indices))
	associated_q2 = [associated_q2[i] for i in valid_indices]


	#save data
	for i in range(len(processed_images)):
		cv2.imwrite(f'captured_data/image_{i}.jpg', processed_images[i])
	np.savetxt('captured_data/associated_q2.csv',associated_q2,delimiter=',')


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
		
		ir_img = display_mat

		#updated flag
		image_updated = True

		

if __name__ == "__main__":
	main()