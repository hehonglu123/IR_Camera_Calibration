import cv2, time, os, traceback
import numpy as np
from RobotRaconteur.Client import *
from calibration import *

ir_img=None
image_consts = None

# Create a directory to save the images
os.makedirs('captured_images', exist_ok=True)


def main():
	global ir_img, image_consts
	###RR FLIR CONNECTION
	
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

	time.sleep(3)


	# print(print(c1.getf_param('atmospheric_temperature').data[0]))
	image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

	p=c1.frame_stream.Connect(-1)

	#Set the callback for when a new pipe packet is received to the
	#new_frame function
	p.PacketReceivedEvent+=new_frame
	try:
		c1.start_streaming()
	except: pass
	

	# Define the dimensions of the checkerboard
	CHECKERBOARD = (4,11)
	blobDetector=blobDetector_initialize()
	pattern_points = pattern_gen(40)
	###########################################################################################################


	# Arrays to store object points and image points from all images
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane
	count = 0
	###CONTINUOUS CAPTURE for CALIBRATION
	while count < 50:
		if ir_img is not None:

			ir_img_inverted = cv2.bitwise_not(ir_img)
			###########################################################################################################
			keypoints = blobDetector.detect(ir_img_inverted) # Detect blobs.

			# Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
			im_with_keypoints = cv2.drawKeypoints(ir_img_inverted, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			###########################################################################################################

			# Find the circular grid
			ret, corners = cv2.findCirclesGrid(ir_img_inverted,CHECKERBOARD,flags=cv2.CALIB_CB_ASYMMETRIC_GRID,blobDetector=blobDetector)   # Find the circle grid
			
			# If found, add object points, image points
			if ret == True:
				# Save the frame to the directory
				cv2.imwrite(f'captured_images/image_{count}.jpg', ir_img_inverted)

				objpoints.append(pattern_points)
				imgpoints.append(corners)
				# Draw and display the corners
				cv2.drawChessboardCorners(ir_img_inverted, CHECKERBOARD, corners, ret)
				count += 1
				print(count)

			# Display the frame
			# print(ir_img.max(), ir_img.min())
			cv2.imshow('IR', im_with_keypoints)

			# If the user presses the 'q' key, exit the loop
			if cv2.waitKey(200) & 0xFF == ord('q'):
				break

	# Release the webcam and destroy all windows
	cv2.destroyAllWindows()

	# Calibrate the camera
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ir_img.shape[::-1], None, None)

	# Print the camera matrix and distortion coefficients
	print("Camera Matrix: \n", mtx)
	print("Distortion Coefficients: \n", dist)


def new_frame(pipe_ep):
	global ir_img, image_consts

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
		
		
		# Normalize the image to the range 0-255
		display_mat_normalized = cv2.normalize(display_mat, None, 0, 255, cv2.NORM_MINMAX)

		# Convert to uint8
		display_mat_normalized = display_mat_normalized.astype(np.uint8)

		# Apply CLAHE to increase contrast
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		display_mat_clahe = clahe.apply(display_mat_normalized)

		# Convert image to gray scale in 8 bit
		ir_img = display_mat_clahe


if __name__ == "__main__":
	main()