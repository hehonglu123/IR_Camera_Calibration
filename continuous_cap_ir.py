import cv2, time, os
import numpy as np
from RobotRaconteur.Client import *

# Create a directory to save the images
os.makedirs('captured_images', exist_ok=True)

def new_frame(pipe_ep):
    global ir_img

    #Loop to get the newest frame
    while (pipe_ep.Available > 0):
        #Receive the packet
        rr_img=pipe_ep.ReceivePacket()
        if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
            data_u16 = np.array(rr_img.data.view(np.uint16))
            mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')

            #convert image to gray scale in 8 bit
            ir_img = cv2.convertScaleAbs(mat, alpha=1/256)

image_consts = None
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

def main():
    ###RR FLIR CONNECTION
    
    url='rr+tcp://localhost:60827/?service=camera'

    c1=RRN.ConnectService(url)
    c1.setf_param("focus_pos", RR.VarValue(int(1400),"int32"))
    c1.setf_param("object_distance", RR.VarValue(0.3,"double"))
    c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
    c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
    c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))

    c1.setf_param("current_case", RR.VarValue(2,"int32"))
    c1.setf_param("ir_format", RR.VarValue("temperature_linear_100mK","string"))
    # c1.setf_param("ir_format", RR.VarValue("radiometric","string"))

    c1.setf_param("object_emissivity", RR.VarValue(0.7,"double"))    
    
    c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
    c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

    time.sleep(3)


    # print(print(c1.getf_param('atmospheric_temperature').data[0]))
    global image_consts
    image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

    p=c1.frame_stream.Connect(-1)

    #Set the callback for when a new pipe packet is received to the
    #new_frame function
    p.PacketReceivedEvent+=new_frame
    try:
        c1.start_streaming()
    except: pass
    
    ###CONTINUOUS CAPTURE for CALIBRATION
    while count < 100:
        ir_img_inverted = cv2.flip(ir_img, 1)
        # Find the circular grid
        ret, corners = cv2.findCirclesGrid(ir_img_inverted, CHECKERBOARD, None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

        # If found, add object points, image points
        if ret == True:
            # Save the frame to the directory
            cv2.imwrite(f'captured_images/image_{count}.jpg', ir_img_inverted)

            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(ir_img_inverted, CHECKERBOARD, corners, ret)
            count += 1
            print(count)

        # Display the frame
        cv2.imshow('IR', ir_img_inverted)

        # If the user presses the 'q' key, exit the loop
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ir_img.shape[::-1], None, None)

    # Print the camera matrix and distortion coefficients
    print("Camera Matrix: \n", mtx)
    print("Distortion Coefficients: \n", dist)


