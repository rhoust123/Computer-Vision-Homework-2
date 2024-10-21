# Homework 3 Computer Vision

# Import libraries
import os
import cv2
import glob
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import img_as_float32
from skimage import io

# Read in images and call functions per problem
def Begin_Homework():
    #Problem_One()
    #Problem_Two()
    Problem_Three()
    
# 2D Transformation: Compute the coordinate of a 2D point p = (10, 20)^T using a rotation of 45 degrees about the x-axis, and a translation of t = (40, -30)^T. Answer/Explain the following: 

# 3D Transformation: A camera is located at point (0, -5, 3) in the world frame. The camera is tilted down by 30 degrees from the horizontal. We want to find the 4x4 homogeneous transformation C_H_W from the world frame {W} (A) to the camera frame {C} (B). Note that in "the world" Z is up (X-Y ground plane) but in "the camera", Z is out (X-Y image plane). 
    # * see photo *
    # Answer/Explain the following: 
    # What is C_H_W? Explain how you computed it. 
    # Using transformation C_H_W, transform the point W_p = (0, 0, 1) in the world frame to the camera frame. Hint: Use the homogeneous coordinates of the point for this transformation.

def Problem_One(): 
    
    # PART A
    print("\nPART A\n")

    # What is the point representation in homogenous coordinates?
    p = [10, 20, 1]
    hCoord = (np.array(p)).T
    print("#1) Homogeneous Coordinate representation of point p:")
    print(hCoord)

    # What is the rotation matrix R?
    theta = np.radians(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    print("\n#2) The rotation matrix, R:")
    print(R)
    print("ANSWER: You can use element-wise multiplication between R and \n the original x and y points to obtain a rotation transformation.")

    # What is the translation vector t?
    t = np.array([[1,0,40],[0,1,-30]])
    print("\n#3) The translation matrix, t:")
    print(t)
    print("ANSWER: You can use element-wise multiplication between a matrix of \n([1 0],[0,1]) concatenated with t ([1,0,t_0], [0,1,t_1]) and the original x \n and y points to obtain a translation transformation.")

    # What is the full transformation matrix (consisting of R, t) that can be used to transform the homogeneous point coordinate? 
    T = R @ t
    print("\n#4) The total transformation matrix, T:")
    print(T)

    # How do we apply this transformation to the point (in homogenous coordinate form)?
    pPrime = T @ hCoord
    print("\nANSWER: We can apply this tranformation to the point, in \n homogeneous coordinate form, by using matrix multiplication between T and p.")

    # What is the coordinate of the transformed point, in homogenous coordinates, and in the cartesian coordinates? 
    p = pPrime
    
    print("\nPoint P' in cartesian coordinates:")
    print(p)
    
    print("\nPoint P' in homogeneous coordinates:")
    print(np.array([pPrime[0], pPrime[1], 1]))

    # PART B
    print("\nPART B\n")

    ax, ay, az = np.radians(0), np.radians(240), np.radians(330)

    sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
    cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

    Rx = np.array([[1,0,0], [0,cx,-sx], [0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])

    # Rotation Matrix of W w.r.t. C
    R_W_C = Rz @ Ry @ Rx

    # Translation - origin of W in C
    tWorg_C = np.array([[0, -5, 3]]).T

    # W_H_C - 4x4 Homogeneous transformation matrix from {W} to {C}
    W_H_C = np.block([[R_W_C, tWorg_C], [0,0,0,1]])

    # What is C_H_W? Explain how you computed it. 
    print("\nANSWER: C_H_W is the 4x4 homogeneous coordinate transform matrix. Using this\n matrix, we can convert any point in the world frame to a point in the camera \nframe. The matrix is a combination of 3 primary elements: 1) the 3x3 rotation\n matrix for points in the world frame to the camera frame, R_W_C, 2) the 3x1\n translation vector, tWorg_C, which is a representation of the world frame's\n origin in the camera frame, and 3) the matrix [0,0,0,1], which when appended,\n makes C_H_W homogeneous. To compute C_H_W, you must append all 3 of these\n elements into a single 4x4 matrix.")

    # Using transformation C_H_W, transform the point W_p = (0, 0, 1) in the world frame to the camera frame. Hint: Use the homogeneous coordinates of the point for this transformation. 
    W_p = np.array([[0,0,1,1]]).T
    print("\nThe world point, W_p:\n",W_p)

    # Conversion
    C_p = W_H_C @ W_p

    print("\n")
    print("\nThe 4x4 homogeneous transformation matrix, from world to camera, W_H_C:\n", W_H_C)
    print("\nThe new homogeneous coordinate in the camera frame, C_p:\n", C_p)

# Problem 2: Camera Calibration 
# Find the calibration/intrinsic matrix of a camera (e.g., your cellphone camera). Use the camera calibration board (print PDF file) provided in the HW3 folder. 
    # Provide a copy of your code in the report
    # Display the images you took from the calibration board (at different angles/locations)
    # After calibration, print out the camera intrinsic matrix 
    # Print out five distortion parameters, and explain what they are for
    # Print out camera extrinsic matrices for all your images

# Notes: For this problem, you can follow the camera calibration instructions at: 
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
def Problem_Two(): 
    
    # Citation: The code shown here is largely derived from the geeks for geeks url above. The author is priyarajtt.

    # Define the dimensions of checkerboard 
    CHECKERBOARD = (7, 5) 
    
    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    
    
    # Vector for 3D points 
    threedpoints = [] 
    
    # Vector for 2D points 
    twodpoints = [] 
    
    #  3D points real world coordinates 
    objectp3d = np.zeros((1, CHECKERBOARD[0]  
                        * CHECKERBOARD[1],  
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                                0:CHECKERBOARD[1]].T.reshape(-1, 2) 
    prev_img_shape = None
    
    # Extracting path of individual image stored 
    # in a given directory. Since no path is 
    # specified, it will take current directory 
    # jpg files alone 
    images = glob.glob('images/' + '*.jpg') 
  
    for filename in images: 

        image = cv2.imread(filename)

        if image is None:
            print("Error: Image not loaded. Check filepath.")
            return
        else:
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
        # Find the chess board corners 
        # If desired number of corners are 
        # found in the image then ret = true 
        ret, corners = cv2.findChessboardCorners( 
                        grayColor, CHECKERBOARD,  
                        cv2.CALIB_CB_ADAPTIVE_THRESH  
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE) 
    
        # If desired number of corners can be detected then, 
        # refine the pixel coordinates and display 
        # them on the images of checker board 
        if ret == True: 
            threedpoints.append(objectp3d) 
    
            # Refining pixel coordinates 
            # for given 2d points. 
            corners2 = cv2.cornerSubPix( 
                grayColor, corners, (11, 11), (-1, -1), criteria) 
    
            twodpoints.append(corners2) 
    
            # Draw and display the corners 
            image = cv2.drawChessboardCorners(image,  
                                            CHECKERBOARD,  
                                            corners2, ret) 
        
        else: 
            print("ret was false for image:", filename)
              
    cv2.imshow('img', image) 
    cv2.waitKey(0) 
  
    cv2.destroyAllWindows() 
    
    h, w = image.shape[:2] 
    
    print("Performing camera calibration")
    # Perform camera calibration by 
    # passing the value of above found out 3D points (threedpoints) 
    # and its corresponding pixel coordinates of the 
    # detected corners (twodpoints) 
    if len(threedpoints) == 0 or len(twodpoints) == 0:
        print("no valid data")
        return
    else:
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
            threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
    
    
    # Displaying required output 
    print(" Camera matrix:") 
    print(matrix) 
    
    print("\n Distortion coefficient:") 
    print(distortion) 
    
    print("\n Rotation Vectors:") 
    print(r_vecs) 
    
    print("\n Translation Vectors:") 
    print(t_vecs) 
    
def ProblemThree(): 
    return 

# MAIN
Begin_Homework()