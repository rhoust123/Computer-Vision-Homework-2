# Homework 3 Computer Vision

# Read in images and call functions per problem
def Begin_Homework():
    return 


# 2D Transformation: Compute the coordinate of a 2D point p = (10, 20)^T using a transformation of 45 degrees about the x-axis, and a translation of t = (40, -30)^T. Answer/Explain the following: 
    # What is the point representation in homogenous coordinates?
    # What is the rotation matrix R? 
    # What is the translation vector t? 
    # What is the full transformation matrix (consisting of R, t) that cna be used to transform the homogeneous point coordinate? 
    # How do we apply this transformation to the point (in homogenous coordinate form)? 
    # What is the coordinate of the transformed point, in homogenous coordinates, and in the cartesian coordinates? 

# 3D Transformation: A camera is located at point (0, -5, 3) in the world frame. The camera is tilted down by 30 degrees from the horizontal. We want to find the 4x4 homogeneous transformation C_H_W from the world frame {W} to the camera frame {C}. Note that in "the world" Z is up (X-Y ground plane) but in "the camera", Z is out (X-Y image plane). 
    # * see photo *
    # Answer/Explain the following: 
    # What is C_H_W? Explain how you computed it. 
    # Using transformation C_H_W, transform the point W_p = (0, 0, 1) in the world frame to the camera frame. Hint: Use the homogeneous coordinates of the point for this transformation. 

# Notes: If you are not familiar with coordinate transforms, please take a look at the notes "Coordinate_Transforms.pdf" in the HW3 folder of course materials. 
def Problem_One(): 
    return 

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
    return 


# MAIN
Begin_Homework()