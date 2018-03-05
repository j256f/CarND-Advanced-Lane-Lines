import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, os
import pickle

images =  glob.glob('./camera_cal/calibration*.jpg') #append all image names into "image"

#Arrays to store object points and images points

objpoints = [] # 3D points in real world space, to be filled with cordinate of corners found
imgpoints = [] # 2D points in image plane


# xH is the number of corners along the horizon, xI are the corners vertically
xH = 9
xI = 6


# Prepare object points
objp = np.zeros((xH*xI,3), np.float32) # prepared zero matix of shape 54 x 3
objp[:,:2] = np.mgrid[0:xH,0:xI].T.reshape(-1,2)  # fill first two collums with 54 combinations of 9 leves and 6 levels, with  mgrid magic using (transpose and reshaping)

# Now first item is [0. 0. 0.] and last is [8. 5. 0]

for idx, fname in enumerate(images): # go through each image in image, enumerate is as range(len(images)) 
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscalee
    ret, corners = cv2.findChessboardCorners(gray, (xH,xI), None) # this function tries to find all 54 corners in every image
    

    if ret == True:                  
        print('working on ', fname)   
        imgpoints.append(corners)    # if corner is found, add the 54 cordinates   
        objpoints.append(objp)       # also add the 54 indexes to relate  each corner
       
        #Display corners

        cv2.drawChessboardCorners(img, (xH,xI), corners, ret)  # draw the corner into the image were it was found
        write_name = 'corners_found'+str(idx)+'.jpg' # how should we name the anotaded image
        cv2.imwrite(write_name, img) # save the image 


# Load  image for reference, in order to get the size

img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0]) # in this case is 1280 x 720


# camera calibration
# this function will take all 17 sets of cordenates of the 54 corners found that are allocated in 'objpoints' 
# we just need 10 sets, 17 will be enough
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


# the function returs camera matrix as "mtx" and distorsion coefficients as "dist", the rotations and translation vectors "rvecs" and "tvecs" wont be used

# we will just pickle the needed data, e.i. the camera matrix and distorsion coefficients: 

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open( "./calibration_pickle_2.p", "wb"))


    
