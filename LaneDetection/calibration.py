import glob
import matplotlib.image as mpimg
import cv2
import numpy as np
import visualizer

# Parameters
debug = False
calibration_directory = 'camera_cal/'
output_directory = 'output_images/'
calibration_file_name = 'calibration*.jpg'
# Chessboard size 9x6
nx = 9
ny = 6

# Returns calibration matrix
def calibrate():
    """
    :return: calibration matrix mtx, distortion coefficients dist
    """

    # Initialize object and image points
    objpoints = []
    imgpoints = []

    # Collect calibration images
    images = glob.glob(calibration_directory+calibration_file_name)

    # Loop over the calibration images
    for i,image_name in enumerate(images):
        # Read image
        image = mpimg.imread(image_name)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Find corners on chessboard
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # Write found corners
        if ret:
            # print(corners)
            if debug:
                image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
                visualizer.save_image(image,output_directory + calibration_directory + 'corners' + str(i))
            objp = np.zeros((nx*ny,3),np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

            # Append points
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print("No corners detected for " + image_name)

    # Determine calibration matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Plot images
    if debug:
        for i,image_name in enumerate(images):
            # Read image
            image = mpimg.imread(image_name)

            # Distortion correction
            dst = cv2.undistort(image, mtx, dist, None, mtx)

            visualizer.plot_two_images(image,dst,'Distortion correction',
                                       output_directory + calibration_directory + 'distortion' + str(i))
    return mtx,dist
