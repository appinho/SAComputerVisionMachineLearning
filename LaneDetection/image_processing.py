import cv2
import numpy as np

def sobel_operator(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Performs sobel operator in x or y direction
    :param gray: Input grayscale image
    :param orient: Either performs sobel operator in 'x' or 'y' direction
    :param sobel_kernel: Kernel size of sobel operator
    :param thresh: Applied lower and upper threshold
    :return: Binary image with sobel operator information
    """

    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return this mask as binary_output image
    return sobel_binary

def gradient_magnitude(gray, sobel_kernel=3, thresh=(0, 255)):
    """
    Performs gradient magnitude
    :param gray: Input grayscale image
    :param sobel_kernel: Kernel size of sobel operator
    :param thresh: Applied lower and upper threshold
    :return: Binary image with gradient magnitude information
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magn = np.sqrt(sobelx * sobelx + sobely * sobely)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magn = np.uint8(255 * magn / np.max(magn))
    # Create a binary mask where mag thresholds are met
    gradient_magnitude_binary = np.zeros_like(scaled_magn)
    gradient_magnitude_binary[(scaled_magn >= thresh[0]) & (scaled_magn <= thresh[1])] = 255
    # Return this mask as your binary_output image
    return gradient_magnitude_binary

def gradient_orientation(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Performs gradient orientation
    :param gray: Input grayscale image
    :param sobel_kernel: Kernel size of sobel operator
    :param thresh: Applied lower and upper threshold
    :return: Binary image with gradient orientation information
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the orientation of the gradient
    orientation = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where orientation thresholds are met
    gradient_orientation_binary = np.zeros_like(orientation)
    gradient_orientation_binary[(orientation >= thresh[0]) & (orientation <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return gradient_orientation_binary