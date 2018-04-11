import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magn = np.sqrt(sobelx*sobelx+sobely*sobely)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_magn = np.uint8(255*magn/np.max(magn))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_magn)
    mag_binary[(scaled_magn >= mag_thresh[0]) & (scaled_magn <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely,abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary

# Read input image
image = mpimg.imread('test_image.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 200))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.9, 1.1))

combined = np.zeros_like(dir_binary)
combined_sobel = np.zeros_like(dir_binary)
combined_gradient = np.zeros_like(dir_binary)
combined_sobel[((gradx == 1) & (grady == 1))] =1
combined_gradient[((mag_binary == 1) & (dir_binary == 1))] =1
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Plot the result
f, axarr = plt.subplots(2, 4, figsize=(24, 9))
f.tight_layout()
axarr[0,0].imshow(image)
axarr[0,0].set_title('Original Image', fontsize=20)
axarr[0,1].imshow(gradx, cmap='gray')
axarr[0,1].set_title('Sobel x', fontsize=20)
axarr[0,2].imshow(grady, cmap='gray')
axarr[0,2].set_title('Sobel y', fontsize=20)
axarr[0,3].imshow(combined_sobel, cmap='gray')
axarr[0,3].set_title('Combined sobel', fontsize=20)
axarr[1,0].imshow(combined, cmap='gray')
axarr[1,0].set_title('Result', fontsize=20)
axarr[1,1].imshow(mag_binary, cmap='gray')
axarr[1,1].set_title('Magnitude', fontsize=20)
axarr[1,2].imshow(dir_binary, cmap='gray')
axarr[1,2].set_title('Direction', fontsize=20)
axarr[1,3].imshow(combined_gradient, cmap='gray')
axarr[1,3].set_title('Combined gradient', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()