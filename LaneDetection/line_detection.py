import matplotlib.image as mpimg
import cv2
import visualizer
import numpy as np
import image_processing

class LineDetector():
    def __init__(self,C, coeff):
        """
        Constructor
        :param C: Calibration matrix
        :param coeff: Distortion coefficients
        """
        self.C = C
        self.coeff = coeff

        self.thresh_min_sobel_x = 10
        self.thresh_max_sobel_x = 255
        self.thresh_min_sobel_y = 30
        self.thresh_max_sobel_y = 255
        self.thresh_min_grad_dir = 0# 0.9
        self.thresh_max_grad_dir = 3.15 #1.1
        self.thresh_min_grad_mag = 10
        self.thresh_max_grad_mag = 255
        self.kernel_size = 3
        self.false_counter = 0

        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.midpoint = 640
        self.pts = np.array([])

        self.history_right_curvature = []
        self.history_left_curvature = []
        self.history_right_line = []
        self.history_left_line = []

        self.state_right_curvature = []
        self.state_left_curvature = []
        self.state_right_line = []
        self.state_left_line = []

        self.smooth_factor = 0.95

        self.max_curvature = 10000

        self.debug = False

    def distortion_correction(self,image):
        """
        Distortion correction
        :param image: distorted image
        :return: undistorted image
        """
        return cv2.undistort(image, self.C, self.coeff, None, self.C)

    def create_thresholded_image(self,image):
        """
        Filters input image with sobel operator and gradients
        :param image: undistorted image
        :return: binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # binary = np.zeros_like(s)
        # binary[(s < 146)] = 1
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_color = np.array([0, 55, 74])
        upper_color = np.array([85, 255, 255])
        bin_color_img = cv2.inRange(hls, lower_color, upper_color)

        return bin_color_img

    def perspective_transform(self,image,src,dst):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        return warped,Minv


    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return grad_binary

    def mag_thresh(self, gray, sobel_kernel=3, thresh=(0, 255)):
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        magn = np.sqrt(sobelx * sobelx + sobely * sobely)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_magn = np.uint8(255 * magn / np.max(magn))
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(scaled_magn)
        mag_binary[(scaled_magn >= thresh[0]) & (scaled_magn <= thresh[1])] = 255
        # 6) Return this mask as your binary_output image
        return mag_binary

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi / 2)):

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        direction = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(direction)
        dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return dir_binary

    def find_polynomial(self,image,binary_warped):

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped, axis=0)
        left_histogram = histogram[:self.midpoint]
        right_histogram = histogram[self.midpoint:]


        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines

        left_base = np.argmax(left_histogram)
        right_base = np.argmax(right_histogram)

        if left_base == 0:
            leftx_base = self.history_left_line[-1]
            print('Left base not found')
        else:
            leftx_base = left_base
        if right_base == 0:
            rightx_base = self.history_right_line[-1]
            print('Right base not found')
        else:
            rightx_base = right_base + self.midpoint

        self.history_left_line.append(leftx_base)
        self.history_right_line.append(rightx_base)

        if len(self.history_left_line) == 1:
            self.state_left_line.append(leftx_base)
        else:
            self.state_left_line.append(self.smooth_factor*self.state_left_line[-1] +
                                        (1-self.smooth_factor)*leftx_base)
        if len(self.history_right_line) == 1:
            self.state_right_line.append(rightx_base)
        else:
            self.state_right_line.append(self.smooth_factor*self.state_right_line[-1] +
                                        (1-self.smooth_factor)*rightx_base)

        if self.debug:
            visualizer.plot_histogram(left_histogram,right_histogram)
            print(histogram.shape, left_base, right_base)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
            print('No lines found')
            self.false_counter += 1
            visualizer.save_image(image,'frame' + str(self.false_counter))
            return []

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # print(left_fit,right_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if self.debug:
            visualizer.plot_polynomial_fit(binary_warped,out_img,left_fitx,right_fitx,ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])

        if left_curverad > self.max_curvature:
            self.history_left_curvature.append(self.max_curvature)
        else:
            self.history_left_curvature.append(left_curverad)
        if right_curverad > self.max_curvature:
            self.history_right_curvature.append(self.max_curvature)
        else:
            self.history_right_curvature.append(right_curverad)

        if len(self.history_left_curvature) == 1:
            self.state_left_curvature.append(left_curverad)
        else:
            self.state_left_curvature.append(self.smooth_factor*self.state_left_curvature[-1] +
                                        (1-self.smooth_factor)*self.history_left_curvature[-1])
        if len(self.history_right_curvature) == 1:
            self.state_right_curvature.append(right_curverad)
        else:
            self.state_right_curvature.append(self.smooth_factor*self.state_right_curvature[-1] +
                                        (1-self.smooth_factor)*self.history_right_curvature[-1])

        # Now our radius of curvature is in meters
        # if self.debug:
        # print(left_curverad, 'm', right_curverad, 'm')

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # back up for next frame
        self.pts = pts

        return pts

    def process_image(self,image):
        """
        Test function to propagate a test image
        :return: None
        """

        # Read image shape
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_size = image.shape
        # print(img_size)

        # Distortion correction
        dist = self.distortion_correction(image)
        if self.debug:
            visualizer.plot_two_images(image,dist,'Distortion')

        # Get thresholded binary image
        binary_image = self.create_thresholded_image(dist)
        if self.debug:
            visualizer.plot_two_images(image,binary_image,'Thresholding')

        # Perspective transform
        src = np.float32([
            [560, 475],
            [205, img_size[0]],
            [1105, img_size[0]],
            [724, 475]
        ])
        dst = np.float32(
            [[(img_size[1] / 4), 0],
             [(img_size[1] / 4), img_size[0]],
             [(img_size[1] * 3 / 4), img_size[0]],
             [(img_size[1] * 3 / 4), 0]])
        # print(src,dst)
        topdown_image,Minv = self.perspective_transform(binary_image,src,dst)
        if self.debug:
            visualizer.plot_perspective_transform(binary_image,topdown_image,src,dst)

        pts = self.find_polynomial(image,topdown_image)

        if len(pts) == 0:
            pts = self.pts
            print('Previous points are taken')
            # print(self.pts)

        result = visualizer.get_result(image,topdown_image,pts,Minv,image,100,-0.23)

        if self.debug:
            visualizer.plot_two_images(image,result,'Process')
        return result

    def adjust_parameters(self,image):
        def nothing(x):
            pass
        
        trackbar = np.zeros((200, 560, 3), np.uint8)
        cv2.namedWindow('Parameter')
        cv2.createTrackbar('R min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('R max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('G min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('G max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('B min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('B max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('H min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('H max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('L min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('L max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('S min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('S max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('Sobel ker', 'Parameter', 0, 10, nothing)
        cv2.createTrackbar('SX min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('SX max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('SY min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('SY max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('GM min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('GM max', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('GO min', 'Parameter', 0, 255, nothing)
        cv2.createTrackbar('GO max', 'Parameter', 0, 255, nothing)

        while (True):
            # Read parameters
            hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            r_min = cv2.getTrackbarPos('R min', 'Parameter')
            r_max = cv2.getTrackbarPos('R max', 'Parameter')
            g_min = cv2.getTrackbarPos('G min', 'Parameter')
            g_max = cv2.getTrackbarPos('G max', 'Parameter')
            b_min = cv2.getTrackbarPos('B min', 'Parameter')
            b_max = cv2.getTrackbarPos('B max', 'Parameter')
            h_min = cv2.getTrackbarPos('H min', 'Parameter')
            h_max = cv2.getTrackbarPos('H max', 'Parameter')
            l_min = cv2.getTrackbarPos('L min', 'Parameter')
            l_max = cv2.getTrackbarPos('L max', 'Parameter')
            s_min = cv2.getTrackbarPos('S min', 'Parameter')
            s_max = cv2.getTrackbarPos('S max', 'Parameter')
            s_kernel = cv2.getTrackbarPos('Sobel ker', 'Parameter')
            sx_min = cv2.getTrackbarPos('SX min', 'Parameter')
            sx_max = cv2.getTrackbarPos('SX max', 'Parameter')
            sy_min = cv2.getTrackbarPos('SY min', 'Parameter')
            sy_max = cv2.getTrackbarPos('SY max', 'Parameter')
            gm_min = cv2.getTrackbarPos('GM min', 'Parameter')
            gm_max = cv2.getTrackbarPos('GM max', 'Parameter')
            go_min = cv2.getTrackbarPos('GO min', 'Parameter')
            go_max = cv2.getTrackbarPos('GO max', 'Parameter')

            # Filter HLS
            lower_rgb = np.array([r_min, g_min, b_min])
            upper_rgb = np.array([r_max, g_max, b_max])

            # Filter HLS
            lower_hls = np.array([h_min, l_min, s_min])
            upper_hls = np.array([h_max, l_max, s_max])

            # Result
            bin_rgb = cv2.inRange(hls, lower_rgb, upper_rgb)
            bin_hls = cv2.inRange(hls, lower_hls, upper_hls)
            bin_sx = ima

            cv2.imshow('Parameter', trackbar)
            cv2.moveWindow('Parameter', 1200, 800)
            # visualizer.plot_two_images(image,bin_color_img,'Thresholding')
            # Apply each of the thresholding functions
            gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=self.kernel_size,
                                          thresh=(self.thresh_min_sobel_x, self.thresh_max_sobel_x))
            grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=self.kernel_size,
                                          thresh=(self.thresh_min_sobel_y, self.thresh_max_sobel_y))
            mag_binary = self.mag_thresh(gray, sobel_kernel=g_kernel*2+1,
                                          thresh=(g_min, g_max))
            dir_binary = self.dir_threshold(gray, sobel_kernel=self.kernel_size,
                                          thresh=(self.thresh_min_grad_dir, self.thresh_max_grad_dir))

            combined = np.zeros_like(dir_binary)
            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

            combined_sobel = np.zeros_like(dir_binary)
            combined_gradient = np.zeros_like(dir_binary)
            combined_sobel[((gradx == 1) & (grady == 1))] = 1
            combined_gradient[((mag_binary == 1) & (dir_binary == 1))] = 1
            cv2.imshow('HLS', bin_color_img)
            cv2.moveWindow('HLS', 1200, 0)
            cv2.imshow('Gradient', mag_binary)
            # visualizer.plot_two_times_four_images([image,gradx,grady,combined_sobel,
            #                             combined,mag_binary,dir_binary,combined_gradient])
            # cv2.imshow('Output', bin_color_img)
            # cv2.imshow('Input', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cv2.destroyAllWindows()

# Color & Gradient Threshold

# Perspective Transform