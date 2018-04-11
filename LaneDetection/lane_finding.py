import cv2
import visualizer
import numpy as np
import image_processing
import lane

debug = False

class LaneFinder():
    def __init__(self,C, coeff, image_size):
        """
        Constructor
        :param C: Calibration matrix
        :param coeff: Distortion coefficients
        """

        # Store calibration information
        self.C = C
        self.coeff = coeff

        # Save image information
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.image_x_center = int(self.image_width/2)

        # Perspective transform
        self.trapezium = np.float32([
            [560, 475],
            [205, self.image_height],
            [1105, self.image_height],
            [724, 475]
        ])
        self.rectangle = np.float32(
            [[(self.image_width / 4), 0],
             [(self.image_width / 4), self.image_height],
             [(self.image_width * 3 / 4), self.image_height],
             [(self.image_width * 3 / 4), 0]])

        # Image thresholding information after running from `adjust_parameters()`
        self.lower_HLS_threshold = np.array([0, 55, 74])
        self.upper_HLS_threshold = np.array([85, 255, 255])

        # Left lane information
        self.left_lane = lane.Lane(self.image_x_center)

        # Right lane information
        self.right_lane = lane.Lane(self.image_x_center)

        # Polynomial fit option
        # Number of sliding windows
        self.nwindows = 9
        self.window_height = np.int(self.image_height / self.nwindows)
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

        # Debug counter
        self.false_counter = 0

    def process_image(self,image):
        """
        Processing pipeline
        :param image: Input image
        :return: Resulting image with highlighted lane information
        """

        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1 Distortion correction
        undistorted_image = self.distortion_correction(image)
        if debug:
            visualizer.plot_two_images(image, undistorted_image, 'Distortion correction',
                                       'output_images/test_image/distortion_correction')

        # 2 Image thresholding
        thresholded_image = self.image_thresholding(undistorted_image)
        if debug:
            visualizer.plot_two_images(undistorted_image, thresholded_image, 'Image Thresholding',
                                       'output_images/test_image/image_thresholding')

        # 3 Perspective transform
        warped_image,M,Minv = self.perspective_transform(thresholded_image)
        if debug:
            visualizer.plot_perspective_transform(thresholded_image, warped_image,
                self.trapezium,self.rectangle,'output_images/test_image/perspective_transform')

        # 4 Fit polynomial
        polygon_points = self.fit_polynomial(warped_image)

        # 5 Create output image
        average_curvature = (self.left_lane.history_smoothed_curvature[-1] +
                             self.right_lane.history_smoothed_curvature[-1])/2
        average_lane_offset = (self.left_lane.calculate_lane_offset() + self.right_lane.calculate_lane_offset())/2
        if len(polygon_points)>0:
            result = visualizer.get_result(image,warped_image,polygon_points,Minv,
                                           undistorted_image,average_curvature,-average_lane_offset)
        else:
            result = image

        if debug:
            visualizer.plot_result(result,'output_images/test_image/result')

        bgr_result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # Return result
        return bgr_result

    def distortion_correction(self,image):
        """
        Distortion correction
        :param image: distorted image
        :return: undistorted image
        """
        return cv2.undistort(image, self.C, self.coeff, None, self.C)

    def image_thresholding(self,image):
        """
        Filters input image to keep lane information
        :param image: undistorted image
        :return: thresholded image
        """

        # Convert image into HLS colorspace
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Threshold image with found thresholds from `adjust_parameters()`
        thresholded_image = np.zeros_like(rgb[:,:,0])
        thresholded_image[(rgb[:,:,0]>45) & (hls[:,:,2]>110)] = 255
        # thresholded_image = cv2.inRange(hls, self.lower_HLS_threshold, self.upper_HLS_threshold)
        # Return thresholded image
        return thresholded_image

    def perspective_transform(self,image):
        """
        Performs perspective transform on the image
        :param image: binary image
        :return: topview binary image & Perspective transform matrices
        """

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(self.trapezium, self.rectangle)
        M_inv = cv2.getPerspectiveTransform(self.rectangle, self.trapezium)
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        # Return results
        return warped, M, M_inv

    def find_peak(self,histogram):
        peak = np.argmax(histogram)
        if peak == 0:
            "No peak found"
        return peak

    def fit_polynomial(self,binary_image):

        # Take a histogram of the image in x direction
        histogram = np.sum(binary_image, axis=0)
        histogram_left = histogram[:self.image_x_center]
        histogram_right = histogram[self.image_x_center:]
        if debug:
            visualizer.plot_histogram(histogram_left,histogram_right,
                                      'output_images/test_image/lane_histograms')

        # Update the lane center
        self.left_lane.update_lane_center(histogram_left,'left')
        self.right_lane.update_lane_center(histogram_right, 'right')

        # Find lane ids
        left_lane_found = self.left_lane.fit_polynomial(binary_image,self.nwindows,self.window_height,
                                      self.margin,self.minpix)
        right_lane_found = self.right_lane.fit_polynomial(binary_image,self.nwindows,self.window_height,
                                      self.margin,self.minpix)

        ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
        if not left_lane_found:
            # print('No left lines found')
            self.false_counter += 1
            # visualizer.save_image(binary_image,'frame' + str(self.false_counter))
            pts_left = self.left_lane.last_pts
        else:
            pts_left, left_fitx = self.left_lane.get_polynomial_points(ploty,'left')

        if not right_lane_found:
            # print('No right lines found')
            self.false_counter += 1
            # visualizer.save_image(binary_image,'frame' + str(self.false_counter))
            pts_right = self.right_lane.last_pts
        else:
            pts_right, right_fitx = self.right_lane.get_polynomial_points(ploty,'right')

        if debug:
            out_img = np.dstack((binary_image, binary_image, binary_image)) * 255
            # Generate x and y values for plotting
            self.left_lane.plot_polynomial(out_img,'left')
            self.right_lane.plot_polynomial(out_img, 'right')
            visualizer.plot_polynomial_fit(binary_image,out_img,left_fitx,right_fitx,ploty,
                     'output_images/test_image/polynomial_fit')

        pts = np.hstack((pts_left, pts_right))


        # Calculate curvature
        self.left_lane.calculate_curvature()
        self.right_lane.calculate_curvature()

        return pts


    def adjust_parameters(self,image):

        # Empty callback function
        def nothing(x):
            pass

        # Define trackbar
        trackbar = np.zeros((1, 560, 3), np.uint8)
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

        # Run loop to be able to continuously change parameters
        while (True):

            # Convert image
            rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Read parameters
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
            bin_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)
            bin_hls = cv2.inRange(hls, lower_hls, upper_hls)
            bin_sx = image_processing.sobel_operator(
                gray,orient='x',sobel_kernel=s_kernel,thresh=[sx_min,sx_max])
            bin_sy = image_processing.sobel_operator(
                gray,orient='y',sobel_kernel=s_kernel,thresh=[sy_min,sy_max])
            bin_gm = image_processing.gradient_magnitude(
                gray,sobel_kernel=s_kernel,thresh=[gm_min,gm_max])
            bin_go = image_processing.gradient_orientation(
                gray,sobel_kernel=s_kernel,thresh=[go_min,go_max])

            combined = np.zeros_like(bin_rgb)
            # combined[((bin_rgb == 1) & (bin_hls == 1)) & ((bin_sx == 1) & (bin_sy == 1))
            #          & ((bin_gm == 1) & (bin_go == 1))] = 255
            combined[(bin_rgb > 0) & (bin_hls > 0)] = 255

            cv2.imshow('Result', combined)
            cv2.imshow('Parameter', trackbar)
            cv2.moveWindow('Parameter', 0, 800)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()