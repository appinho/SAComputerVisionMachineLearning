# Imports
import numpy as np

# Class to store characteristics of a Lane
class Lane():
    def __init__(self, image_x_center):
        self.image_x_center = image_x_center
        self.current_lane_center = None
        self.history_lane_center = []
        self.history_smoothed_lane_center = []
        self.current_curvature = None
        self.history_curvature = []
        self.history_smoothed_curvature = []
        self.lane_pixels = np.array([])
        self.polynomial = np.array([])
        self.history_polynomial = []
        self.last_pts = np.array([])
        self.max_curvature = 10000
        self.lane_offset = 0
        self.nonzero_x = np.array([0])
        self.nonzero_y = np.array([0])
        self.smoothing_factor = 0.95
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/930
        self.A_max = 0.0003
        self.C_diff_max = 100
        # was the line detected in the last iteration?
        # self.detected = False
        # x values of the last n fits of the line
        # self.recent_xfitted = []
        # #average x values of the fitted line over the last n iterations
        # self.bestx = None
        # #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        # #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        # #radius of curvature of the line in some units
        # self.radius_of_curvature = None
        # #distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # #x values for detected line pixels
        # self.allx = None
        # #y values for detected line pixels
        # self.ally = None

    def update_lane_center(self,histogram,side):
        """
        Update the current lane center and its history
        :param histogram: Histogram of binary hits in x direction
        :param side: left or right lane indicator
        :return: None
        """

        # Determine peak in histogram
        peak = np.argmax(histogram)

        # If there is no peak
        if peak == 0:
            "No peak found"
        # If there is a peak
        else:
            # For the left lane
            if side == 'left':
                self.current_lane_center = peak
            # For the right lane
            else:
                self.current_lane_center = peak + self.image_x_center
        # Append lane center to its history
        self.history_lane_center.append(self.current_lane_center)

        # Add smoothed lane center information
        if len(self.history_smoothed_lane_center) == 0:
            self.history_smoothed_lane_center.append(self.current_lane_center)
        else:
            self.history_smoothed_lane_center.append(
                self.smoothing_factor * self.history_smoothed_lane_center[-1] +
                (1 - self.smoothing_factor) * self.current_lane_center)

    def fit_polynomial(self, binary_image, n_windows, window_height, margin, min_pix):
        """
        Fit polynomial through the white pixels of a binary image
        :param binary_image: Input binary image
        :param n_windows: Number of sliding windows
        :param window_height: Window height in pixel
        :param margin: Margin of window in pixel
        :param min_pix: Minimum number of pixels
        :return: Boolean if polynomial has been found
        """
        lane_inds = []
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_image.shape[0] - (window + 1) * window_height
            win_y_high = binary_image.shape[0] - window * window_height
            win_x_low = self.current_lane_center - margin
            win_x_high = self.current_lane_center + margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > min_pix pixels, recenter next window on their mean position
            if len(good_inds) > min_pix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        self.lane_pixels = np.concatenate(lane_inds)

        # Sanity check
        sanity = True
        # If no pixels for the lane have been found
        if len(self.lane_pixels) == 0:
            sanity = False
        else:
            # Fit polynomial of degree 2
            self.nonzero_x = nonzerox[self.lane_pixels]
            self.nonzero_y = nonzeroy[self.lane_pixels]
            polynomial = np.polyfit(self.nonzero_y, self.nonzero_x, 2)
            # If A is unreasonable high
            if abs(polynomial[0]) > self.A_max:
                sanity = False
            # If the offset of the found lane changed too much
            if len(self.history_polynomial) > 0 and \
                            abs(polynomial[2]-self.history_polynomial[-1][2]) > self.C_diff_max:
                sanity = False

        # In case no lane pixels have been found or the polynomial seems unreasonable
        # take most recent solution
        if not sanity and len(self.history_polynomial) > 0:
            polynomial = self.history_polynomial[-1]


        self.polynomial = polynomial
        self.history_polynomial.append(polynomial)


        # Smooth polynomial
        # if len(self.polynomial) == 0:
        #     self.polynomial = polynomial
        # else:
        #     self.polynomial = (self.smoothing_factor*self.polynomial) + \
        #                           (1-self.smoothing_factor)*polynomial

        x_meters = self.nonzero_x*self.xm_per_pix
        y_meters = self.nonzero_y*self.ym_per_pix
        self.polynomial_meters = np.polyfit(y_meters, x_meters, 2)

        return sanity

    def calculate_curvature(self):
        y_eval = 720
        curvature = ((1 + (2*self.polynomial_meters[0]*y_eval*self.ym_per_pix
                                  + self.polynomial_meters[1])**2)**1.5) \
                         / np.absolute(2*self.polynomial_meters[0])

        if curvature > self.max_curvature:
            curvature = self.max_curvature

        self.current_curvature = curvature
        self.history_curvature.append(self.current_curvature)

        # Add smoothed curvatyre information
        if len(self.history_smoothed_curvature) == 0:
            self.history_smoothed_curvature.append(self.current_curvature)
        else:
            self.history_smoothed_curvature.append(
                self.smoothing_factor * self.history_smoothed_curvature[-1] +
                (1 - self.smoothing_factor) * self.current_curvature)

    def get_polynomial_points(self, ploty, side):
        fit_x =self.polynomial[0] * ploty ** 2 \
                        + self.polynomial[1] * ploty \
                        + self.polynomial[2]
        if side == 'left':
            pts = np.array([np.transpose(np.vstack([fit_x, ploty]))])
        else:
            pts = np.array([np.flipud(np.transpose(np.vstack([fit_x, ploty])))])

        self.last_pts = pts
        return pts, fit_x

    def plot_polynomial(self, output_image, side):
        if side == 'left':
            output_image[self.nonzero_y, self.nonzero_x] = [255, 0, 0]
        else:
            output_image[self.nonzero_y, self.nonzero_x] = [0, 0, 255]

    def calculate_lane_offset(self):
        self.lane_offset = (self.current_lane_center-self.image_x_center)*self.xm_per_pix
        return self.lane_offset