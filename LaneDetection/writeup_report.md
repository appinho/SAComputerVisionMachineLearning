# Project: Advanced Lane Finding Project

[//]: # (Image References)

[image1]: ./output_images/camera_cal/corners13.png "Corner detection"
[image2]: ./output_images/camera_cal/distortion14.png "Chessboard distortion"
[image3]: ./output_images/test_image/distortion_correction.png "Test image distortion"
[image4]: ./output_images/test_image/colorspaces.png "Colorspaces"
[image5]: ./output_images/test_image/image_thresholding.png "Thresholded image"
[image6]: ./output_images/test_image/perspective_transform_straight.png "Perspective transform"
[image7]: ./output_images/test_image/lane_histograms.png "Histograms of lane information"
[image8]: ./output_images/test_image/polynomial_fit.png "Polynomial fit"
[image9]: ./output_images/test_image/result.png "Result"
[image10]: ./output_images/test_image/history.png "History of curvature and lane offset"

### Pipeline

The resulting video can be found on YouTube by clicking on the image below:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/joKuHeSrCAo/0.jpg)](https://www.youtube.com/watch?v=joKuHeSrCAo)

The executable code can be found in: `main.py`

### Preprocessing: Camera Calibration

Because each camera has its own characteristics, a calibration is necessary to obtain undistorted, clean images to further work on. Therefore, 20 images from different perspectives are taken of a chessboard pattern that is hanging at a wall. The chessboard pattern delivers a high contrast structure to find corners that are used to calculate the distortion factor.  

The code for this step is contained in the file: `calibration.py`

A loop over all calibration images is performed. In each iteration, the image is converted into grayscale first. Then the OpenCV command `findChessboardCorners()` is executed to obtain the detected corners of the images. In the case the correct amount of corners are detected, for a chessboard with 10 rows and 7 columns it would mean (10-1)x(7-1)= 54 corners, the result is stored in an image point vector. Moreover, a object point vector is filled that has the undistorted coordinates in it and where the z coordinate is 0, because these points should all lay in one plane. 

An example of the detected corners can be seen here:

![alt text][image1]

After all calibration images append their results to the image and object point vectors, these vectors are used within the OpenCV command `calibrateCamera()` which returns, among other things, the calibration matrix and the distortion coefficients.  

An example of this step can be seen here:

![alt text][image2]

### Pipeline processing

An object of the class in `lane_finding.py` is initialized that contains all functions for the entire processing pipeline. To demonstrate the pipeline one input image in processed step by step. Later this pipeline is simply executed for each frame of a video which results in the output video `output_video.mp4`.

#### Step 1: Undistort input image.

First, the input image is undistorted based on the found information about the calibration matrix and the distortion coefficients from the preprocessing step.  

The code can be found in `lane_finding.py` in function `distortion_correction()` and the result looks like this:

![alt text][image3]

Especially, if you look at the edges of both images you can see the effect of the distortion correction.

#### Step 2: Threshold the undistorted image to obtain lane information

In this step, the undistorted image is investigated for its lane information. Therefore, each channel or the RGB and HLS are displayed. Moreover, the Sobel Operator in x- and y- direction is applied on the grayscale version of the image as well as the Gradient Magnitude and Gradient Orientation.  

The code can be found in `visualizer.py` in function `plot_colorspaces()` and applied on the test image all following channels can be seen:  

![alt text][image4]

A combination of the Red Channel and the Saturation Channel are used. Therefore, each channel is filtered by a minimum value and both results are combined by an "AND" operator.  

The code can be found in `lane_finding.py` in function `image_thresholding()` and applied on the test image it looks like this:  

![alt text][image5]

#### 3. Perspective transform 

The code for the perspective transform includes a function called `perspective_transform()` in `lane_finding.py`.
For the test image `straight_lines1.jpg` the trapezium and the resulting warped rectangle are defined as:

```python
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
```

The result of the perspective transform can be seen here:

![alt text][image6]

#### 4. Polynomial fit

The code of the polynomial fit can again be found in `lane_finding.py` under the function `fit_polynomial()`.
Then, the warped image is investigated for peaks in the histogram of their x-values to find the x-position of the lane beginnings. The beginnings are meant to be the starting at the bottom of the image or in other words as close as possible to the ego vehicle.

The resulting histograms of the test image can be seen here:  

![alt text][image7]

The clear peaks indicate that finding the beginning of the lanes is not to hard for this example.  

Next, a polynomial of second degree is fit through the defined sliding windows. The result of the polynomial fit can be seen here:

![alt text][image8]

#### 5. Calculation of curvature and lane offset

The calculations can be found in `lane.py` within the methods `calculate_curvature()` and `calculate_lane_offset()`.
The curvature and the lane offset can be determined by using the found polynomial fit f(y) = A*y^2 + B*y + C. For the lane curvature A and B of the equation are used whereas the lane offset is calculated by the coefficient C. It is important to mention that a transformation from pixel values into real world meters is performed.

#### 6. Result of pipeline

The resulting detected lane as well as the curvature and the lane offset are integrated in the following resulting picture:

![alt text][image9]

For the video input `project_video.mp4`, a sanity check is included that filters out odd polynomial fits and if no pixels are found that would represent a lane. Then, the last decent estimation of the lane is taken. Moreover, a smoothing of the lane curvature and the lane offset is integrated to avoid odd jumps within the solution.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
