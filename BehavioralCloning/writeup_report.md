# Project: Behavioral Cloning

[//]: # (Image References)

[image1]: ./report_images/nvidia_net.png "Neural net architecture"
[image2]: ./report_images/image.jpg "Training data image"
[image3]: ./report_images/flipped_image.jpg "Flipped training data image"
[image4]: ./report_images/left_image.jpg "Left training data image"
[image5]: ./report_images/right_image.jpg "Right training data image"
[image6]: ./report_images/hist.png "Histogram of steering angles"

### 1 Final Model Architecture

The chosen neural net architecture follows the model of the paper "End to End Learning for Self-Driving Cars" from NVIDIA (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  
The subsequent layers with their attributes are listed in the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							|
| Cropping2D         		| 66x200x3 RGB image   							|
| Normalization        		| 66x200x3 normalized RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 24 kernels 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same, outputs 31x98x24				|
| Convolution 5x5	    | 1x1 stride, 36 kernels      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same,  outputs 14x47x36 				|
| Convolution 5x5	    | 1x1 stride, 48 kernels      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same,  outputs 5x22x48				|
| Convolution 3x3	    | 1x1 stride, 64 kernels, padding=valid      									|
| RELU					|	outputs 3x20x64											|
| Convolution 3x3	    | 1x1 stride, 64 kernels, padding=valid      									|
| RELU					|	outputs 1x18x64		
| Flatten       | outputs 1152  |
| Fully connected		| outputs 100        									|
| Fully connected		| outputs 50        									|
| Fully connected		| outputs 10        									|
| Fully connected		| outputs 1        									|

The model first crops the input with an additional Keras Cropping layer to remove parts next to the track (e.g. vegetation/sky).
Next, the input is normalized using a Keras lambda layer to have pixel values between -0.5 and 0.5 to ensure good initialization.
Then, 5 Convolution Layers are performed in which each layer includes an RELU activation function to introduce nonlinearities.
Moreover, the first 3 Convolutions are followed by 2x2 stride MaxPool Layers to narrow down the feature space.
Towards the end, the network contains 4 fully connected layers to break down the information flow to one single output neuron which models the resulting steering angle.

### 2 Attempts to reduce overfitting in the model

The model performs well without any further regularization methods.
Nevertheless, dropout layers could be integrated after each pooling layer to generalize the model even better.

### 3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Moreover, as loss function the Mean-Squared-Error (MSE) was chosen which is a common loss function for regression models since the steering angle output is a floating number.

### 4 Appropriate training data

Training data was chosen to keep the vehicle driving on the road and different kind of scenarios were recorded to ensure that the model is not overfitting. Therefore, 2 counter-clockwise loops and 1 clockwise loop of the track were recorded to generalize the model better in both directions. Additionally, several safety maneuvers, which navigate back from the left or right edge of the track towards the center of it, were recorded in both driving directions. Specifically, on the bridge and in tight curves more recovering data was recorded to generalize better on different textures (tiles on bridge) and rare driving cases (like tight curves). For details about how I came up creating this training set, see the next section. 

### 5 Solution Design Approach

The first step was to use a single-layer fully connected network to simply setup the whole pipeline.
Also, only one recorded driven loop was used to evaluate the first try.
This model had big errors since the input was not normalized but the pipeline was already working.  

The second step was to introduce a normalization layer at the beginning of the network with Keras lambda layer.
All three image channels were normalized to have bounding pixel values between -0.5 and 0.5.
This reduced the losses and garantueed that the initialized weights and biases are within the same range as the input.
Nevertheless, the network structure was to simply to keep the car in the lane.

The third step was to use the LeNet architecture which uses Convolution Layers that are able to detecting complex shapes like edges or curves of the track.
However again, the car was not able to drive autonomously because whenever it drifted off the lane's center it was not able to recover from that offset.  

This is why, multiple recovering data were recorded.
As a result, the car tended to drive leftwards.
This is due to the fact, that the recording data was only counter-clockwise which means that more left curves were within the dataset than right curves. Therefore, the input images were flipped and the according steering angles were negated which augments the dataset and removes any tendency of right and left curves.
As an example you can see the following two images which show one training image and its flipped version plus their steering angles:

![alt text][image2]
![alt text][image3]

Moreover, the images from the left and right camera images were used with a correction factor of +-0.2 respectively.
These left and right images from the example frame above can be seen here plus their steering angles:

![alt text][image4]
![alt text][image5]

This augmentation technique leads to 6 images for each recorded time frame.
Now, the car could drive up to the bridge but couldn't handle narrow curves.  

This is why further data was recorded. First, one more loop in each directions were recorded to have in sum 3 loops of data in both directions to generalize better. Addtionally, multiple recovering manoveurs were recorded on the bridge and in narrow curves to handle these kinds of corner cases.  

The resulting histogram of steering angles can be seen here:

![alt text][image6]

The three peaks occur because most of the recording time the car drives straight which implies small steering angles around 0. With the above mentioned augmentation technqiue these values are further processed by the correction factor of 0.2 for the left and right recorded images. Other than this, the histogram shows that the entire range of steering angles is covered so the trained model should also handle tighter curves.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by a 80/20 ratio respectively. I found that my first tries had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  

Finally, the LeNet architecture was replaced by the above mentioned NVIDIA end-to-end network structure. This more comprehensive architecture is useful to learn more feature channels and also possess more convolutions layers as well as fully-connected layers to learn more dependencies between the bigger input data set and the output.

The architecture can be seen here and all details are already mentioned in Section 1:

![alt text][image1]

Now, the training and validation losses were both low and within the same region which shows that the overfitting was avoided.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road which can be seen in the output video "video.mp4".
