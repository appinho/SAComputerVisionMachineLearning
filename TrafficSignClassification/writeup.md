**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[image1]: ./report_images/traffic_sign_examples.png "3 examples of the data set"
[image2]: ./report_images/histogram_trainingdata.png "Histogram of training data"
[image3]: ./report_images/histogram_validationdata.png "Histogram of validation data"
[image4]: ./report_images/histogram_testdata.png "Histogram of test data"
[image5]: ./test_images/bumpy_road.jpg "Bumpy Road Traffic Sign"
[image6]: ./test_images/limit60.jpg "Speed limit 60 Traffic Sign"
[image7]: ./test_images/limit80.jpg "Speed limit 80 Traffic Sign"
[image8]: ./test_images/right_turn.jpg "Right Turn Traffic Sign"
[image9]: ./test_images/road_work.jpg "Road Work Traffic Sign"
[image10]: ./test_images/stop.jpg "Stop Traffic Sign"
[image11]: ./test_images/yield.jpg "Yield Traffic Sign"

### Step 1 Data Set Summary & Exploration

The German Traffic Sign Dataset can be found with the following link: http://benchmark.ini.rub.de

First, the pickle files of the training, validation and test data set are loaded. Next, the images and labels of the 3 data partitions are stored separately. All the images possess the format of 32x32 pixels and with the numpy library command "unique" all occuring label identifier for all images can be found. Alltogether, there are 43 different traffic signs within the complete data set. The whole data set is splitted up so that the training data possess 34799, the validation set 4410 and the test set 12630 examples. It is usual to use the majorities of examples to train comprehensively. The classification accuracy of the trained neural net is evaluated by the validation set later.

3 examples with their assigned label id as caption can be seen here:  

![alt text][image1]

The axis also show that the resolution of the images is indeed 32x32 pixels.  
Moreover, it is useful to plot the histogram of all occuring traffic sign per data set. They can be seen here:  

![alt text][image2] ![alt text][image3] ![alt text][image4]

All 3 histograms show that the number of examples per labels varies a lot. Usually, a uniformly distributed trainind data set leads to better classification results later. Therefore, a data augmentation step can be performed to find more examples of rare labels. Common methods for this data augmetation step are to rotate, translate, flip or add noise to the already existing examples. However, the invariance of each label class must be considered by performing one of the following transformations because a mirrored "30km/h speed limit" sign would lead to confusions within the dataset.

### 2 Design and Test a Model Architecture

As a first step, all images are converted into grayscale images to reduce the number of channels from 3 to 1. The paper of the later used network architecture recommends the usage of grayscale images instead of RGB images which can be read more in detail here: (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). As next step, the images are normalized to have 0 mean and equal variance to ensure that the input data is within the same range. Moreover, the gradient descent method together with the initialization of the weights and the biases of the neural net perform better with normalized data.
Here is an example of a traffic sign image before and after grayscaling.

The architecture of the chosen LeNet neural net can be summarized by the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalalized grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400 input, 120 output        									|
| RELU					|												|
| Fully connected		| 120 input, 84 output        									|
| RELU					|												|
| Fully connected		| 84 input, 43 output        									|
 
The training of the model is performed by 50 epochs to ensure a long enough training phase. For each epoch a batch size of 128 is used to find a good balance in enough updates of the weights and a big enough batch size to incorporate enough information for a gradient descent step. The weights and biases of the neural network are initialized by generating random values from a gaussian distribution with 0 mean and standard deviation of 0.1 to be within the same range as the normalized input data. The learning rate is chosen to be 0.001 so that the learning steps are not big enough to risk a diverging in the search of a minimum of the loss function and not small enough to make no significant changes within the parameters of the neural net. As optimizer the AdamOptimizer has been selected.

This set up ended up in an accuracy of 94,6% for the validation set.

### 3 Test a Model on New Images

Here are 7 German traffic signs that can be found on the web:  
![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

The right turn image has a blank white background whereas the other examples have the environment as their background. The challenge in this step is that these examples has not been used within the training of the model and are now used to analyze the robustness of the implemented solution.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road     		| Bumpy Road  									| 
| Speed limit 60     			| Speed limit 60									|
| Speed limit 80					| Speed limit 50											|
| Right Turn	      		| Right Turn				 				|
| Road Work			| Road Work      							|
| Stop      		| Stop					 				|
| Yield			| Yield      							|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This result is comparable to the achieved validation accuracy of 94,6% and would even diverge less if more test examples are used.

For the first image, the model is absolutely sure that this is a bumpy road sign (probability of 99,9%). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99804080e-01         			| Bumpy road   									| 
| 1.95957939e-04     				| Bicycles crossing 										|
| 1.67392972e-14					| Road narrows on the right											|
| 7.63634159e-24	      			| Slippery road					 				|
| 1.58469440e-28				    | Children crossing      							|


For the third image a misclassification occur and a "Speed limit 80km/h" sign is falsely classified as "Speed limit 50km/h" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Speed limit (50km/h)  									| 
| 2.59618493e-10     				| Speed limit (30km/h) 										|
| 3.93475183e-12					| Speed limit (80km/h)											|
| 4.36482955e-23	      			| Speed limit (60km/h)					 				|
| 1.08011173e-30				    | Yield      							|

It can be observed that the trained model is pretty sure that is a speed limit sign because the top 4 guesses are all coming from speed limits. Also the correct label is part of it on the third rank.  
However, the calculated probabilities show a really "hard" decision manner because at least for these 7 test examples it always result in a prediction with almost 100% certainty. This could be a hint of overfitting the training data and could be improved by an aforementioned data augmentation step or a regularization technique (Dropout, L2 regularization).
