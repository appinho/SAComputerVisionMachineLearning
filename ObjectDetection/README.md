# Term 3 Additional Project: Object Detection Lab

<p align="center">
  <img src="./ssd.gif">
</p>

### Mobile Nets

They are very efficient for mobile and embedded devices and purposed on running with high FPS and low memory footprint:
* Perform a depthwise convolution (Acts on each input channel separately with a different kernel and saves by a factor of N where N is the number of kernels for a usual convoluation) followed by a 1x1 convolution rather than a standard convolution (saves about 9 times the runtime)
* Use a "width multiplier" - reduces the size of the input/output channels, set to a value between 0 and 1
* Use a "resolution multiplier" - reduces the size of the original input, set to a value between 0 and 1

Of course, generally models with more paramters achieve a higher accuracy. MobileNets are no silver bullet, while they perform very well larger models will outperform them. MobileNets are designed for mobile devices, NOT cloud GPUs.

### SSD

The SSD architecture consists of a base network to find ROIs followed by several convolutional layers to fit the bounding box. Usually, each bounding box carries the box geometry (x,y,w,h) and the class probabilities (c1, ... , cp). However, SSd only detects where the box is and uses one of the k predetermined shapes.  
Q: Why does SSD use several differently sized feature maps to predict detections?  
A: Differently sized feature maps allow for the network to learn to detect objects at different
resolutions. This is illustrated in the figure with the 8x8 and 4x4 feature maps. This may remind you
of skip connections in fully convolutional networks.  
Q: What are some ways which we can filter nonsensical bounding boxes?  
A: The SSD paper does 2 things:  
1. Filters boxes based on IoU metric. For example, if a box has an IoU score
less than 0.5 on all ground truth boxes it's removed.  
2. *Hard negative mining*. This is a fancy way of saying "search for negatives examples
the highest confidence". For example, a box that misclassifies a dog as a cat with 80% confidence.
The authors of the SSD paper limit the positive to hard negative ratio to 3:1 at most. The actual positive to negative ratio is typically much higher and the number of boxes are typically reduced substantially.  
[SSD Code](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  
[SSD Paper](https://arxiv.org/abs/1611.10012)
