# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project our goal is to create a software pipeline to detect vehicles in a video using the following steps:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps we normalize our features and randomize a selection for training and testing.
* Implement a sliding-window technique and use our trained classifier to search for vehicles in images.
* Run our pipeline on a video stream (out.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/figure_1.png
[image2]: ./output_images/figure_2.png
[image3]: ./output_images/figure_8.png
[image4]: ./output_images/figure_4.png
[image5]: ./output_images/figure_5.png
[image6]: ./output_images/figure_6.png
[image7]: ./output_images/figure_7.jpg
[video1]: ./project_video.mp4

Bringing in the Training Data
---
On `search_classify.py` line 19-31, I started by reading in all the `vehicle` and `non-vehicle` images. I print the count of each class respectively to obtain 8792 and 8968. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Extracting Histogram of Oriented Gradient Features from the Training Data
---
After obtaining the cars and not-car images, we extract their HOG features using the function extract_features() from `lesson_function.py` which calls on get_hog_features(). The features are extracted for each class as shown on `search_classify.py` lines 53 and 66 respectively.

At first the parameters for the HOG features were not obvious to select this required many trails and error. On `search_classify.py` line 37-47 my final selection was to use the YCrCb colorspace, oreint equal 9, 8 pix per cell, and 32 histogram bins. After fine-tuning these parameters I acheive 99% accuracy on my test set.

Here is an example of my selected HOG parameters.
![alt text][image2]

Normalizing the Extracted HOG Features
---

Before feeding the extracted HOG features we normalize them using sklearn.preprocessing's StandardScaler() as seen on `search_classify.py` lines 83-88. Here is what the features looked like before and after normalization.
![alt text][image4]


Training a Support Vector Classifier (SVC) using HOG Features
---

On `search_classify.py` lines 91 I define the labels vector and on line 95 I split the data into training and testing data, I chose a 90/10 split because of the limited amount of data and figured 10% training would offer sufficient samples to determine the accuracy. On line 105 I trained the SVC using the HOG features.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input Layer         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    |    outputs 10x10x16 				   									|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		|       Input = 5x5x16. Output = 400  									|
| Fully Connected				|        Input = 400. Output = 200									|
| RELU					|												|
| DROPOUT					|												|
| Fully Connected				|        Input = 200. Output = 120									|
| RELU					|												|
| DROPOUT					|												|
| Fully Connected				|        Input = 120. Output = 43									|

Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
16.58 Seconds to train SVC...
Test Accuracy of SVC =  0.9893
My SVC predicts:  [ 0.  0.  1.  1.  0.  0.  0.  0.  1.  1.]
For these 10 labels:  [ 0.  0.  1.  1.  0.  0.  0.  0.  1.  1.]
0.00083 Seconds to predict 10 labels with SVC





###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

Here is what the features looked like before and after normalization.
![alt text][image4]


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  





