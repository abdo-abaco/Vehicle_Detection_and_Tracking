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
[image8]: ./output_images/figure_8.jpg
[video1]: ./output_project_video.mp4

Bringing in the Training Data
---
On `search_classify.py` line 19-31, I started by reading in all the `vehicle` and `non-vehicle` images. I print the count of each class respectively to obtain 8792 and 8968. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Extracting Histogram of Oriented Gradient Features from the Training Data
---
After obtaining the cars and not-car images, we extract their HOG features using the function extract_features() from `lesson_function.py` which calls on get_hog_features(). The features are extracted for each class as shown on `search_classify.py` lines 53 and 66 respectively.

At first the parameters for the HOG features were not obvious to select this required many trails and error. On `search_classify.py` line 37-47 my final selection was to use the YCrCb colorspace, oreint equal 9, 8 pix per cell, and 32 histogram bins. The goal was to fine-tune these parameters so I acheive 99% accuracy on my test set.

Here is an example of my selected HOG parameters.
![alt text][image2]

Normalizing the Extracted HOG Features
---
Before feeding the extracted HOG features we normalize them using sklearn.preprocessing's StandardScaler() as seen on `search_classify.py` lines 83-88. Here is what the features looked like before and after normalization.
![alt text][image4]

Training a Support Vector Classifier (SVC) using HOG Features
---
On `search_classify.py` lines 91 I define the labels vector and on line 95 I split the data into training and testing data, I chose a 90/10 split because of the limited amount of data and figured 10% training would offer sufficient samples to determine the accuracy. On line 105 I trained the SVC using the HOG features.

The Support Vector Classifier and the accompanying parameters get stored `svc_pickle.p` as shown on `search_classify.py` lines 119-127. This generating file is used in the rest of the project as I discuss `project.py`. I record the results of the experience on the table below:


| Test Set         		|     Result	        					| 
|:---------------------:|:---------------------------------------------:| 
| orientations        		| 9 							| 
| pixels per cell         		| 8  							| 
| cells per block         		| 2  							| 
| Feature vector length         		| 8460  							| 
| Seconds to train SVC     	| 16.58 |
| Test Accuracy of SVC					|	0.9893 |
| My SVC predicts	      	| 				|
| 0    |    0				   									|
| 0    |    0				   									|
| 1    |    1				   									|
| 1    |    1				   									|
| 0    |    0				   									|
| 0    |    0				   									|
| 0    |    0				   									|
| 0    |    0				   									|
| 1    |    1				   									|
| 1    |    1				   									|
| Seconds to predict 10 labels with SVC   |    0.00083 				   									|

Sliding Window Search and Detection
---
Now that we have a classifier we are ready to use it to find vehicles in an image. Unfortunately, we cannot just feed the entire image to the classifer we must segment the entire image into overlapping boxes of different scales and slide these 'windows' across the entire image. This requires a great deal of processing.

Initially the boxes are not overlapping and are scanned across the entire image as shown below:
![alt text][image5]

To save some processing, I select to scan only the pixels in the vertical range between 400 and 656 as shown in `project.py` line 132 and 133. Due to the cars being near and far from the camera the resulting car images appear to be small or big depending on their distance from the camera, therefore multiple scales are used.

To save processing I limit my scales to 0.9 and 1.5 as shown on line 134. This captures most of the range, however, this does not perform well for extremely close or extremly far vehicles. 

These new search parameters get passed along with the classifier parameters to find_cars() on line 146 which detects vehicles and returns bounding boxs for their location. A heatmap is generated which increments a pixel location for every pixel in a bounding box.

With the scales set to 0.9 and 1.5, I chose to have cells_per_step = 1 which provides sufficient overlap and increases accuracy. To improve reliability and prevent many false positives we add the heatmap from the previous frame and increase the threshold on lines 148 to 159.

To find the final boxes from the heatmap I use label function from scipy.ndimage.measurements as shown in line 168. Both the heatmap and resulting detections are shown below.

![alt text][image7]
![alt text][image8]

Video Implementation 
---
Here's a [link to my video result](./output_project_video.mp4)

Discussion
---
Scaling proved to be a problem as adding more scales significantly reduced performance by requiring many processing. By limiting the scales to two values we ran the risk of not properly detecting vehicles in other scales. False positive detections forced us to increase our threshold which ran the risk of some true positives being thresholded. To make this more robust I suggest using more modern computer vision and machine learning techniques as these techniques can be outdated with the rapid changes in this field.





