## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction in addition to color histogram and binned color feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and utilize the trained classifier to search for vehicles in images.
* Run the aforementioned pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The code for this project is distributed among several files:
- trainingModel.py: Training and saving a classifier for vehicle detection
- sampleTest.py: Test the trained model across different images
- featureExtract.py: Contains functions to extract the image features required for vehicle detection
- mainCode.py: The pipeline where the trained model is applied on several images and video to detect vehicles
- cars.py: Defining a class that processes the results of the model and keeps track of vehicles across frames
- findLines.py: The pipeline for lane detection
- laneLine.py: Defining a class to keep track of lane lines across multiple frames

[//]: # (Image References)
[image1]: ./report_images/sample_images.jpg
[image2]: ./report_images/HOG_images.jpg
[image3]: ./report_images/classifier_test.jpg
[image4]: ./report_images/sliding_window.jpg
[image5]: ./report_images/boxed_window.jpg
[image6]: ./report_images/heatmaps.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I have considered the rubric points individually and this report describes how I addressed each point in my implementation.

---
### Writeup / README

### Training the model

The code associated with the training process is available in the 'trainingModel.py' file of the repository which applies three main steps:

#### 1. Reading data

The machine learning model developed in the code utilizes data from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). The first step is reading data from a zip file for these labeled data bases for the 'vehicle' and 'non-vehicle' cases separately. The corresponding code can be seen in lines 40-64 and 84-95 of 'trainingModel.py'. A sample of these images can be seen below:

![alt_text][image1]

#### 2. Feature Extraction

The feature extraction process contains three types, Histogram of Oriented Gradients (HOG), Color Histograms, and spatial binning features. The code for these steps is provided in 'featureExtract.py'

##### 2.1 Histogram of Oriented Gradients (HOG)

The 'skimage.hog()' function was used for this purpose as seen in lines 7-26 of 'featureExtract.py'. Different parameters of `orientations`, `pixels_per_cell`, and `cells_per_block` were tested to arrive to a model, however they did not provide much of a difference toward the end results. Below is example of applying HOG on a vehicle and non-vehicle images in 'YUV' color space with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt_text][image2]

##### 2.2 Color histograms

In addition HOG, color histogram features were applied and included as an added input to the trained model. The application can be seen in lines 36-48 of 'featureExtract.py' where histogram features were obtained using 'numpy.histogram' for each channel individually and the concatenated together. The main parameter in this case is the number of bins, which was selected as 16 which is reasonable given the range of 0-255.

##### 2.3 Spatial Binning

Spatial binning was also used as a feature for training the model where 'cv2.resize' and 'ravel' where used to obtain a 1D list of the values of a downscaled image. This step can be seen in lines 29-33 of 'featureExtract.py' and the tunable parameter in this case is the 'spatial_size' which was selected to be (16, 16).

The main parameters that remain to choose are the color space and color channels to use for the feature extraction process. 'YCrCb' and 'YUV' provided the highest accuracies upon training. However, upon testing other sets of images, a couple of shortcomings were observed. The models trained using 'YCrCb' and 'YUV' were very susceptible to shadows which caused several false positive detections. Additionally, the detection results were affected by the colors of the vehicles in an image. In order to compensate for these shortcomings, the saturation channel of the 'HLS' color space was added to the feature extraction and training processes in place of the third channel of the 'YUV' color space. The final color spaces and channels used were: * All channels of 'YUV' color space * Saturation channel of 'HLS' color space

The final selected parameters can be seen in lines 22-31 of 'trainingModel.py'.

#### 3. Training the model

Once all features are extracted, the next step was training a classifier. A SVC was implemented using sklearn.SVC with the 'kernel' chosen as 'rbf' and the default values of 'C' and 'gamma' were used. Prior to passing the data to the classifier, the features and labels of the dataset were first divided into a training and testing set using 'sklearn.model_selection.train_test_split' and the features were normalized using 'sklearn.preprocessing.StandardScaler'. The model was then fitted using the training set and evaluated using the testing set yielding and accuracy of 99.2% using the features described in the previous section. The model was then saved to be used later using the 'pickle' package. These steps can be observed in lines 123-139 of 'trainingModel.py'. The figure below shows the result of the testing the classifier on random set of the training and testing data:

![alt_text][image3]


### Sliding window search

In order to detect vehicles in an image, a sliding window search method was adopted where each window is then evaluated using the classifier discussed in previous sections. These windows were applied on different scales depending on the location in the image, where smaller windows were used for far objects and vice versa. These windows were then scaled back to an image size of (64,64) in order to be in line with images used in training the model. Additionally, since the location at which a vehicle exist is limited in an image, the sliding windows were also limited to the bottom half of the image. The image below shows the windows used in detecting vehicles in my code, where they were applied on three different scales: 1,1.5 and 2, which provided a good balance of accurate detection along with minimizing the required computations.

![alt_text][image4]
* The actual implementation of sliding window search included more intersecting windows, making it hard to visualize.

Rather than extracting HOG features for each window separately, HOG was applied to the entire image for all the required color spaces and channels. Color histograms and spatial binning were then obtained for each window separately and the features were passed to the classifier to determine whether each window contains a vehicle or not. The results of applying these steps to several sample images can be seen below:

![alt_text][image5]

The implementation of the sliding window search can be observed in the 'find_cars' function in lines 81-204 of 'mainCode.py'

### Processing positive windows

Once positive detections were observed, a filtering process was carried out in 'cars.py' to identify vehicle positions. This process mainly consists of obtaining a heatmap and thresholding it. The heatmap was obtained by basically adding up all the detected positions the a threshold was applied to filter out false positives. The heatmaps of several examples are shown below:

![alt_text][image6]

Finally, `scipy.ndimage.measurements.label()` was utilized to identify individual blobs in the heatmap and each blob was assumed to corresponded to a different vehicle. Bounding boxes were constructed over these to cover the area of each blob detected as illustrated in the examples below:

![alt_text][image7]

### Video Implementation

All of the steps described above were also implemented on a video by obtaining separate frame using the 'moviepy' package. The additional step for vehicle detection in video was averaging the heatmaps over different frames which can be seen in lines 47-49 of 'cars.py'. 

Lanes detection feature was also added to the pipeline, detailed explanation of lane detection is provided in this [project](https://github.com/abdullaayyad96/CarND-Advanced-Lane-Lines).

Here's a [link to my video result](./test_videos/project_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main shortcoming of my implementation of vehicle detection is high computational requirements making it unsuitable for real-time implementation. The main cause for this problem is utilizing two  different color spaces which feature extraction is applied to. Reasonabily accurate results however can be obtained by using a single color space. Additionally, the SVC utilized is a bit complex although good solutions are obtainable with smaller linear SVC.
In order to reduce the computational requirments a smarter way of implementing the sliding window search should be adopted where a full search across the entire frame are only taken at a low frequency and restricting window search to areas around a previously detected object in most frames. 

