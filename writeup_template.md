## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction in addition to color histogram and binned color feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and utilize the trained classifier to search for vehicles in images.
* Run the aforementioned pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I have considered the rubric points individually and this report describes how I addressed each point in my implementation.

---
### Writeup / README

### Training the model

The code for this step can be seen in the 'trainingModel.py' file of the repository which applies three main steps:

#### 1. Reading data

The machine learning model developed in the code utilizes data from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). The first step is reading data from a zip file for these dlabeled ata bases for the 'vehicle' and 'non-vehicle' cases seperately. The correspodning code can be seen in lines 40-64 and 85-96 of 'trainingModel.py'. A sample of these images can be seen below:

[alt_text][image1]

#### 2. Feature Extraction

The feature extraction process contains three types, Histogram of Oriented Gradients (HOG), Color Histograms, and spatial bining features. The code for these steps is provided in 'featureExtract.py'

##### 2.1 Histogram of Oriented Gradients (HOG)

The 'skimage.hog()' function was used for this puprose as seen in lines 7-26 of 'featureExtract.py'. Different parameters of `orientations`, `pixels_per_cell`, and `cells_per_block` were tested to arrive to a model, however they did not provide much of a differance toward the end results. Below is example of applying HOG on a vehcile and non-vehicle images in 'YUV' color space with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

[alt_text][image2]

##### 2.2 Color histograms

In addition HOG, color histogram features were applied and included as an added input to the trained model. The application can be seen in lines 36-48 of 'featureExtract.py' where histogram features were obtained using 'numpy.histogram' for each channel individualy and the concatnated together. The main parameter in this case is the number of bins, which was selected as 16 which is reasonable given the range of 0-255.

##### 2.3 Spatial Bining

Spatial bining was also used as a feature for training the model where 'cv2.resize' and 'ravel' where used to obtain a 1D list of the values of a downscaled image. This step can be seen in lines 29-33 of 'featureExtract.py' and the tunable parameter in this case is the 'spatial_size' which was selected to be (16, 16).

The main parameters that remain to choose are the color space and color channels to use for the feature extraction process. 'YCrCb' and 'YUV' provided the highest accuracies upon training. However, upon testing other sets of images, a couple of shortcomings were observed. The models trained using 'YCrCb' and 'YUV' were very susciptible to shadows which caused several false positive detections. Additionally, the detection results were affected by the colors of the vehicles in an image. In order to compensate for these shortcomings, the saturation channel of the 'HLS' color space was added to the feature extraction and training processes and the final color spaces and channels used were: * All channels of 'YUV' color space * Saturation channel of 'HLS' color space

The final selected parameters can be seen in lines 22-31 of 'trainingModel.py'.

#### 3. Training the model

Once all features are extracted, the next step is training a classifier. These steps can be observed 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

