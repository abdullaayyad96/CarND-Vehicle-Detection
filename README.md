# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[image1]: output_images/test4.jpg "sample output"
[image2]: output_videos/project_video_Trim.gif "output video"

## Overview

In this project, the goal is to develop a software pipeline to detect vehicles and lane lines in images and videos. A combination of machine learning and image processing techniques were combined to achieve the desired results.

![alt_text][image1]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier SVM classifier
* apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream.
* Estimate a bounding box for vehicles detected.

## Dependancies

- Python3
- NumPy
- OpenCV
- SciPy
- scikit-learn
- MoviePy
- Matplotlib
- pickle


## Usage

### Running on images
To run the pipeline on a set of images, run the following command:
```
python run.py image input_directory output_directory
```
The script will automatically iterate through all the image in the directory and apply the pipeline. Sample images are provided in 
`test_images` with corresponding outputs in `output_images`.

### Testing on a video
To run the script on an mp4 video, the following command can be used:
```
python run.py video path_to_video.mp4 output_video.mp4
```
A sample video is provided in `test_videos` with it's corresponding output in `output_videos`. Here's a link to my [video result](output_videos/project_video.mp4).

![alt_text][image2]


## Technical Overview

The lane detection pipeline was adopt from this [repository](https://github.com/abdullaayyad96/CarND-Advanced-Lane-Lines). Feel free to visit it for a thorough overview of the lane detection 

As for the vehicle detection pipeline, a support vector machine was trained on a a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/). The machine learning algorithim utilizes Histogram of Oriented Gradients 'HOG', Spatial Bining and Color Histograms as input data to make classification decisions. For more details regarding the technical implementation of the pipeline and the machine learning implemention, refer to the [technical writeup](technical_writeup.md).


