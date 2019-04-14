#The code in this file is responsible for training and saving a linear SVC for vehicle detection

import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sklearn
import random
import pickle
from featureExtract import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from skimage.feature import hog

#Training data directory
vehicle_dir = 'dataSet/vehicles.zip'
nonvehicle_dir = 'dataSet/non-vehicles.zip'

#properties of feature extraction
color_spaces = ['YUV', 'YUV', 'HLS'] # List of color spaces to use, can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
channels = [0, 1, 2] # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

#empty lists for features and labels
features = []
labels = []
vehicle_features = []
non_vehicle_features = []

#reading features for vehicle images
#accessing file in zip folder
archive = zipfile.ZipFile(vehicle_dir, 'r')
archive_files = archive.namelist()


#iterating through files in the zip directory
for file in archive_files:

    #If the file correspond to an image in the vehicles folder
    if ((file[-3:]=="png") & (file[:8]=="vehicles")):
        

        if file.find("GTI")==1:
            #GTI contains time series data in which multiple images can be almost identical, to avoid problems associated with 
            #identical images, random sample will be droped
            keep_prob = random.random() #obtain a floating point number between 0 and 1, which would later be compared to a threshold to determone whether to read image
        else:
            #if not time series data, all sample should be kept
            keep_prob = 1

        if(keep_prob>0.9):

            #read and scale image
            img_dir = archive.open(file)
            image = mpimg.imread(img_dir)
            image = (image * 255).astype(np.uint8)
        
            image_features = []  #list to store the features of each image
            for i in range(len(color_spaces)):
                #extracting the features corresponding to each of the desried color spaces
                color_space = color_spaces[i]
                channel = channels[i]
                #Extract image features
                image_features.extend(extract_features(image, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                channel=channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat))
            
            #append image features for the vehicle case
            vehicle_features.append(np.asarray(image_features))
            
#reading features for non-vehicle images
#accessing file in zip folder
archive = zipfile.ZipFile(nonvehicle_dir, 'r')
archive_files = archive.namelist()

for file in archive_files:

    #If the file correspond to an image in the non-vehicles folder
    if ((file[-3:]=="png") & (file[:12]=="non-vehicles")):

        #read and scale image
        img_dir = archive.open(file)
        image = mpimg.imread(img_dir)
        image = (image * 255).astype(np.uint8)

        image_features = []#list to store the features of each image
        for i in range(len(color_spaces)):
            #extracting the features corresponding to each of the desried color spaces
            color_space = color_spaces[i]
            channel = channels[i]
            #Extract image features
            image_features.extend(extract_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            channel=channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat))
        
        #append image features for the non vehicle case
        non_vehicle_features.append(np.asarray(image_features))

# Create an array stack of feature vectors
features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

# Define the labels vector
labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
train_features, test_featueres, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = sklearn.preprocessing.StandardScaler().fit(train_features)
# Apply the scaler to X
train_features = X_scaler.transform(train_features)
test_featueres = X_scaler.transform(test_featueres)

#Define and train linear state vector calssifier
clf = SVC()
clf.fit(train_features, train_labels)

#evaluating accuracy of classifier
accuracy = clf.score(test_featueres, test_labels)
print("The accuracy of the classifier is: ", accuracy )

#save model
pickle.dump(clf, open('saved_files/trained_SVC_model_YUV_HLS.sav', 'wb'))
pickle.dump(X_scaler, open('saved_files/X_scalar_YUV_HLS.sav', 'wb'))


