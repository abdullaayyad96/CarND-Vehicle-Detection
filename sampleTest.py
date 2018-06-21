import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sklearn
import random
import pickle
from featureExtract import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#loading data
vehicle_dir = 'dataSet/vehicles.zip'
nonvehicle_dir = 'dataSet/non-vehicles.zip'
X_scaler = pickle.load(open('saved_files/X_scalar_YUV_HLS.sav', 'rb'))
clf = pickle.load(open('saved_files/trained_SVC_model_YUV_HLS.sav', 'rb'))

#properties of feature extraction
color_spaces = ['YUV', 'HLS'] # List of color spaces to use, can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
channels = ['ALL', 2] # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


#sample tests
#obtain random samples for vehicle and non-vehicle images from the data set
n_images = 6 #number of images to ibtain
n_class_images = n_images/2

sample_images = []
sample_features = []
actual_labels = []

#reading images for the vehicle case
archive = zipfile.ZipFile(vehicle_dir, 'r')
archive_files = archive.namelist()
j=1

while (j<=n_class_images):
    #obtain random samples for vehicle images
    index = random.randint(0, len(archive_files)-1)
    file = archive_files[index]
    if ((file[-3:]=="png") & (file[:8]=="vehicles")):
        #read image
        j+=1
        img_dir = archive.open(file)
        image = mpimg.imread(img_dir)
        image = (image * 255).astype(np.uint8)
        sample_images.append(image)

        image_features = [] #list to store the features of each image

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
        sample_features.append(np.asarray(image_features))
        actual_labels.append('Vehicle')

#reading images for the non-vehicle case
archive = zipfile.ZipFile(nonvehicle_dir, 'r')
archive_files = archive.namelist()
j=1
while (j<=n_class_images):
    #obtain random samples for non-vehicle images
    index = random.randint(0, len(archive_files)-1)
    file = archive_files[index]
    if ((file[-3:]=="png") & (file[:12]=="non-vehicles")):
        #read image
        j+=1
        img_dir = archive.open(file)
        image = mpimg.imread(img_dir)
        image = (image * 255).astype(np.uint8)
        sample_images.append(image)

        image_features = [] #list to store the features of each image

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
        sample_features.append(np.asarray(image_features))
        actual_labels.append('Non Vehicle')


#Normalizing the obtained features for all images using the loaded fitting scale
sample_features = X_scaler.transform(sample_features)
#Predicting each outcome for the images using the loaded machince learning model
sample_labels = clf.predict(sample_features)

#Converting the model prediction to actual messages 
result_labels = []
for label in sample_labels:
    if(label==1):
        result_labels.append('Vehicle')
    else:
        result_labels.append('Non vehicle')

#printing and saving results
rows = n_images/2
columns = n_images/rows
plt.figure(figsize=(10, 15))
plt.suptitle('Comparing predictions', fontsize=16)
for i in range(n_images):
    plt.subplot(rows,columns,(i+1))
    plt.imshow(sample_images[i])
    plt.title('Actual: {} \n Prediction: {}'.format(actual_labels[i], result_labels[i]), fontsize=12)

plt.savefig("report_images/classifier_test.jpg")
plt.show()



# Visualizing sample images and hog feature extraction
#plt.figure()
#plt.suptitle('Sample images', fontsize=16)
#plt.subplot(1,2,1)
#plt.imshow(sample_images[1])
#plt.title('Vehicle Image')
#plt.subplot(1,2,2)
#plt.imshow(sample_images[4])
#plt.title('Non-Vehicle Image')

#plt.savefig("report_images/sample_images.jpg")
#plt.show()

#vehicle_YUV = cv2.cvtColor(sample_images[1], cv2.COLOR_RGB2YUV)
#nonvehicle_YUV = cv2.cvtColor(sample_images[4], cv2.COLOR_RGB2YUV)

#plt.figure(figsize=(15, 15))
#plt.suptitle('YUV HOG features', fontsize=16) 
#for i in range(1,4):
#    hogs, hog_vehicle = get_hog_features(vehicle_YUV[:,:,(i-1)], orient, pix_per_cell, cell_per_block, True)
#    hogs, hog_vnonehicle = get_hog_features(nonvehicle_YUV[:,:,(i-1)], orient, pix_per_cell, cell_per_block, True)
#    plt.subplot(3,4,4*i-3)
#    plt.imshow(vehicle_YUV[:,:,i-1], cmap='gray')
#    plt.title('Vehicle CH:{}'.format(i))
#    plt.subplot(3,4,4*i-2)
#    plt.imshow(hog_vehicle, cmap='gray')
#    plt.title('Vehicle HOG CH:{}'.format(i))
#    plt.subplot(3,4,4*i-1)
#    plt.imshow(nonvehicle_YUV[:,:,i-1], cmap='gray')
#    plt.title('Non-Vehicle CH:{}'.format(i))
#    plt.subplot(3,4,4*i)
#    plt.imshow(hog_vnonehicle, cmap='gray')
#    plt.title('Non-Vehicle HOG CH:{}'.format(i))


#plt.savefig("report_images/HOG_images.jpg")
#plt.show()



