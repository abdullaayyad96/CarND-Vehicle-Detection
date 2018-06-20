import numpy as np
import cv2
from skimage.feature import hog


#a function that returns HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

#a function to compute color histogram features 
def color_hist(img, nbins=32, bins_range=(0, 256)):

    #expand dimensions in case of a single channel, to be able to iterate through channels later
    if (len(img.shape)<3):
        img = np.expand_dims(img, axis=2)

    hist_features = [] #a list to store the channel histogram features of all channels in the input image
    for i in range(img.shape[2]):
        #obtain and appen histogram features
        channel_hist, channel_edges = np.histogram(img[:,:,i], bins=nbins, range=bins_range)
        hist_features.extend(channel_hist)
    # Return the feature vector
    return hist_features

# Define a function to extract all desired features from an image
def extract_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    image_features = [] #a list to store all the desired features of an image
        
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)   
    else:   feature_image = np.copy(image)

    if channel == 'ALL':
        #if feature extraction is required for all channels
        if spatial_feat == True:
            # Apply bin_spatial
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            image_features.extend(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            image_features.extend(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = []
            for channel in range(feature_image.shape[2]):
                #obtain hog features for each channel individually and append them
                hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True).ravel())    
            # Append the new feature vector to the features list
            image_features.extend(hog_features)
    else:
        if spatial_feat == True:
            # Apply bin_spatial
            spatial_features = bin_spatial(feature_image[:,:,channel], size=spatial_size)
            image_features.extend(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image[:,:,channel], nbins=hist_bins)
            image_features.extend(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            hog_features = np.ravel(hog_features)  
            # Append the new feature vector to the features list
            image_features.extend(hog_features)

    # Return features list
    return image_features
    
