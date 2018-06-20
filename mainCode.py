import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sklearn
import random
import pickle
import os
from featureExtract import *
from findLines import *
from sklearn.svm import LinearSVC
from moviepy.editor import VideoFileClip
from cars import cars
from laneLines import Line

#set video or image mode
mode = 'video'
#images directory
img_dir = 'test_images/'
img_out_dir = 'output_images/'
#videos directory
video_dir = 'test_videos/project_video.mp4'
video_dir_out = 'output_videos/project_video.mp4'

#create instance of car detection object
frame_cars = cars(image_dim=[720, 1280], box_threshold=2, input_mode=mode)

#Create instance of line class
myLine = Line()
myLine.set_param([720, 1280], 3/110, 3.7/640)

#loading trained model and fitting scaler
X_scaler = pickle.load(open('X_scalar_YUV_HLS.sav', 'rb'))
clf = pickle.load(open('trained_SVC_model_YUV_HLS.sav', 'rb'))

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

#properties for sliding window
x_start_stop = [None, None]
y_start_stop = [400, 700]
xy_window = (64, 64)
xy_overlap=(0.5, 0.5)

#Loading camera calibration files
cal_mtx_dir = "cal_mtx.sav"
cal_dist_dir = "cal_dist_dir"
mtx = pickle.load(open(cal_mtx_dir, 'rb'))
dist = pickle.load(open(cal_dist_dir, 'rb'))

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a  function that can extract features using hog sub-sampling and make predictions
def find_cars(img, color_spaces, ystart, ystop, scale, svc, scaler, orient, pix_per_cell, cell_per_block, channels, spatial_size, hist_bins, hist_feat, spatial_feat):
    boxes = []

    n_channels = 0
    for channel in channels:
        if channel=="ALL":
            n_channels+=3
        else:
            n_channels+=1
 
    img_tosearch = img[ystart:ystop,:,:]
    # apply color conversion if other than 'RGB'
    ctrans_tosearch = np.zeros([img_tosearch.shape[0], img_tosearch.shape[1], n_channels], dtype=np.uint8)
    n_channels = 0
    start_channel = 0
    for i in range(len(color_spaces)):
        if channels[i]=="ALL":
            n_channels+=3
            current_channel = np.linspace(0, 2, num=3, dtype=np.uint8)
        else:
            n_channels+=1
            current_channel = [channels[i]]
        color_space=color_spaces[i]
        if color_space != 'RGB':
            if color_space == 'HSV':
                ctrans_tosearch[:,:, start_channel:n_channels] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)[:,:,current_channel]
            elif color_space == 'LUV':
                ctrans_tosearch[:,:, start_channel:n_channels] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)[:,:,current_channel]
            elif color_space == 'HLS':
                ctrans_tosearch[:,:, start_channel:n_channels] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)[:,:,current_channel]
            elif color_space == 'YUV':
                ctrans_tosearch[:,:, start_channel:n_channels] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)[:,:,current_channel]
            elif color_space == 'YCrCb':
                ctrans_tosearch[:,:, start_channel:n_channels] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)[:,:,current_channel]
        else: ctrans_tosearch[:,:, start_channel:n_channels] = np.copy(img_tosearch)[:,:,current_channel]
        start_channel = n_channels
     

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch = ctrans_tosearch
  
    # Define blocks and steps as above
    nxblocks = (ch[:,:,0].shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch[:,:,0].shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog_ch = []
    for i in range(n_channels):
        hog_ch.append(get_hog_features(ch[:,:,i], orient, pix_per_cell, cell_per_block, feature_vec=False))
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            test_features=[]
            start_channel = 0
            for i in range(len(color_spaces)):
                if channels[i]=="ALL":
                    current_channel = np.linspace(0, 2, num=3, dtype=np.uint8)
                else:
                    current_channel = [channels[i]]
                end_channel = start_channel+len(current_channel)
                #hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat4 = hog4[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat5 = hog5[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat6 = hog6[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = []
                for hog_chi in hog_ch[start_channel:end_channel]:
                    hog_features.extend(hog_chi[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                #hog_features = hog_feat3

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window, start_channel:end_channel], (64,64))
            
                # Get color features
                if spatial_feat:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                else:
                    spatial_features = []
                if hist_feat:
                    hist_features = color_hist(subimg, nbins=hist_bins)
                else:
                    hist_features = []

                # Scale features and make a prediction
                test_features.extend(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1).ravel())  
                start_channel += len(current_channel)


            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))  
            test_features = scaler.transform(np.asarray(test_features).reshape(-1,len(test_features))) 
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes

def process_image(frame, car_obj=frame_cars, line_obj=myLine):
    #undistort frame
    frame = undistort(frame, mtx, dist)

    #Running line detection code
    #perform color and sobel thresholding
    thresh_image, color_img, region_masked_img = threshold(frame)  
  
    #apply perspective transform
    per_img = perspective_transform(thresh_image, source_points, destination_points)
   

    #finding and fitting lane lines
    line_finding_img = find_lines(per_img, line_obj, slide_mode)
    
    if(line_obj.detected):
        #if lines were detected

        #marking lane lines
        wraped_marked_img = plot(per_img, line_obj)

        #applying inverse perspective transform on marked image
        marked_img = perspective_transform(wraped_marked_img, destination_points, source_points)

        #adding marked image to original image
        added_img = cv2.addWeighted(frame, 1, marked_img, 0.5, 0)

        #annotating image 
        annotate_img = cv2.putText(added_img,"Curveture radius: {0:.2f} km".format(line_obj.radius_of_curvature/1000), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
        annotate_img = cv2.putText(annotate_img,"Displacement from lane center: {0:.2f} cm".format(line_obj.center_displacement), (100,150), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)

    else:
        #if no line were detected return an image marked with the desired text
        annotate_img = cv2.putText(input_img,"No Line Detected", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    #find cars 
    on_boxes=[]
    on_boxes.extend(find_cars(frame, color_spaces, 550+64, 700, 3, clf, X_scaler, orient, pix_per_cell, cell_per_block, channels, spatial_size, hist_bins, hist_feat, spatial_feat))
    on_boxes.extend(find_cars(frame, color_spaces, 400, 550, 1, clf, X_scaler, orient, pix_per_cell, cell_per_block, channels, spatial_size, hist_bins, hist_feat, spatial_feat))

    boxed_image = draw_boxes(frame, on_boxes)

    #pass to cars object
    car_obj.add_frame(on_boxes)

    #draw new boxes
    output_image = draw_labeled_bboxes(annotate_img, car_obj.processed_boxes)

    return output_image

def main():
    #image mode
    if (mode == 'image'):

        #read images in directory
        images = os.listdir(img_dir)
        for image in images:
            if (image[-3:]!="jpg"):
                images.remove(image)
        n_images = len(images)
        
        for i in range(n_images):

            image = mpimg.imread(img_dir+images[i])
            
            #clear Line and car objects for each image
            myLine = Line()
            myLine.set_param([720, 1280], 3/110, 3.7/640)
            frame_cars = cars(image_dim=image.shape[:2], box_threshold=2, input_mode=mode) #apply a higher threshold for images

            output_image=process_image(image, car_obj=frame_cars, line_obj=myLine)
          
            mpimg.imsave(img_out_dir+images[i], output_image)

        
        #mpimg.imsave("output_images/output1.jpg", output_image)
        #plt.subplot(311)
        #plt.imshow(boxed_image)
        #plt.subplot(312)
        #plt.imshow(labels[0], cmap='gray')
        #plt.subplot(313)
        #plt.imshow(output_image)
        #plt.show()
        ##iterating images in directory
        #for i in range(n_images):
            
    #video mode
    elif (mode == 'video'):
        #Read video
        input_clip = VideoFileClip(video_dir)
        #process video frames
        output_clip = input_clip.fl_image(process_image)
        #save output video
        output_clip.write_videofile(video_dir_out, audio=False)
    

main()