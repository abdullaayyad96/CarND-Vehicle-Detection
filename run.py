import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sklearn
import random
import pickle
import os
from featureExtract import *
from findLane import *
from sklearn.svm import SVC
from moviepy.editor import VideoFileClip
from cars import cars
from laneLines import Line


#create instance of car detection object
frame_cars = cars(image_dim=[720, 1280], box_threshold=1.5)

#Create instance of line class
myLine = Line()
myLine.set_param([720, 1280], 3/110, 3.7/640)

#loading trained model and fitting scaler
X_scaler = pickle.load(open('saved_files/X_scalar_YUV_HLS.sav', 'rb'))
clf = pickle.load(open('saved_files/trained_SVC_model_YUV_HLS.sav', 'rb'))

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


#Loading camera calibration files
cal_mtx_dir = "saved_files/cal_mtx.sav"
cal_dist_dir = "saved_files/cal_dist_dir"
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

# A function that applies sliding window search on different scales, extracts features and makes predictions
def find_cars(img, color_spaces, y_start, y_stop, scales, svc, scaler, orient, pix_per_cell, cell_per_block, channels, spatial_size, hist_bins, hist_feat, spatial_feat):

    boxes = [] #a list to store the windows at which a vehicle was detected

    #Calculating the total numbe rof channels to be processes
    n_channels = 0
    for channel in channels:
        if channel=="ALL":
            n_channels+=3
        else:
            n_channels+=1

    #obtain region of interest 
    img_tosearch = img[min(y_start):max(y_stop),:,:]

    # apply color conversion if other than 'RGB' for all the required color spaces
    ctrans_tosearch_all = np.zeros([img_tosearch.shape[0], img_tosearch.shape[1], n_channels], dtype=np.uint8)
    channel_jump = 0
    start_channel = 0
    for i in range(len(color_spaces)):
        if channels[i]=="ALL":
            channel_jump+=3
            current_channel = np.linspace(0, 2, num=3, dtype=np.uint8)
        else:
            channel_jump+=1
            current_channel = [channels[i]]
        color_space=color_spaces[i]
        if color_space != 'RGB':
            if color_space == 'HSV':
                ctrans_tosearch_all[:,:, start_channel:channel_jump] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)[:,:,current_channel]
            elif color_space == 'LUV':
                ctrans_tosearch_all[:,:, start_channel:channel_jump] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)[:,:,current_channel]
            elif color_space == 'HLS':
                ctrans_tosearch_all[:,:, start_channel:channel_jump] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)[:,:,current_channel]
            elif color_space == 'YUV':
                ctrans_tosearch_all[:,:, start_channel:channel_jump] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)[:,:,current_channel]
            elif color_space == 'YCrCb':
                ctrans_tosearch_all[:,:, start_channel:channel_jump] = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)[:,:,current_channel]
        else: ctrans_tosearch_all[:,:, start_channel:channel_jump] = np.copy(img_tosearch)[:,:,current_channel]
        start_channel = channel_jump
     
    #Apply sliding window search on different scales
    for i in range(len(scales)):
        scale = scales[i]
        ystart = y_start[i]
        ystop = y_stop[i]
        ctrans_tosearch = ctrans_tosearch_all[(ystart-min(y_start)):(ystop-min(y_start)),:,:]
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
        cells_per_step = 2  # how many cells to step for the sliding window
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
        # Compute individual channel HOG features for the entire image
        hog_ch = []
        for i in range(n_channels):
            hog_ch.append(get_hog_features(ch[:,:,i], orient, pix_per_cell, cell_per_block, feature_vec=False))
    
        #The sliding window search
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch with the desired color channels
                test_features = []
                start_channel = 0
                for i in range(len(color_spaces)):
                    if channels[i]=="ALL":
                        current_channel = np.linspace(0, 2, num=3, dtype=np.uint8)
                    else:
                        current_channel = [channels[i]]
                    end_channel = start_channel+len(current_channel)
                    
                    hog_features = []
                    for hog_chi in hog_ch[start_channel:end_channel]:
                        hog_features.extend(hog_chi[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch for the desired channel
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

                    #stack the HOG and color features of the current channels
                    test_features.extend(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1).ravel())  
                    start_channel += len(current_channel)


                # Scale features and make a prediction  
                test_features = scaler.transform(np.asarray(test_features).reshape(-1,len(test_features))) 
                test_prediction = svc.predict(test_features)
            
                #If a vehicle detected append the window the the boxes list
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
        annotate_img = cv2.putText(annotate_img,"Displacement from lane center: {0:.2f} m".format(line_obj.center_displacement), (100,150), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)

    else:
        #if no line were detected return an image marked with the desired text
        annotate_img = cv2.putText(input_img,"No Line Detected", (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    #Searching cars
    
    #define scales and range to which each scale is applied
    scales = [2, 1.5, 1]
    y_start = [550, 420, 400]
    y_stop = [700, 620, 510]

    #Apply sliding window search and obtain windows with detected vehicles
    on_boxes = find_cars(frame, color_spaces, y_start, y_stop, scales, clf, X_scaler, orient, pix_per_cell, cell_per_block, channels, spatial_size, hist_bins, hist_feat, spatial_feat)

    #draw the detected boxes
    boxed_image = draw_boxes(frame, on_boxes)

    #pass to cars object for processing
    car_obj.add_frame(on_boxes)

    #draw processed boxes
    output_image = draw_labeled_bboxes(annotate_img, car_obj.processed_boxes)

    return output_image


def main(argvs):

	input_type = ''	
	input_dir = ''
	output_dir = ''

	if (len(argvs) == 4):
		#set video or image mode
		input_type = argvs[1]	
		#input & output directories directory
		input_dir = argvs[2]
		output_dir = argvs[3]
		
		if (input_dir[-1] != "/") and (input_dir[-3:] != "mp4"):
			input_dir += "/"
		
		if output_dir[-1] != "/" and (output_dir[-3:] != "mp4"):
			output_dir += "/"
	else:
		print("3 arguments are required, only %s were provided" % (len(argvs)-1))
		sys.exit()	
		
		
	if (input_type == 'image'):
		#image mode

		#read images in directory
		images = os.listdir(input_dir)
		for image in images:
			if (image[-3:]!="jpg"):
				images.remove(image)
		n_images = len(images)
		
		for i in range(n_images):
			#iterate through ia=mages

			image = mpimg.imread(input_dir+images[i])
			
			#clear Line and car objects for each image
			myLine = Line()
			myLine.set_param([720, 1280], 3/110, 3.7/640)
			frame_cars = cars(image_dim=image.shape[:2], box_threshold=1, input_mode=input_type) #apply a higher threshold for images

			#perform pipeline
			output_image = process_image(image, car_obj=frame_cars, line_obj=myLine)
			  
			#save image
			mpimg.imsave(output_dir+images[i], output_image)    	
			

	elif (input_type == 'video'):
		#video mode

		#Read video
		input_clip = VideoFileClip(input_dir)
		
		#process video frames
		output_clip = input_clip.fl_image(process_image)

		#save output video
		output_clip.write_videofile(output_dir, audio=False)


main(sys.argv)
sys.exit()
