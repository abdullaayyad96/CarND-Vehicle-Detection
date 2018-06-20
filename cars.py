import numpy as np
from scipy.ndimage.measurements import label

class cars():
    """This class keeps track of the vehicles in a series of frames"""
    def __init__(self, image_dim, box_threshold=2, input_mode='video'):

        #List of tuples containing box credentials of vehicles detected in last frame
        self.boxes = None
        #list of processed boxes containing cars upon averaging and thresholding
        self.processed_boxes = None
        #Number of processed frames
        self.nframes = 0
        #heatmap to keep track of boxes
        self.heatmap = np.zeros((image_dim[0], image_dim[1])).astype(np.float)
        #threshold for boxes to detect as objects
        self.threshold = box_threshold
        #video or image mode
        self.mode = input_mode

    def add_heat(self, heat, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heat

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        thresh_heatmap = np.copy(heatmap)
        thresh_heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return thresh_heatmap

    def add_frame(self, new_boxes):
        #add and process new frame
        new_heatmap = self.add_heat(np.zeros_like(self.heatmap), new_boxes)
        if (self.mode == 'image'):
            #if image mode, directly update heatmap
            self.heatmap = new_heatmap
        else:
            #if image mode, avergae newly obtained heatmap with previous heatmap
            #averaging_val = 0.975 *(1-np.exp(-self.nframes/2))
            averaging_val = 0.975 #
            self.heatmap = averaging_val*self.heatmap + (1-averaging_val)*new_heatmap
            
        self.nframes += 1 #update number of processed frames

        #apply threshold on heatmap
        threshold_heatmap = self.apply_threshold(self.heatmap, self.threshold)    

        #obtain boxes based on thresholded heatmap
        self.processed_boxes = label(threshold_heatmap)

        


