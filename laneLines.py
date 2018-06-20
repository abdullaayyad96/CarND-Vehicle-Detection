import numpy as np

class Line():
    """This class tracks """
    def __init__(self):
        #A boolean to show whether lines were detected previously
        self.detected = False  
        #Array of recent y and x points for left and right lines
        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None
        #are the last added values valid
        self.valid_new = False
        #number of frames since last valid reading 
        self.last_valid_frame = 0
        #dimensions of the images
        self.dim = None
        #conversion parameters from pixel to actual dimesions
        self.ym_per_pix = None
        self.xm_per_pix = None
        #matrix to convert polynomial from pixel to actual dimensions
        self.cvrt_mtx = None
        #curvature of recent left and right lanes in pixel dimensions
        self.right_curv = None
        self.left_curv = None
        #right and left polynomials of the most recent fit in pixel dimensions
        self.right_poly = [np.array([False])]  
        self.left_poly = [np.array([False])]  
        #average curvature of left and right lanes in pixel dimensions
        self.avg_right_curv = None
        self.avg_left_curv = None
        #average curvature of left and right lanes in actual dimensions
        self.act_avg_right_curv = None
        self.act_avg_left_curv = None
        #radius of curvature of the line in  actual dimensions
        self.radius_of_curvature = None 
        #polynomial coefficients averaged in pixel dimensions
        self.avg_right_poly = [np.array([False])]  
        self.avg_left_poly = [np.array([False])]
        #polynomial coefficients averaged in actual dimensions
        self.act_avg_right_poly = [np.array([False])]  
        self.act_avg_left_poly = [np.array([False])]
        #averaging factor
        self.avg_factor = 0.8
        #average distance between the two left and right lines
        self.lines_distance = 3.7
        #distance between the left and right lines in recent frame
        self.recent_distance = 0
        #distance between the left and right lines in the base of the recent frame
        self.base_distance = 0
        #distance from the center of two lines in meter
        self.center_displacement = 0
        
    
    def set_param(self, image_shape, ym_per_pix, xm_per_pix):
        #Setting parameters for use by functions in the object

        self.dim = image_shape
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        self.cvrt_mtx = np.diag([ (self.xm_per_pix / self.ym_per_pix**2),  (self.xm_per_pix / self.ym_per_pix), self.xm_per_pix  ])

    def find_curvature(self, mode='recent'):
        #accepts polynomial fit and pixel to actual dimension conversion parameters and returns curveture on the polynomial in actual dimensions 

        #defining evaluation point for y
        y_eval = self.dim[0]

        if (mode=='recent'):
            # Calculate the new radii of curvature of left and right lines in pixel dimensions
            self.right_curv = ((1 + (2*self.right_poly[0]*y_eval + self.right_poly[1])**2)**1.5) / np.absolute(2*self.right_poly[0])
            self.left_curv = ((1 + (2*self.left_poly[0]*y_eval + self.left_poly[1])**2)**1.5) / np.absolute(2*self.left_poly[0])
        elif (mode=='avg'):
            # Calculate the new radii of curvature of left and right lines in pixel dimesions
            self.avg_right_curv = ((1 + (2*self.avg_right_poly[0]*y_eval + self.avg_right_poly[1])**2)**1.5) / np.absolute(2*self.avg_right_poly[0])
            self.avg_left_curv = ((1 + (2*self.avg_left_poly[0]*y_eval + self.avg_left_poly[1])**2)**1.5) / np.absolute(2*self.avg_left_poly[0])
            # Calculate the new radii of curvature of left and right lines in actual dimesions
            self.act_avg_right_curv = ((1 + (2*self.act_avg_right_poly[0]*y_eval*self.ym_per_pix + self.act_avg_right_poly[1])**2)**1.5) / np.absolute(2*self.act_avg_right_poly[0])
            self.act_avg_left_curv = ((1 + (2*self.act_avg_left_poly[0]*y_eval*self.ym_per_pix + self.act_avg_left_poly[1])**2)**1.5) / np.absolute(2*self.act_avg_left_poly[0])
            self.radius_of_curvature = (self.avg_right_curv + self.avg_left_curv) / 2

            #for straight lines radius of curvs would be very high
            if(self.radius_of_curvature > 4000):
                self.radius_of_curvature = np.inf

    def calc_displacement(self):
        midpoint = self.dim[1]/2
        y_eval = self.dim[0]

        leftx_base = self.avg_left_poly[0]*y_eval**2 + self.avg_left_poly[1]*y_eval + self.avg_left_poly[2]
        rightx_base = self.avg_right_poly[0]*y_eval**2 + self.avg_right_poly[1]*y_eval + self.avg_right_poly[2]
        lane_center = (leftx_base + rightx_base) / 2

        self.center_displacement = self.xm_per_pix * (lane_center - midpoint)

    def calc_avg_distance(self, mode='recent'):
        y_eval = self.dim[0]
        #calculating the average area between the two polynomials by integrating the differance between the two polynomials from 0 to y_eval
        #calculating in pixel dimensions
        if(mode=='recent'):
            #perform calculations on recently added polynomials
            distance_pxl = ((1/3)*self.right_poly[0]*y_eval**3 + (1/2)*self.right_poly[1]*y_eval**2 + self.right_poly[2]*y_eval - (1/3)*self.left_poly[0]*y_eval**3 - (1/2)*self.left_poly[1]*y_eval**2 - self.left_poly[2]*y_eval ) / y_eval
            self.recent_distance = distance_pxl * self.xm_per_pix
        elif(mode=='avg'):
            #perform calculations on averaged polynomials 
            avg_distance_pxl = ((1/3)*self.avg_right_poly[0]*y_eval**3 + (1/2)*self.avg_right_poly[1]*y_eval**2 + self.avg_right_poly[2]*y_eval - (1/3)*self.avg_left_poly[0]*y_eval**3 - (1/2)*self.avg_left_poly[1]*y_eval**2 - self.avg_left_poly[2]*y_eval) / y_eval
            self.lines_distance = avg_distance_pxl * self.xm_per_pix
    
    def sanity_check(self):
          
        #calculating average distance error between recently detected left and right lines compared to nominal value of 3.7m
        self.calc_avg_distance(mode='recent') #calculate the average distance between the two recent polynomials
        distance_error = abs(self.recent_distance - 3.7) / 3.7 #calculate the error in the distance between the two polynomials

        #calculating distance error between recently detected left and right lines at base compared to nominal value of 3.7m
        y_eval = self.dim[0]
        leftx_base = self.left_poly[0]*y_eval**2 + self.left_poly[1]*y_eval + self.left_poly[2]
        rightx_base = self.right_poly[0]*y_eval**2 + self.right_poly[1]*y_eval + self.right_poly[2]
        base_diff_pix = rightx_base - leftx_base
        self.base_distance = self.xm_per_pix * base_diff_pix
        base_error = abs(self.base_distance - 3.7) / 3.7

        #comparing average distance between left and right lines of recent and averaged polynomial fittings
        if(self.detected):
            #if previous lines were detected
            avg_distance_error = abs(self.recent_distance - self.lines_distance) / self.lines_distance #calculate the average distance between the averaged polynomials
        else:
            avg_distance_error = 0

        #checking the validaty of newly fitted polynomials based on calculated errors
        if((distance_error<0.125) & (avg_distance_error<0.05) & (base_error<0.125)):
            self.valid_new = True
            self.last_valid_frame=0
        else: 
            self.valid_new = False
            self.last_valid_frame += 1

            #if 50 frames since last valid lane detection the search will reset
            if(self.last_valid_frame>=50):
                self.detected = False
                self.last_valid_frame = 0
    
    def cvrt_2_act(self):
        #convert polynomials from pixel to actual dimensions in meter
        self.act_avg_right_poly = np.matmul(self.cvrt_mtx, self.avg_right_poly)
        self.act_avg_left_poly = np.matmul(self.cvrt_mtx, self.avg_left_poly)

    def find_avg(self):
        #Averaging new polynomials with previous values

        if (self.detected):
            #if previous values exist
            self.avg_right_poly = np.add(np.multiply(self.avg_factor, self.avg_right_poly), np.multiply((1-self.avg_factor), self.right_poly ) )
            self.avg_left_poly = np.add(np.multiply(self.avg_factor, self.avg_left_poly), np.multiply((1-self.avg_factor), self.left_poly ) )
        else:
            #if no previous values exit
            self.avg_right_poly = self.right_poly
            self.avg_left_poly = self.left_poly
            self.detected = True

        self.calc_avg_distance(mode='avg') #update the average distance between the left and right lines
        self.cvrt_2_act() #obtain a version of the average polynomials in actual dimensions in meters

    def fit_poly(self):
        # Fit a second order polynomial to each line
        self.left_poly = np.polyfit(self.lefty, self.leftx, 2)
        self.right_poly = np.polyfit(self.righty, self.rightx, 2)

        self.sanity_check() #perform sanity checks to check the validity of the newly fitted polynomial
        if(self.valid_new):
            #if the new polynomials are valid

            self.find_avg() #averaging the new polynomials with previous ones
            self.find_curvature(mode='avg') #calculate radius of curvuture
            self.calc_displacement()

    def add_points(self, lefty, leftx, righty, rightx):
        #Add points of right and left lines for polynomial evaluation and fitting

        if((len(lefty)>0) & (len(leftx)>0)): 
            self.leftx = leftx
            self.lefty = lefty
        if((len(righty)>0) & (len(rightx)>0)): 
            self.rightx = rightx
            self.righty = righty

        self.fit_poly() #fit the newly added points to a polynomial
        
    
            
