# This is to poly fit the lanelines by sliding window method
import numpy as np
import cv2
import csv
from CLine import Line

class CLocateLines:
    def __init__(self, nWindows = 9, nMargin = 100, nMinPixel = 50, shape = [720, 1280],
    xm_per_pix = 3.7/700, ym_per_pix = 30/720, data_file = 'data.txt'):
        self.__nWindows = nWindows      # Choose the number of sliding windows
        self.__nMargin = nMargin        # Set the width of the windows +/- margin
        self.__nMinPixel = nMinPixel    # Set minimum number of pixels found to recenter window
        self.__bImg = None
        self.__left_lane = Line(xm_per_pix, ym_per_pix)
        self.__right_lane = Line(xm_per_pix, ym_per_pix)
        # file = open(data_file, mode = 'w')
        # self.__csvwriter = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.process_method = 'None'

    def update(self, binary_img):
        self.__set_binary_img(binary_img)

        if self.__left_lane.detected is False or self.__right_lane.detected is False:
            lanes_xy = self.__sliding_window()
            self.process_method = 'FirstSliding'
            self.__left_lane.detected = True
            self.__right_lane.detected = True
        else:
            lanes_xy = self.__poly_fit_prev(self.__left_lane.current_fit, self.__right_lane.current_fit)
            self.process_method = 'PrevFit'
            if lanes_xy is None or lanes_xy.__len__() == 0:
                lanes_xy = self.__sliding_window()
                self.process_method = 'ExceptionSliding'

        l_current_fit, l_radius, l_distance = self.__left_lane.try_fitting(lanes_xy[0], lanes_xy[1])
        r_current_fit, r_radius, r_distance = self.__right_lane.try_fitting(lanes_xy[2], lanes_xy[3])

        lane_width = abs(l_distance - r_distance)

        if l_radius == 0.0 or r_radius == 0.0:
            radius_ratio = 1.0
        elif l_radius < r_radius:
            radius_ratio = l_radius / r_radius
        else:
            radius_ratio = r_radius / l_radius

        if radius_ratio < 0.1 or (lane_width < 3.0 or lane_width > 4.5):
            # result is too worse, do sliding window again
            lanes_xy = self.__sliding_window()
            l_current_fit, l_radius, l_distance = self.__left_lane.try_fitting(lanes_xy[0], lanes_xy[1])
            r_current_fit, r_radius, r_distance = self.__right_lane.try_fitting(lanes_xy[2], lanes_xy[3])
            self.process_method = 'WorseResultSliding'
        else:
            # TODO use EMWA to find out a better fit
            if self.__left_lane.current_fit is not None:
                l_current_fit = self.__left_lane.current_fit * (1.0 - radius_ratio) + l_current_fit * radius_ratio
            if self.__right_lane.current_fit is not None:
                r_current_fit = self.__right_lane.current_fit * (1.0 - radius_ratio) + r_current_fit * radius_ratio

        # write debug data
        # self.__csvwriter.writerow([l_radius, r_radius, radius_ratio, lane_width, self.process_method])

        self.__left_lane.update(lanes_xy[0], lanes_xy[1], l_current_fit, l_radius, l_distance)
        self.__right_lane.update(lanes_xy[2], lanes_xy[3], r_current_fit, r_radius, r_distance)

    def visualize(self):
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((self.__bImg, self.__bImg, self.__bImg))*255
        window_img = np.zeros_like(out_img)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.__bImg.shape[0] - 1, self.__bImg.shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.__left_lane.current_fit[0]*ploty**2 + self.__left_lane.current_fit[1]*ploty + self.__left_lane.current_fit[2]
        right_fitx = self.__right_lane.current_fit[0]*ploty**2 + self.__right_lane.current_fit[1]*ploty + self.__right_lane.current_fit[2]

        # Generate a polygon to illustrate the lane line area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_line_poly = np.hstack((left_line_pts, right_line_pts))

        # fillPoly: shape[1, 1440, 2] -> xy coordinates with 720 * 2 size

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_(lane_line_poly), (0,255, 0))
        
        return window_img

    def get_mean_curvature(self):
        left_radius = self.__left_lane.radius_of_curvature
        right_radius = self.__right_lane.radius_of_curvature
        return (left_radius + right_radius) // 2

    def get_radius(self):
        return [self.__left_lane.radius_of_curvature, self.__right_lane.radius_of_curvature]

    def get_line_pos(self):
        lane_width = abs(self.__left_lane.line_base_pos - self.__right_lane.line_base_pos)
        return [self.__left_lane.line_base_pos, self.__right_lane.line_base_pos, lane_width]

    def get_deviation(self):
        # get deviated dist from center
        return -1.0 * (self.__left_lane.line_base_pos + self.__right_lane.line_base_pos) / 2.0

    def __set_binary_img(self, binary_img):
        self.__bImg = binary_img

    def __sliding_window(self):
        """
        fit polynomail with sliding windows
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.__bImg[self.__bImg.shape[0] // 2:,:], axis = 0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.__bImg.shape[0] // self.__nWindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.__bImg.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.__nWindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.__bImg.shape[0] - (window+1)*window_height
            win_y_high = self.__bImg.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - self.__nMargin  # Update this
            win_xleft_high = leftx_current + self.__nMargin  # Update this
            win_xright_low = rightx_current - self.__nMargin  # Update this
            win_xright_high = rightx_current + self.__nMargin

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.__nMinPixel:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.__nMinPixel:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return [leftx, lefty, rightx, righty]

    def __poly_fit_prev(self, left_fit, right_fit):
        """
        fit polynomail with previous coefficients
        """
        # Grab activated pixels
        nonzero = self.__bImg.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        leftXMin = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2] - self.__nMargin
        leftXMax = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2] + self.__nMargin
        rightXMin = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2] - self.__nMargin
        rightXMax = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2] + self.__nMargin
        
        left_lane_inds = ((nonzerox >= leftXMin) & (nonzerox < leftXMax)).nonzero()[0]
        right_lane_inds = ((nonzerox >= rightXMin) & (nonzerox < rightXMax)).nonzero()[0]
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
            return None

        # Fit new polynomials
        return [leftx, lefty, rightx, righty]
