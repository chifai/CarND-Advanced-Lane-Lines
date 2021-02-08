# This is to poly fit the lanelines by sliding window method
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class CLocateLines:
    def __init__(self, nWindows = 9, nMargin = 100, nMinPixel = 50):
        self.__nWindows = nWindows      # Choose the number of sliding windows
        self.__nMargin = nMargin        # Set the width of the windows +/- margin
        self.__nMinPixel = nMinPixel    # Set minimum number of pixels found to recenter window
        self.__bImg = None

    def set_binary_img(self, binary_img):
        self.__bImg = binary_img

    def sliding_window(self):
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

        return self.fit_poly(leftx, lefty, rightx, righty)

    def fit_poly(self, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return [left_fit, right_fit]

    def fit_polynomial(self, left_fit, right_fit):
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
            return []

        # Fit new polynomials
        return self.fit_poly(leftx, lefty, rightx, righty)

    def visualize(self, leftfit_coeff, rightfit_coeff):
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((self.__bImg, self.__bImg, self.__bImg))*255
        window_img = np.zeros_like(out_img)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.__bImg.shape[0] - 1, self.__bImg.shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = leftfit_coeff[0]*ploty**2 + leftfit_coeff[1]*ploty + leftfit_coeff[2]
        right_fitx = rightfit_coeff[0]*ploty**2 + rightfit_coeff[1]*ploty + rightfit_coeff[2]

        # Generate a polygon to illustrate the lane line area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_line_poly = np.hstack((left_line_pts, right_line_pts))

        # fillPoly: shape[1, 1440, 2] -> xy coordinates with 720 * 2 size

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_(lane_line_poly), (0,255, 0))
        
        return window_img
