# Define a class to receive the characteristics of each line detection
import numpy as np
from CImgProcessor import CImgProcessor

class Line():
    def __init__(self, xm_per_pix, ym_per_pix, shape = [720, 1280]):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # meter per pixel
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        self.__shape = shape

    def update(self, allx, ally):
        # when new binary image comes
        # 1. update the member
        self.allx = allx
        self.ally = ally
        # 2a. if not detected: calculate new best bit coefficient
        self.__poly_fit()
        # 3. calculate new curvature
        self.__cal_curvature()
        # 4. calculate new value of distance deviated from center
        self.__cal_dist_from_center()

    def __poly_fit(self):
        # self.current_fit = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

    def __cal_curvature(self):
        # choose maximum
        y_eval = np.max(self.ally) * self.ym_per_pix
        # equtaion to calculate curvature
        fit = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * fit[0])

    def __cal_dist_from_center(self):
        #mean_pixel = np.mean(self.allx)
        mean_pixel = self.current_fit[0]*(self.__shape[0]/2)**2 + self.current_fit[1]*self.__shape[0]/2 + self.current_fit[2]
        self.line_base_pos = (mean_pixel - self.__shape[1] // 2) * self.xm_per_pix