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
        self.current_fit = None
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

    def try_fitting(self, allx, ally):
        # 2. calculate new best bit coefficient
        current_fit = self.__poly_fit(allx, ally)
        # 3. calculate new curvature
        radius_of_curvature = self.__cal_curvature(allx, ally)
        # 4. calculate new value of distance deviated from center
        dist_from_center = self.__cal_dist_from_center(current_fit)
        return current_fit, radius_of_curvature, dist_from_center

    def update(self, allx, ally, current_fit, radius_of_curvature, dist_from_center):
        # update to self member
        self.allx = allx
        self.ally = ally
        self.current_fit = current_fit.copy()
        self.radius_of_curvature = radius_of_curvature
        self.line_base_pos = dist_from_center

    def __poly_fit(self, allx, ally):
        return np.polyfit(ally, allx, 2)

    def __cal_curvature(self, allx, ally):
        # choose maximum
        y_eval = np.max(ally) * self.ym_per_pix
        # equtaion to calculate curvature
        fit = np.polyfit(ally * self.ym_per_pix, allx * self.xm_per_pix, 2)
        return ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * fit[0])

    def __cal_dist_from_center(self, current_fit):
        #mean_pixel = np.mean(self.allx)
        mean_pixel = current_fit[0]*(self.__shape[0]/2)**2 + current_fit[1]*self.__shape[0]/2 + current_fit[2]
        return (mean_pixel - self.__shape[1] // 2) * self.xm_per_pix