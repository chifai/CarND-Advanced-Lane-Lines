import numpy as np
from numpy.core.fromnumeric import size
from CImgProcessor import CImgProcessor
from CLocateLines import CLocateLines
from CLine import Line
import cv2
import os
from moviepy.editor import VideoFileClip

ImgPro = CImgProcessor()
LL = CLocateLines()

def process_image(img):
    # undistorted
    undist = ImgPro.undistorted(img)

    # take saturation and gradient threshol

    # take saturation and gradient threshold
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    bin_dir = ImgPro.dir_thres(gray, 15, (0.7, 1.2))
    bin_mag = ImgPro.mag_thres(gray, 3, (40, 255))
    bin_sat = ImgPro.saturation_thres(undist, 130, 255)

    bin = np.zeros(bin_dir.shape, dtype=np.uint8)
    bin[((bin_dir > 0) & (bin_mag > 0)) | (bin_sat > 0)] = 255

    bin = ImgPro.unwarp(bin, 450, 530, 720, 80, 0)

    LL.update(bin)
    visImg = LL.visualize()
    radius = LL.get_mean_curvature()
    deviation = LL.get_deviation()
    wrap = ImgPro.warp(visImg)
    combined = cv2.addWeighted(undist, 1, wrap, 0.3, 0)

    # text 
    text_radius = 'Radius: ' + str(radius) + 'm'
    text_deviation = 'Deviated from center: ' + str(round(deviation, 2)) + 'm (+ve means to the right)'

    combined = ImgPro.add_text(combined, text_radius, (20, 60))
    combined = ImgPro.add_text(combined, text_deviation, (20, 100))

    return combined

video_name = 'project_video.mp4'
white_output = 'output_images/' + video_name

clip1 = VideoFileClip(video_name)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)