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

LeftLane = Line()
RightLane = Line()

def process_image(img):
    # undistorted
    undist = ImgPro.undistorted(img)
    unwrap = ImgPro.unwrap(undist, 450, 530, 720, 80, 0)

    # take saturation and gradient threshol

    # take saturation and gradient threshold
    gray = cv2.cvtColor(unwrap, cv2.COLOR_BGR2GRAY)
    bin_dir = ImgPro.dir_thres(gray, 15, (0.7, 1.2))
    bin_mag = ImgPro.mag_thres(gray)
    bin_sat = ImgPro.saturation_thres(unwrap, 130, 255)
    bin = np.zeros(bin_dir.shape, dtype=np.uint8)
    bin[((bin_dir > 0) & (bin_mag > 0)) | (bin_sat > 0)] = 255

    LL.set_binary_img(bin)

    if LeftLane.detected is False or RightLane.detected is False:
        [LeftLane.best_fit, RightLane.best_fit] = LL.sliding_window()
    else:
        best_fit = LL.fit_polynomial(LeftLane.best_fit, RightLane.best_fit)
        if best_fit.__len__() == 0:
            [LeftLane.best_fit, RightLane.best_fit] = LL.sliding_window()
        else:
            LeftLane.best_fit = best_fit[0]
            RightLane.best_fit = best_fit[1]

    LeftLane.detected = True
    RightLane.detected = True

    visImg = LL.visualize(LeftLane.best_fit, RightLane.best_fit)
    wrap = ImgPro.wrap(visImg)

    combined = cv2.addWeighted(undist, 1, wrap, 0.3, 0)

    return combined


video_name = 'project_video.mp4'
white_output = 'output_images/' + video_name

clip1 = VideoFileClip(video_name)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)