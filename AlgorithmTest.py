import numpy as np
from numpy.core.fromnumeric import size
from CImgProcessor import CImgProcessor
from CLocateLines import CLocateLines
from CLine import Line
import cv2
import os
from moviepy.editor import VideoFileClip

tif = 'test_images/'
ImgPro = CImgProcessor()
LL = CLocateLines()
fList = os.listdir(tif)     # list all files in test_images

def extract_frames(movie, times):
    ''' extract frames from video '''
    clip = VideoFileClip(movie)
    img_ls = []
    for t in times:
        img = clip.get_frame(t)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_ls.append(img)

    return img_ls

def process_binary(img, bUnwrap = True):
    pro_img = ImgPro.undistorted(img)
    if bUnwrap == True:
        pro_img = ImgPro.unwrap(pro_img, 450, 530, 650, 150, 0)

    gray = cv2.cvtColor(pro_img, cv2.COLOR_BGR2GRAY)
    # take saturation and gradient threshold
    bin_dir = ImgPro.dir_thres(gray, 15, (0.7, 1.2))
    bin_mag = ImgPro.mag_thres(gray)
    bin = np.zeros_like(bin_dir)
    bin_sat = ImgPro.saturation_thres(pro_img, 130, 255)
    bin[((bin_dir > 0) & (bin_mag > 0)) | (bin_sat > 0)] = 255

    return bin

def fit_lanelines(undist, bin):
    LL.set_binary_img(bin)
    leftx, rightx = LL.sliding_window()
    visImg = LL.visualize(leftx, rightx)

    wrap = ImgPro.wrap(visImg)
    combined = cv2.addWeighted(undist, 1, wrap, 0.3, 0)
    return combined

def show_image(img, time = 1000):
    cv2.imshow('img', img)
    cv2.waitKey(time)

movie = 'project_video.mp4'
times = 25, 26
# img_ls = extract_frames(movie, times)
imgname = 'mytest4.jpg'
filename = tif + imgname
img = cv2.imread(filename)
bin = process_binary(img, False)
cv2.imwrite('image_process/bin_process/' + imgname, bin)

