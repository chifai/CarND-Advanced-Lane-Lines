import numpy as np
from numpy.core.fromnumeric import size
from CImgProcessor import CImgProcessor
from CLocateLines import CLocateLines
from CLine import Line
import cv2
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    gray = cv2.cvtColor(pro_img, cv2.COLOR_BGR2GRAY)
    # take saturation and gradient threshold
    bin_dir = ImgPro.dir_thres(gray, 15, (0.7, 1.2))
    bin_mag = ImgPro.mag_thres(gray, 3, (20, 255))

    bin = np.zeros_like(bin_dir)
    bin_sat = ImgPro.saturation_thres(pro_img, 130, 255)
    bin[((bin_dir > 0) & (bin_mag > 0)) | (bin_sat > 0)] = 255

    if bUnwrap == True:
        bin = ImgPro.unwrap(bin, 450, 530, 720, 80, 0)

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

def save_testingbinary(img_name, input_dir = 'test_images/', output_dir = 'image_process/bin_process/'):
    # display testing binary image from tif
    bin = show_testingbinary(img_name, input_dir)
    cv2.imwrite(output_dir + img_name, bin)

def show_testingbinary(img_name, input_dir = 'test_images/'):
    img = cv2.imread(input_dir + img_name)
    bin = process_binary(img, True)
    return bin

def save_frame_from_video(movie_name, times, dir, img_name_prefix = 'mytest', img_starting_ind = 0):
    # extract particular frames
    img_ls = extract_frames(movie_name, times)
    for i in range(img_ls.__len__()):
        filename = img_name_prefix + str(i + img_starting_ind) + '.jpg'
        cv2.imwrite(dir + filename, img_ls[i])

def save_binary_process_diff(img_name, input_dir = 'test_images/', output_dir = 'image_process/bin_process/'):
    ori_img = mpimg.imread(input_dir + img_name)
    bin_img = show_testingbinary(img_name, input_dir)
    bin_img = np.dstack((bin_img, bin_img, bin_img))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(ori_img)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(bin_img)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(output_dir + img_name)

movie_name = 'challenge_video.mp4'
input_dir = 'image_process/challenge_video/ori_img/'
output_dir = 'image_process/challenge_video/output_img/'
output_dir2 = 'image_process/challenge_video/output_img_unwrap/'


# time_stamp = []
# t = 0
# interval = 0.2
# for i in range(20):
#     time_stamp.append(t)
#     t += interval
# save_frame_from_video(movie_name, time_stamp, input_dir, 'timestamp')

for i in range(20):
    img_name = 'timestamp' + str(i) + '.jpg'
    save_binary_process_diff(img_name, input_dir, output_dir)