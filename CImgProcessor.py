import json
import numpy as np
import cv2
import matplotlib.image as mpimg
class CImgProcessor():
    def __init__(self):
        self.__nx = 9
        self.__ny = 6
        self.__cam_mtx = None
        self.__dist_coeff = None
        self.__unwraped_mat = None
        self.__wraped_mat = None
        self.__img_size = None 
        with open('camera_cali_result.json', 'r') as read_file:
            data = json.load(read_file)
            self.__cam_mtx = np.array(data['camera_matrix'])
            self.__dist_coeff = np.array(data['dist_coeff'])

    def get_wraped_mat(self):
        return self.__wraped_mat

    def read_image(self, img_path):
        return cv2.imread(img_path)

    def undistorted(self, cv2_img):
        undist = cv2.undistort(cv2_img, self.__cam_mtx, self.__dist_coeff, None, self.__cam_mtx)
        return undist

    def unwarp(self, cv2_img, x1 = 450, y1 = 520, x2 = 720, y2 = 100, offset = 0):        
        # cv2_img shape: 720, 1280, 3
        src_corners = [[x1, y1], [x1, cv2_img.shape[1] - y1], [x2, cv2_img.shape[1] - y2], [x2, y2]]
        for el in src_corners: el.reverse()
        self.__img_size = (cv2_img.shape[1], cv2_img.shape[0])
        dst_corners = [  [offset, offset], [self.__img_size[0] - offset, offset], 
                            [self.__img_size[0]-offset, self.__img_size[1] - offset], [offset, self.__img_size[1] - offset]]

        src = np.float32(src_corners)
        dst = np.float32(dst_corners)
        self.__unwraped_mat = cv2.getPerspectiveTransform(src, dst)
        self.__wraped_mat = np.linalg.inv(self.__unwraped_mat)
        unwarped = cv2.warpPerspective(cv2_img, self.__unwraped_mat, self.__img_size)
        return unwarped

    def warp(self, cv2_img):
        warped = cv2.warpPerspective(cv2_img, self.__wraped_mat, self.__img_size)
        return warped

    def gradient_thres(self, cv2_img, thres_min = 20, thres_max = 100):
        # img should be undistorted
        # output binary by gradient
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thres_min) & (scaled_sobel <= thres_max)] = 1

        return sxbinary

    def dir_thres(self, gray_img, sobel_kernel = 15, dir_thres = (0.7, 1.3)):
        # 1) Take the absolute value of gradient in x and y separately
        absSobelX = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        absSobelY = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

        # 2) calculate the direction
        direction = np.arctan2(absSobelY, absSobelX)
        binary = np.zeros(direction.shape, dtype=np.uint8)
        # 3) Create a binary mask where direction thresholds are met
        #binary = np.uint8(np.zeros_like(direction))
        binary[(direction >= dir_thres[0]) & (direction <= dir_thres[1])] = 1

        return binary

    def mag_thres(self, gray_img, sobel_kernel = 3, mag_thres = (40, 255)):
        # 1) Take the absolute value of gradient in x and y separately
        sobelX = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelY = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        mag = np.sqrt(sobelX ** 2 + sobelY ** 2)
        mag = np.uint8(255 * mag / np.max(mag))

        binary = np.zeros(mag.shape, dtype=np.uint8)
        binary[(mag >= mag_thres[0]) & (mag <= mag_thres[1])] = 1

        return binary

    def saturation_thres(self, cv2_img, sat_min = 170, sat_max = 255):
        # img should be undistorted
        hls = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HLS)
        sat_ch = hls[:,:,2]     # saturation channel
        binary = np.zeros(sat_ch.shape, dtype=np.uint8)
        binary[(sat_ch >= sat_min) & (sat_ch <= sat_max)] = 1
        return binary

    def add_text(self, cv2_img, text, position,
    font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 255, 255), thickness = 2):
        return cv2.putText(cv2_img, text, position, font, fontScale,
                    color, thickness, cv2.LINE_AA, False)