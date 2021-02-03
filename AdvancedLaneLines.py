from CImgProcessor import CImgProcessor
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

tif = 'test_images/'
ImgPro = CImgProcessor()



for fname in os.listdir(tif):
    # read undistorted image
    undist = ImgPro.undistorted(tif + fname)

    # process
    bin = ImgPro.saturation_thres(undist, 170, 255)
    bin |= ImgPro.gradient_thres(undist)
    bin *= 255

    cv2.imwrite('image_process/' + fname, bin)


#plt.imshow(bin, cmap='gray')
#plt.show()