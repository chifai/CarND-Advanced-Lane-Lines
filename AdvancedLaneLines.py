from numpy.core.fromnumeric import size
from CImgProcessor import CImgProcessor
from CLocateLines import CLocateLines
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

tif = 'test_images/'
ImgPro = CImgProcessor()
LL = CLocateLines()

fList = os.listdir(tif)
for i in range(fList.__len__()):
    # read undistorted and unwraped image
    img = ImgPro.read_image(tif + fList[i])
    undist = ImgPro.undistorted(img)
    unwrap = ImgPro.unwrap(undist, 450, 530, 650, 150, 0)

    # take saturation and gradient threshold
    bin = ImgPro.saturation_thres(unwrap, 170, 255)
    bin |= ImgPro.gradient_thres(unwrap)
    bin *= 255

    LL.set_binary_img(bin)
    leftx, lefty, rightx, righty, out_img = LL.find_lane_pixels()
    cv2.imwrite('image_process/bin_slidingwindow/' + fList[i], out_img)

    #cv2.imshow('f1', undist)
    #cv2.waitKey(0)


#plt.imshow(bin, cmap='gray')
#plt.show()