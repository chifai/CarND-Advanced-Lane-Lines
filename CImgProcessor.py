import json
import numpy as np
import cv2
class CImgProcessor():
    
    def __init__(self):
        self.__nx = 9
        self.__ny = 6
        self.__cam_mtx = None
        self.__dist_coeff = None
        with open('camera_cali_result.json', 'r') as read_file:
            data = json.load(read_file)
            self.__cam_mtx = np.array(data['camera_matrix'])
            self.__dist_coeff = np.array(data['dist_coeff'])

    def undistorted(self, img_path):
        img = cv2.imread(img_path)
        undist = cv2.undistort(img, self.__cam_mtx, self.__dist_coeff, None, self.__cam_mtx)
        return undist

    def unwrap(self, chessboard_img_path, nx, ny):
        img = cv2.imread(chessboard_img_path)
        undist = cv2.undistort(img, self.__cam_mtx, self.__dist_coeff, None, self.__cam_mtx)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        offset = 100
        img_size = (gray.shape[1], gray.shape[0])

        src = np.float32([corners[0], corners[nx - 1], corners[nx * ny - 1], corners[nx * (ny - 1)]])
        dst = np.float32([  [offset, offset], [img_size[0]-offset, offset], 
                            [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
        return warped



p = CImgProcessor()

# undist = p.undistorted('camera_cal/calibration1.jpg')
# undist = p.undistorted('test_images/straight_lines1.jpg')
w = p.unwrap('camera_cal/calibration3.jpg', 9, 6)
cv2.imwrite('image_process/cali3_uw.jpg', w)