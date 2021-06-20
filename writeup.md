# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Undistorted Chessboard"
[image2]: ./output_images/calibrated_chessboard.jpg "Calibrated Chessboard"
[image3]: ./output_images/original.jpg "Original"
[image4]: ./output_images/undistorted.jpg "Undistorted"
[image5]: ./output_images/warped_binary.jpg "Warped Binary Example"
[image6]: ./output_images/unwarped_binary.jpg "Unwarped Binary Example"
[image7]: ./output_images/final_result.png "Final Result"

[video1]: ./output_images/project_video.mp4 "Project Video"
[video2]: ./output_images/challenge_video.mp4 "Challenge Video"
[video3]: ./output_images/harder_challenge_video.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`camera_calibrate.py` is a program to compute camera matrix and distortion coefficients and then to save the result to a separated file.
1. use 20 chessboard pictures in ./camera_cal to obtain `objpoints` and `imgpoints`
2. use `cv2.calibrateCamera()` to compute camera matrix and distortion coefficients
3. save the result to `camera_cali_result.json` in json format

**Original Chessboard**
![alt text][image1]

**Undistorted and Unwarped Chessboard**
![alt text][image2]

### Pipeline (single images)

`CImgProcessor` is a class to process image. 
`__init__` function is to extract the camera matrix and distortion coefficient from json file, and then they are used in other class functions to process a image.

Take below original image as an example to illustrate the following image processing pipline

**Original Image**
![alt text][image3]

#### 1. Provide an example of a distortion-corrected image.

`CImgProcessor.undistorted()` is to undistort an image

**Undistorted Image**
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A thresholded binary image is combined by three criterias: direction, magnitude and saturation, with following logical rule:
`((direction AND magnitude) OR saturation)`

*Refer to `AdvancedLaneLines.py` function `process_image` for parameters setting*

**Warped Binary Image**
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

`CImgProcessor.unwarp()` is to unwarp an image.
By inputting two points of a trapezoid as argument `x1, x2, y1, y2`, which are mirrored as a symmetric trapezoid is the area of transformation.
Those two points are (450, 520) & (720, 100).

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

`CLocatelines` is a class to identify lane-line pixels and save the results to `CLine` class
1. At first time, `CLocatelines.__sliding_window()`is used to fit polynomial for both left and right lane lines.
2. After that, `CLocatelines.__poly_fit_prev()` is used to fit polynomial by utilitizing previous polynomial coefficients.
3. Radius of left and right lines, lane width are checked if they are within acceptable ranges. If not, EMWA method is used to obtain weighted result of the fitting polynomial coefficients by considering the previous result too. Or Sliding Window method is repeated once again if the result is way too worse.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The detected points obtained in previous step are feeded to `CLine` class by calling the function `CLine.update()`
Then the radius of curvature and the position of the vehicle with respect to center are calculated and stored as private members.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

At the end, the detected lane line area is marked. Refer to the following image as an example

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Videos are rendered in `AdvancedLaneLine.py` program. Function `process_image` is to render each image with the use of `CImgProcessor` and `CLocateLines` classes.
Below are the links to three videos rendered.

#### project_video
1. Pretty good on `project_video`, lanes are tracked in most of the time despite some disturbance from the shadows
[project_video][video1]

2. Quite terrible with `challenge_video`, but I guess more than 60% of the time is usable result.
[challenge_video][video2]

3. Very bad with `harder_challenge_video`, I would not want to be in this car.
[harder_challenge_video][video3]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Having added sanity check, and putting previous result into considerations to each processed image. Better results are obtained from all three videos. Most of the time in `project_video`, it works fine with finding the lane. Even with some disturbance from the tree's shadow, the result is quite robust.

However it is quite terrible with `challenge_video`. lanes finding are often misled by the shadow and the mark on the ground. One of the ways to distinugish from **shadow** and **real lane** could be using color. Lane color should only be white or yellow, while shadow's color is often different scale of grey.

The same algorithm applied on `harder_challenge_video` is much worse. Constant change in video brightness, shadow of the trees affect the processed image a lot. And sharp turns make results prior to current frame more unreliable, weight of the previous frame should be smaller in such case.
