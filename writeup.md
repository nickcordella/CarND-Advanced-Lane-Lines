## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistorted_checkerboard_5.png "Undistorted Checkerboard"
[image2]: ./output_images/undistorted_test4.png "Undistorted Road"
[image3]: ./output_images/thresholded_test5.jpg "Binary Example"
[image4]: ./output_images/warped_perspective.png "Warp Example"
[image5]: ./output_images/fit_polynomials_test3.jpg "Fit Visual"
[image6]: ./output_images/projection_test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the "Calibrate Camera" section of the IPython notebook located in `./Advanced_Lane_Finding.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the a random image for which chessboard corners were not detected, and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the exact same undistortion function to a test image of the roadway to show that those images are also being undistorted:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The work here is included in the "Color and Gradient Thresholding" section of `Advanced_Lane_Finding.py`, and specifically the exact filters are applied in the `threshold` method. After undistorting, I applied the Sobel Operator on the blue channel of the original image to enable better detection of the yellow line. I combined that with a filter on the saturation values in the HLS color space, as all the lane lines are distinctly more saturated than the road. Here's an example of my output for this step. Its worth noting that this image does probably the worst job of highlighting the lane lines, which is pretty good I think!

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective warping is included in the "Perspective Warping" section of my jupyter notebook. I manually measured source points on an image with straight lane lines, and projected them into destination points that would result in vertical lines from a birds-eye view. I also omitted the bottom part of the image to avoid the hood of the car when choosing these points.

The actual source and destination points used were:

| Source        | Destination   |
|:-------------:|:-------------:|
| 581, 460      | 320, 0        |
| 248, 690      | 320, 720      |
| 1059, 690     | 960, 720      |
| 667, 460      | 960, 0        |

I verified that my perspective transform was working as expected by warping the test image and verifying that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This step is handled in the "Lane line polynomials" section of the jupyter notebook. I first identified two functions - `find_window_centroids` and `fit_lane_pixels_from_windows`.

`find_window_centroids` Started by finding the maximum value of a convolution between a uniform-valued "window template" and the vertical column of my binary warped image with the most pixels over the bottom quarter of the image. Then it incremented the window by a certain height and recentered it, successively until it reached the top of the window. The output was a list of x-value centroids for the left and right windows.

`fit_lane_pixels_from_windows` Takes the same image, along with the paired centroids, and effectively filters out the nonzero pixels of the binary image that don't fall within the set of windows that are specified by the window centroids, and their height and width.  Then we have simply a collection of pixels for each lane boundary, that can be fit to a second degree polynomial using `np.polyfit`. An example image is shown below.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature calculation is defined in the `radius_calc` method, which is invoked in the `fit_lane_pixels_from_windows` method, and then passed to the plotting scripts.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in the "Projection onto road" section of the jupyter notebook, using example code given in the "Tips and Tricks portion of the project description"

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video_images/lane_tracking_smoothed.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I thought I did a decent job honing in on good filters to use for recognizing lane lines, but there were of course hiccups when it was run on a long, continuous video clip. In order to get rid of them I implemented a couple "sanity checks within" the `process_image_smoothed` function. It was a rather simple-minded implementation that told the lane estimator to not update itself if any of the the following conditions were met between frames:
* The lane offset jumped by more than 0.5 meters
* The radius of curvature was already small (< 1km) and shrank again by more than 30%
* The lane separation deviated by more than 0.3 meters from the known standard of 3.7 meters

While these rules were effective in smoothing a one minute video, I imagine they would be not so successful out in the real world. I'd want to implement something a bit more sophisticated, like proper smoothing of lines, that was not susceptible to edge cases I haven't thought of yet.

Another area where my implementation could be improved is the performance. Instead of re-calculating the window centroids for every single frame based on convolutions, I could have projected the last frame's polynomial to filter lane pixels in a more efficient manner. It was not too tedious to use the slower methods in this exercise, but in an actual car, this becomes a matter of safety!
