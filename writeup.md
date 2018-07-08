

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[hog]: ./output_images/hog_image.PNG


[search_grids]: ./output_images/search_grids.PNG
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Requriemetns: [Rubric](https://review.udacity.com/#!/rubrics/513/view) 

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In `lesson_functions.py`
```python
def extract_features
```
is a wraper function that takes the path of dataset as argument.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

This image is HOG feature with only R channel in `RGB` color space
`RGB` color space.
![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the final optimal result is to use `HLS` color space with **ALL** channels

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In `train_classifier_hog.ipynb`
I used SVM as classifier. The feautures are composed of **hog features**, **color features** and **spatial feautres**

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the sake of efficiency, the hog features are extracted once and using the sub-sampling. Codes are in
`vehicle_detector.py`
```python
def find_cars
```

The scales of the search grid need a lot of trails. you can check
`vehicle_detection_hog.ipynd`
section **Generate grids**

![alt text][search_grids]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
**[SVM and HOG](https://www.youtube.com/watch?v=BOG2ufoKnHw)**

**[YOLOv3 and SSD](https://www.youtube.com/watch?v=pbxRuyjMF3Q)**


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's the debug view of the result vedio: **[SVM and HOG](https://www.youtube.com/watch?v=BOG2ufoKnHw)**. You will see the raw detections, heatmap, label, and merge detection windows.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* SVM-HOG method is time-consuming of parameters tuning and not reliable to different driving scenarios.

* YOLOv3 outperform SSD for the small object detections(the opposite-driving direction cars). 

