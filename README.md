# Lane Lines Detection Using Python and OpenCV
---
**In this project, I have detected lane lines on the road using _Python_ and _OpenCV_.**

**I have developed a computer vision pipeline that processes a group of test images then applied this pipeline to two test video streams.**

![example_output](https://i.imgur.com/DhInZRk.jpg)

## Pipeline Architecture:
---
1. Load test images
2. Apply grayscale transform
3. Smooth the image to suppress noise and any spurious gradients
4. Apply Canny edge detection algorithm
5. Cut-out any edges that are out of the lane lines region (**region of interest**)
6. Get the lines located in the region of interest using hough transform
7. Fitting and extrapolating these lines for both right and left lane
   * Determining the x and y points of each of the right and left lane lines
   * Extrapolating or fitting these points for both lane lines
8. Draw the lane lines on the original image
   * Drawing the lane lines on a blank image firstly
   * Weighting the original image with the lane lines image

In the next, I will be explaining each step of the pipeline.
## Environment:
---
* Ubuntu 16.04 LTS
* Python 3.6.4
* OpenCV 3.1.0
* Anaconda 4.4.10

### 1. Load test images
---
I have loaded the test images from the test_images directory. 
These images are loaded to a list `test_image` which will be used for feeding the pipeline with the test images for detecting the lane liens.
```python
images_list = os.listdir("test_images/")
test_images = []
for img in images_list:
    test_images.append(mpimg.imread("test_images/"+img))
```
### 2. Apply grayscale transform
---
The goal of this step is to convert the image under processing to grayscale level. For many applications of image processing, color information doesn't help us identify important edges or other features. So, it is preferable in many computer vision appplications to have the image in grayscale. Specifically, for our goal here to detect lane lines in an image, it is needed to convert the image to grayscale to get the best output of the next steps of our pipeline.
In the following is my function to convert to grayscale:
```python
def grayscale(image):
    """
    Description:
        Applies the gray-scale transform
    Parameters:
        image: A color input image
    Output:
        A gray-scaled image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```
### 3. Smooth the image to suppress noise and any spurious gradients
---
Here we apply the gaussian smoothing which is essentially a way of suppressing noise and spurious gradients by averaging.
We choose the `kernel_size` of Gaussian smoothing to be any odd number.
A large `kernel_size` implies averaging ,or smoothing, over a large area
In the following is my function that uses the `cv2.GaussianBlur` function to apply the gaussian smoothing.
```python
def gaussian_blur(image, kernel_size = 5):
    """
    Description:
        Smoothes the image by applying guassian filter to the image
    Parameters:
        image: A gray-scaled input image
        kernel_size (Default = 5): This is the window the convolves the whole image applying the filter to the image
    Output:
        Smoothed (Averaged or Filtered) image
    """
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```
### 4. Apply Canny edge detection algorithm
---
#### Firstly, What is an edge ?
Rapid changes in brightnesss is that we call **_edges_**.

### What is Canny Edge Detection Algorithm ?

**From Wikipedia:**

```
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.

Process of Canny edge detection algorithm

1. Apply Gaussian filter to smooth the image in order to remove the noise
2. Find the intensity gradients of the image
3. Apply non-maximum suppression to get rid of spurious response to edge detection
4. Apply double threshold to determine potential edges
5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
```
The goal of **Canny Edge Detection** is to identify the boundaries of an object of an image (specifiacally here, our lane lines).
#### What is the intensity gradient of an image ?
You can think of it as the strength an edge as being defined by the how different the values are in adjacent pixels in the image.
As our image is a mathematical function of x and y "f(x,y)" so we can perfom mathematical operations on it.
So taking the derivative of this function w.r.t x and y simultaneously is called the gradient.

#### What is the impact of `low_threshold` and `high_threshold` values ?
The canny edge detection algorithm will firstly detect strong edges (strong gradients pixels) above the high threshold and rejects pixels below the low threshold. Pixels with values between low and high thresholds will be included as long as they are connected to strong edges. The ouputs is a binary image with pixels tracing out the detected edges and black everything else.

The following is the **Canny Edge Detection** function that uses `cv2.Canny`

```python
def canny(image, low_threshold  = 50, high_threshold = 150):
    """
    Description:
        Canny edge detector which detects strong edges (strong gradients pixels) above the high threshold and 
        rejects pixels below the low threshold. Pixels with values between low and high thresholds will be included
        as long as they are connected to strong edges
    Parameters:
        image: The input gray-scaled smoothed image
        low_thresold (Default = 50): The threshold  value of rejecting pixel
        high_threshold (Default = 150): The threshold of strong edges in the image
    Output:
        A binary image with pixels tracing out the detected edges and black everything else
    """
    
    return cv2.Canny(image, low_threshold, high_threshold)
```

### 5. Cut-out any edges that are out of the lane lines region (region of interest)
---
As our goal is to identify and detect the lane lines on the road so we are not interest in other shapes, objects, ... etc in the image.
So, here we cut-out and focus on the region of the lane lines will be located assuming that there is a front-camera mounted at the center of the vehicle (say it is mounted on the ceter mirror).
This region of interest is defined by a polygon in which the detected edges will appear and black everywhere else.

Here is the function that defines the polygon of region of interest and masks it:
```python
def region_of_interest(image):
    """
    Descrition:
        Applies an image mask.Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
    Paramteres:
        image: The image from Canny edge detector
    Output:
        An image that has the edges in the defined polygon and black everywhere else
    """
    
    image_shape  = image.shape
    # The following points are the vertices points of the polygon
    bottom_left  = (110,image_shape[0])
    up_left      = (450, image_shape[0]/1.65)
    bottom_right = (image_shape[1]-450, image_shape[0]/1.65)
    up_right     = (image_shape[1]-20,image_shape[0])
    vertices     = np.array([[bottom_left, up_left, bottom_right, up_right]], dtype=np.int32)
    
    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
```

### 6. Get the lines located in the region of interest using Hough Transform
---
#### What is Hough Transform ?

**From Wikipedia:**
```
The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing.[1] The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
```

Simply speaking, the hough transfrom will take an image (canny edge detectoin and region of interes applied) which has the shape of lane lines scatered along it as white dots/edges/small or long lines. The hough transform officially gives you the parameters of that line that passes through each piece of lane line (based on how you define hough parameters). Using the `cv2.HoughLinesP`, it will give a list having the whole lines of that image. Each item of this list is the start and end (x,y) coordinates of that line.

Here is the function for hough tranfrom:
```python
def hough_transform(image):
    """
    Descrition:
        Hough Transfrom is used to indetify lines located in the region of interest
    Parameters:
        image: This is the output a Canny transform and after applying the region of interest mask.
    Output:
        Returns a list with hough lines.
    """
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines
```
### 7. Fitting and extrapolating these lines for both right and left lane
---
Now we have the lane lines but at least one of them is segmented and can not be drawn as a complete line on the image.
Here, we develop a way of extrapolating the lane lines. This is done through two steps:
* Determining the x and y points of each the right and left lane lines
* Extrapolating or fitting these points for both lane lines
#### Determining the x and y points of each the right and left lane lines
In order to be able to extrapolate the line, we need firstly to determine whether this line belongs to the right lane of the left lane.
To do so, we will depend on the slope of the lines detected from the hough transfrom. If the slope is positive, so this indicates a line belonging to the left lane line. On the other hand, if the slope is negative, so the line is beloning to right lane line. For both lines, we collect the x and y points to be able to extrapolate the line in the next step.

Here is the function for determining the right lane line points:
```python
def get_right_lane_points(lines_image):
    """
    Descrition:
        This function calculates the slope of each line detected from hough transform. 
        If the slope is +ve, then this line belongs to the right lane.
        Note: 0.4 is selected for better filteration of the lines ouput
    Parameters:
        lines_image: This is the output of hough tranfrom (a list contains the hough lines)
    Output
        right_lane_x_points: a list contains the all x points of the right lane
        right_lane_y_points: a list contains the all y points of the right lane
    """
    
    right_lane_x_points = []
    right_lane_y_points = []
    
    for line in lines_image:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if(slope > 0.4): #+ve slope -> right lane
                #print("Right Lane Detected\n")
                right_lane_x_points.append(x1)
                right_lane_x_points.append(x2)
                right_lane_y_points.append(y1)
                right_lane_y_points.append(y2)
                
    return right_lane_x_points, right_lane_y_points
```
And this is the function for determining the left lane line points:
```python
def get_left_lane_points(lines_image):
    """
    Descrition:
        This function calculates the slope of each line detected from hough transform. 
        If the slope is -ve, then this line belongs to the left lane.
        Note: -0.6 is selected for better filteration of the lines ouput
    Parameters:
        lines_image: This is the output of hough tranfrom (a list contains the hough lines)
    Output
        left_lane_x_points: a list contains the all x points of the left lane
        left_lane_y_points: a list contains the all y points of the left lane
    """
    
    left_lane_x_points  = []
    left_lane_y_points  = []

    for line in lines_image:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if(slope < -0.6): #-ve slope -> left lane
                #print("Left Lane Detected\n")
                left_lane_x_points.append(x1)
                left_lane_x_points.append(x2)
                left_lane_y_points.append(y1)
                left_lane_y_points.append(y2)
                    
    return left_lane_x_points, left_lane_y_points
```
#### Extrapolating or fitting these points for both lane lines
Now, by knowing the point of the right and left lane lines separetely, we can extrapolate these point and get the whole line passing through these points all.
IF:   y = mx + b
THEN: x = (y - b)/m
Given that: y is defined from the region of interest.

Here is the function for fitting and extrapolating the lane lines:
```python
def fit_lane_line(lane_line_x_points, lane_line_y_points, image_shape):
    """
    Descrition:
        The function extrapolates the detectioned points in each lane from the line equation given that
        Ymin and Ymax are defined from the region of interest.
        IF:   y = mx + b
        THEN: x = (y - b)/m
    Parameters:
        lane_line_x_points: This is a list for the x points of our intended lane line
        lane_line_y_points: This is a list for the y points of our intended lane line
        image_shape: The height and width of the image to get the Ymin and Ymax
        
    Output
        Xmin, Ymin: The start point of our intended lane line
        Xmax, Ymax: The end point of our intended lane line
    """
    
    #Getting the slop and intersect of the right lane line
    fit_lane_line =  np.polyfit(lane_line_x_points, lane_line_y_points, 1)
    m = fit_lane_line[0] # Slope
    b = fit_lane_line[1] # Intercept

    #These are the y values defined at my region of interest
    Ymax = image_shape[0]
    Ymin = image_shape[0]/1.65
    
    # If equation of a line: y = m*x + b
    # Hence: x = (y - b)/m
    Xmax =  (Ymax - b) / m
    Xmin =  (Ymin - b) / m

    #Converting the values from float to int to be suitable for cv2.line
    Ymax = int(Ymax)
    Ymin = int(Ymin)
    Xmax = int(Xmax)
    Xmin = int(Xmin)
    
    return Xmax, Xmin, Ymax, Ymin
```

### 8. Draw the lane lines on the original image
---
In this step, we are:
* Drawing the lane lines on a blank image firstly
* Weighting the original image with the lane lines image

#### Drawing the lane lines on a blank image firstly
We draw the lane line on a blank image firstly knowing the start and end point of each line
```python
def draw_lines(image, Xmin, Xmax, Ymin, Ymax):
    """
    Descrition:
        This function is used to draw the lane line on a blank image       
    Parameters:
        Xmin, Ymin: The start point of our intended lane line
        Xmax, Ymax: The end point of our intended lane line
    Output:
        Image of the lane line drawn on it
    """
    
    #Drawing the lane line on the blank image
    return cv2.line(image,(Xmin,Ymin),(Xmax,Ymax),(255,0,0),10)
```
#### Weighting the original image with the lane lines image
Then we draw the lane lines on the original image
```python
def weighted_img(image, initial_img, α=0.8, β=1., γ=0.):
    """
    Descrition:
        This function is used to weight (combine) the lane lines image and the original one
    Parameters:
        image: The image which has the lane lines
        initial_img: should be the image before any processing.
    Output:
        image is computed as follows: initial_img * α + image * β + γ
        
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, image, β, γ)
```

## Process Pipeline:
This is our algorithm function that wraps and calls the whole parts of our pipeline. It takes a color test image and returns an image in which the lane lines on the road should be annotated and detected.
```python
def process_pipeline(image):
    """
    Descrition:
        This function is save an image to the specified path with the same origianal name
    Parameters:
        image: image under test to detect the lane lines in it
    Output:
        result: The output of our pipeline. It should be the original image with the lane lines annotated.
    """
    image_shape  = image.shape

    gray_scale_image = grayscale(image)
    
    smoothed_gray_scale_image = gaussian_blur(gray_scale_image)
    
    edge_detected_image = canny(smoothed_gray_scale_image)
    
    masked_edge_detected_image = region_of_interest(edge_detected_image)
    
    lines_image = hough_transform(masked_edge_detected_image)
    
    right_lane_x_points , right_lane_y_points = get_right_lane_points(lines_image)
    left_lane_x_points  , left_lane_y_points  = get_left_lane_points(lines_image)
    
    Xright_min , Xright_max , Yright_min , Yright_max = fit_lane_line(right_lane_x_points, right_lane_y_points, image_shape)
    Xleft_min  , Xleft_max  , Yleft_min  , Yleft_max  = fit_lane_line(left_lane_x_points, left_lane_y_points, image_shape)
    
    
    lane_lines_image = np.copy(image)*0  # creating a blank to draw lines on
    lane_lines_image = draw_lines(lane_lines_image, Xright_min , Xright_max , Yright_min , Yright_max)
    lane_lines_image = draw_lines(lane_lines_image, Xleft_min  , Xleft_max  , Yleft_min  , Yleft_max)
    
    result = weighted_img(lane_lines_image, image)
    
    return result
```

## Conclusion
This is the computer vision algorithm pipeline for detecting lane lines on the road of an image. The algorithm is working on straight lane lines of both test images and video streams. 
_Curve lane liens are byond the scope of this projet._

License
----

MIT
