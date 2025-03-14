# Image Processing in Python

## Introduction
Image processing involves performing operations on images and videos to:
- Enhance images
- Extract useful information
- Analyze and make decisions

## Applications
Image processing is widely used in various fields, including:
- **Medical Image Analysis**: Identifying diseases from X-rays, MRIs, and CT scans.
- **Artificial Intelligence**: Training models for object detection, face recognition, and more.
- **Image Restoration and Enhancement**: Removing noise and improving image quality.
- **Geospatial Computing**: Processing satellite images for land classification and mapping.
- **Surveillance**: Enhancing security footage for crime investigation.
- **Robotic Vision**: Enabling robots to perceive their environment.
- **Automotive Safety**: Lane detection and autonomous driving assistance.

## Purposes of Image Processing
1. **Visualization** - Making invisible objects visible by enhancing image features.
2. **Image Sharpening and Restoration** - Improving image clarity and removing distortions.
3. **Image Retrieval** - Searching and identifying images in large datasets.
4. **Measurement of Patterns** - Extracting meaningful patterns and shapes from images.
5. **Image Recognition** - Identifying objects, faces, or handwritten text in an image.

## Introduction to OpenCV
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning library. It is designed for computational efficiency and real-time applications.

### Installing OpenCV
```bash
pip install opencv-python
```

## Loading and Displaying an Image
```python
import cv2
image = cv2.imread('image.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- `cv2.imread()` loads an image from a file.
- `cv2.imshow()` displays the image in a window.
- `cv2.waitKey(0)` waits for a key press before closing the window.
- `cv2.destroyAllWindows()` closes all OpenCV windows.

## Converting to Grayscale
```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- Grayscale images reduce computational complexity by converting color images to a single channel.

## Image Resizing
```python
resized_image = cv2.resize(image, (400, 400))
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- Resizing helps normalize image dimensions before processing.

## Edge Detection using Canny Algorithm
```python
edges = cv2.Canny(image, 100, 200)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- The Canny algorithm detects edges by finding areas of rapid intensity change.

## Introduction to Scikit-Image
Scikit-Image is another powerful Python library for image processing, offering:
- An easy-to-use interface
- Machine learning capabilities
- Pre-built complex algorithms

## Understanding Images
An image can be represented as an array of pixel values.

### Working with Images in Scikit-Image
```python
from skimage import data
rocket_image = data.rocket()
```

### RGB vs. Grayscale
```python
from skimage import color
grayscale = color.rgb2gray(original)
rgb = color.gray2rgb(grayscale)
```

- **RGB images** contain three color channels (Red, Green, Blue).
- **Grayscale images** contain a single intensity channel.

## Visualizing Images
```python
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
```

- `plt.imshow()` displays an image.
- `plt.axis('off')` removes axis ticks for clarity.

## NumPy for Image Processing
### Loading Images as NumPy Arrays
```python
import matplotlib.pyplot as plt
madrid_image = plt.imread('madrid.jpeg')
type(madrid_image) # <class 'numpy.ndarray'>
```

### Extracting RGB Channels
```python
red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]
```

- Each pixel consists of three intensity values (R, G, B).

### Image Shape and Size
```python
madrid_image.shape  # (height, width, channels)
madrid_image.size  # Total number of pixels
```

## Histograms in Image Processing
### Generating Histograms
```python
plt.hist(red.ravel(), bins=256)
```

- Histograms visualize the distribution of pixel intensities.

## Thresholding
Thresholding separates an image into foreground and background.

### Applying Thresholding
```python
thresh = 127
binary = image > thresh
show_image(image, 'Original')
show_image(binary, 'Thresholded')
```

- Pixels above the threshold become white, and others turn black.

## Filtering
Filters enhance images by:
- Emphasizing features
- Smoothing
- Sharpening
- Detecting edges

### Edge Detection using Sobel Filter
```python
from skimage.filters import sobel
edge_sobel = sobel(image_coins)
plot_comparison(image_coins, edge_sobel, "Edge with Sobel")
```

## Contrast Enhancement
### Histogram Equalization
```python
from skimage import exposure
image_eq = exposure.equalize_hist(image)
show_image(image, 'Original')
show_image(image_eq, 'Histogram equalized')
```

- Equalization enhances image contrast.

## Image Transformations
### Rotating Images
```python
from skimage.transform import rotate
image_rotated = rotate(image, -90)
show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees clockwise')
```

## Morphological Operations
- **Dilation** expands object boundaries.
- **Erosion** shrinks object boundaries.

### Binary Erosion
```python
from skimage import morphology
eroded_image = morphology.binary_erosion(image_horse)
plot_comparison(image_horse, eroded_image, 'Erosion')
```

### Binary Dilation
```python
dilated_image = morphology.binary_dilation(image_horse)
plot_comparison(image_horse, dilated_image, 'Dilation')
```

## Conclusion
This README provides an introduction to image processing concepts using Python, OpenCV, scikit-image, and NumPy. These tools allow for powerful image manipulation, analysis, and enhancement.
