# Image Processing in Python

## Introduction
Image processing involves performing operations on images and videos to:
- Enhance images
- Extract useful information
- Analyze and make decisions

## Applications
Image processing is widely used in various fields, including:
- Medical image analysis
- Artificial intelligence
- Image restoration and enhancement
- Geospatial computing
- Surveillance
- Robotic vision
- Automotive safety

## Purposes of Image Processing
1. **Visualization** - Making invisible objects visible
2. **Image Sharpening and Restoration** - Enhancing image quality
3. **Image Retrieval** - Finding specific images
4. **Measurement of Patterns** - Analyzing image patterns
5. **Image Recognition** - Identifying objects within an image

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

## Converting to Grayscale
```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Image Resizing
```python
resized_image = cv2.resize(image, (400, 400))
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Edge Detection using Canny Algorithm
```python
edges = cv2.Canny(image, 100, 200)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

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

## Visualizing Images
```python
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
```

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

### Image Shape and Size
```python
madrid_image.shape  # (height, width, channels)
madrid_image.size  # Total number of pixels
```

### Flipping Images
#### Vertically
```python
import numpy as np
vertically_flipped = np.flipud(madrid_image)
show_image(vertically_flipped, 'Vertically flipped image')
```
#### Horizontally
```python
horizontally_flipped = np.fliplr(madrid_image)
show_image(horizontally_flipped, 'Horizontally flipped image')
```

## Histograms in Image Processing
### Generating Histograms
```python
plt.hist(red.ravel(), bins=256)
```

## Thresholding
Thresholding is used to separate an image into foreground and background by converting it into a binary format (black and white).

### Applying Thresholding
```python
thresh = 127
binary = image > thresh
show_image(image, 'Original')
show_image(binary, 'Thresholded')
```

### Inverted Thresholding
```python
inverted_binary = image <= thresh
show_image(image, 'Original')
show_image(inverted_binary, 'Inverted thresholded')
```

### Global vs. Local Thresholding
- **Global Thresholding** is best for uniform backgrounds.
- **Local (Adaptive) Thresholding** is used for uneven lighting conditions.

#### Using Otsu's Method (Global Thresholding)
```python
from skimage.filters import threshold_otsu
thresh = threshold_otsu(image)
binary_global = image > thresh
show_image(image, 'Original')
show_image(binary_global, 'Global Thresholding')
```

#### Using Local Thresholding
```python
from skimage.filters import threshold_local
block_size = 35
local_thresh = threshold_local(text_image, block_size, offset=10)
binary_local = text_image > local_thresh
show_image(text_image, 'Original')
show_image(binary_local, 'Local Thresholding')
```

## Filtering
Filters help enhance images by:
- Emphasizing or removing features
- Smoothing
- Sharpening
- Edge detection

### Edge Detection using Sobel Filter
```python
from skimage.filters import sobel
edge_sobel = sobel(image_coins)
plot_comparison(image_coins, edge_sobel, "Edge with Sobel")
```

### Gaussian Smoothing
```python
from skimage.filters import gaussian
gaussian_image = gaussian(amsterdam_pic, multichannel=True)
plot_comparison(amsterdam_pic, gaussian_image, "Blurred with Gaussian filter")
```

## Contrast Enhancement
### Histogram Equalization
```python
from skimage import exposure
image_eq = exposure.equalize_hist(image)
show_image(image, 'Original')
show_image(image_eq, 'Histogram equalized')
```

### Adaptive Equalization (CLAHE)
```python
from skimage import exposure
image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
show_image(image, 'Original')
show_image(image_adapteq, 'Adaptive equalized')
```

## Image Transformations
### Rotating Images
```python
from skimage.transform import rotate
image_rotated = rotate(image, -90)
show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees clockwise')
```

### Rescaling and Resizing
```python
from skimage.transform import rescale, resize
image_rescaled = rescale(image, 1/4, anti_aliasing=True, multichannel=True)
image_resized = resize(image, (400, 500), anti_aliasing=True)
show_image(image, 'Resized image')
```

## Morphological Operations
- **Dilation** expands object boundaries
- **Erosion** shrinks object boundaries

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