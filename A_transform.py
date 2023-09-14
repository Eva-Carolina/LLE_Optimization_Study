# import the necessary packages
from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

#canny and canny variation
#apply normal canny
#edged = cv2.Canny(gray, 10, 50)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	print(v)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def my_canny(img, sigma=1.0, low_threshold=100, high_threshold=200,ksize=15,b_size=25):
    # Apply Gaussian filter to smooth the image
    blur =cv2.GaussianBlur(img, (b_size,b_size), sigma)

    # Compute gradient magnitude and direction using Sobel operator
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize)
    magnitude = cv2.magnitude(sobelx, sobely)
    direction = cv2.phase(sobelx, sobely, angleInDegrees=True)

    # Perform non-maximum suppression to thin the edges
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            if direction[i,j] < 0:
                direction[i,j] += 180
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                n1 = magnitude[i,j+1]
                n2 = magnitude[i,j-1]
            elif (22.5 <= direction[i,j] < 67.5):
                n1 = magnitude[i+1,j-1]
                n2 = magnitude[i-1,j+1]
            elif (67.5 <= direction[i,j] < 112.5):
                n1 = magnitude[i+1,j]
                n2 = magnitude[i-1,j]
            else:
                n1 = magnitude[i-1,j-1]
                n2 = magnitude[i+1,j+1]
            if (magnitude[i,j] >= n1) and (magnitude[i,j] >= n2):
                suppressed[i,j] = magnitude[i,j]

    # Apply hysteresis thresholding to detect and link the edges
    edges = cv2.Canny(img, low_threshold, high_threshold)

    return edges

#dilate and erode
def apply_morphology_operations(img, size=5):
    # Define kernel
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    
    # Dilate image
    img = cv2.dilate(img, kernel_1, iterations=1)
    
    # Erode image
    img = cv2.erode(img, kernel_1, iterations=1)
    
    return img

#pre-treatment
def apply_histogram_equalization(img):
    # Apply histogram equalization
    img_equalized = cv2.equalizeHist(img)
    return img_equalized

def apply_clahe_HE(img, clip_limit=2.5, tile_size=(8,8)):
    # Create a Contrast Limited Adaptive Histogram Equalization (CLAHE) object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Apply adaptive histogram equalization using the CLAHE object
    img_adapteq = clahe.apply(img)
    return img_adapteq

def unsharp_masking_helena(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=90):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharp = float(amount + 1) * image - float(amount) * blurred
    sharp[sharp > threshold] = threshold+50
    sharp = sharp.astype('uint8')
    return sharp

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharp = float(amount + 1) * image - float(amount) * blurred
    sharp[sharp < threshold] = threshold
    sharp = sharp.astype('uint8')
    return sharp

def unsharp_masking_mod(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharp = float(amount + 1) * image - float(amount) * blurred
    sharp[sharp < threshold] = threshold-(threshold)
    sharp = sharp.astype('uint8')
    return sharp

def contrast_stretching(image):
    min_val, max_val, _, _ = cv2.minMaxLoc(image)
    stretched = cv2.normalize(image, None, min_val, max_val, cv2.NORM_MINMAX)
    return stretched 

def high_pass_filter(image, kernel_size=(15, 15)):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(image - laplacian)
    return sharpened

def median_filtering(image, kernel_size=15):
    filtered = cv2.medianBlur(image, kernel_size)
    return filtered

def display(images,scale=3):
    """
    Given a list of images, resize them to fit the screen and stack them in a grid with two rows.
    """
    num_images = len(images)
    if num_images < 1:
        return None
    elif num_images == 1:
        return cv2.resize(images[0], (640, 480))
    else:
        h, w = images[0].shape[:2]
        resized_images = [cv2.resize(img, (w//scale, h//scale)) for img in images]
        if num_images % 2 != 0:
            resized_images.append(np.zeros_like(resized_images[0])) # add a blank image if the number of images is odd
        num_cols = (num_images + 1) // 2
        rows = []
        for i in range(0, num_images, num_cols):
            row = np.hstack(resized_images[i:i+num_cols])
            rows.append(row)
        grid = np.vstack(rows)
        return grid

def display_row(images, scale=3):
    """
    Given a list of images, resize them and concatenate them into a single row.
    """
    num_images = len(images)
    if num_images < 1:
        return None
    elif num_images == 1:
        return cv2.resize(images[0], (640, 480))
    else:
        h, w = images[0].shape[:2]
        resized_images = [cv2.resize(img, (w // scale, h // scale)) for img in images]
        row = np.hstack(resized_images)
        return row

