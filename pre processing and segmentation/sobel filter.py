# Import the required libraries
from PIL import Image
#import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread
from skimage.filters import gaussian
from skimage import filters
import cv2
import numpy as np
import os

image_path = "/Users/matiasluna/Library/CloudStorage/OneDrive-UniversidadedoPorto/CC4016 - Visão Computacional/Projeto/plantvillage/color/Potato___Late_blight/0acdc2b2-0dde-4073-8542-6fca275ab974___RS_LB 4857.JPG"

# Create the initial form of the snake. A circle.
s = np.linspace(0, 2*np.pi, 1200)
r = 135 + 150*np.sin(s)
c = 120 + 150*np.cos(s)
init = np.array([r, c]).T

print("Reading '%s'" %image_path)
# Read the image
image_original = imread(image_path)
image_original = np.array(image_original)         # Transform the image to array
#img = rgb2gray(img)        # Transform the image to grayscale

print("Transforming '%s'" %image_path)
# Find edges in an image using the Sobel filter. Computes the gradient of the image.
image_filtered =  filters.sobel(image_original)

# Plot the image, with the sobel filter applied

plt.imshow(image_filtered)
plt.show()
