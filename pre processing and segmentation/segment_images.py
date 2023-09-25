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

# Define the directory
root_dir = 'plantvillage/color'
cutted_dir = 'cutted_images/cutted1'

for directory in os.listdir(root_dir):
    if not (str(directory).endswith('.DS_Store')):

        # Create folder        
        cutted_path = cutted_dir+'/'+directory
        print("Directory '%s' created" %cutted_path)
        os.mkdir(cutted_path)

        for file in os.listdir(root_dir+'/'+directory):
            if not ((str(root_dir)+'/'+str(directory)).endswith('.DS_Store')):

                image_path = root_dir+'/'+directory+'/'+file

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

                # Find the edges by using active countours.
                # Blurring for removing the noise

                print("Applying snakes on '%s'" %image_path)
                # 1st iteration with the following parameters:
                snake = active_contour(gaussian(image_filtered, 3, preserve_range=False),
                                        init, alpha=0.001, beta=5, gamma=0.001, w_line=-0.035)

                # 2nd iteration with the following parameters:
                snake = active_contour(gaussian(image_filtered, 3, preserve_range=False),
                                        snake, alpha=0.1, beta=10, gamma=0.001, max_px_move=1)

                # Plot the image, with the segmented line
                #fig, ax = plt.subplots(figsize=(7, 7))
                #ax.imshow(image_filtered, cmap=plt.cm.gray)
                #ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
                #ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
                #ax.set_xticks([]), ax.set_yticks([])
                #ax.axis([0, image_filtered.shape[1], image_filtered.shape[0], 0])
                #plt.show()

                print("Applying mask on '%s'" %image_path)
                # IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
                #image_original = cv2.imread('plantvillage/color/Tomato___Target_Spot/0a610d40-f5b8-4580-8e32-cc90b3620017___Com.G_TgS_FL 8191.JPG', -1)
                # mask defaulting to black for 3-channel and transparent for 4-channel
                mask = np.zeros(image_original.shape, dtype=np.uint8)
                roi_corners = np.array(snake, dtype=np.int32)
                # fill the ROI so it doesn't get wiped out when the mask is applied
                channel_count = image_original.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (255,)*channel_count
                cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
                # from Masterfool: use cv2.fillConvexPoly if you know it's convex

                # Apply the mask
                masked_image = cv2.bitwise_and(image_original, mask)

                print("Saving the following image: '%s'" %image_path)
                image_file = cutted_path+'/'+file
                # Save the result
                cv2.imwrite(image_file+'.jpg', masked_image)