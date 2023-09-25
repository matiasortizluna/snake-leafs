# Welcome to snake-leafs repository

Computer Science Master's degree “Computer Vision” subject project: Application of different filters and active countour model to detect the leaf form and apply Deep Learning Convulotional Neural Network to train a model for classification of the leaf type and disease. By Matias Luna & Rodrigo Martins.

![Segmentation using Active Countours](https://github.com/matiasortizluna/snake-leafs/assets/64530615/5e2d4498-9db2-4630-932b-5f6be4361fbb)

The organization of the files and order to be run are:

1. Pre-Processing and Segmentation 

   1.1. sobel filter.py

   1.2. segment_images.py

   1.3. select snake region from image.py

   1.4. plot results.py

2. Deep Learning and Classification

   2.1. cnn model.py

   2.2. connect to google drive in collab.py

   2.3. gaussain filter.py

# The project 

The dataset used was from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset, and consisted of 4 GB of pictures of leaves in 3 folders. The first folder contained images in colour, the second it contained pictures in grayscale and folder number 3 had the same pictures as the other 2, but with segmentation already performed, leaving out the background and shadows of each leaf. Within each folder there were 14 different types of fruits and vegetables, ranging from Apple to pepper, and each type had between 0 and 9 types of diseases with their respective dataset. All kinds of leaves had a reasonable number of photos representing the healthy state of the plant, but the size of the folders with the defect foliage varied significantly in size and variety, that why later we perform a data augmentation technique. 

# Pre-Processing and Segmentation 
The first step in the image processing pipeline involves applying a Gaussian filter, commonly known as a Gaussian blur, to the image. The Gaussian filter is a low-pass filter that helps reduce high-frequency noise and unwanted details in the image. It achieves this by convolving the image with a Gaussian kernel, effectively smoothing out the pixel intensities and creating a blurred version of the original image.

Following the blurring process, the Sobel filter is applied. The Sobel filter is used for edge detection in images. It calculates the gradient magnitude of the image at each pixel, identifying regions of rapid intensity changes which typically correspond to edges in the image. The Sobel filter uses convolution with specialized kernels to estimate the gradient in the x and y directions, helping highlight these edges.

The final step involves utilizing an active contour algorithm to accurately detect the form or outline of the leaf in the processed image. Active contour models, also known as snakes, are deformable curves or contours that iteratively adjust their shape to align with significant features in the image. The active contour with the inwards direction was selected which eliminated the background all together, but it didn’t suppress most shadows. The parameters which contributed the most for the good performance of the active contour were the alpha,which made the “snake” contract faster, beta, smoothed out the line in order to contour correctly regions with high noise and not output sharp corners caused by noise and w_edge to incentivise it to move towards the edges. We also used the first snake as the starting contour for the second snake and used negative values on w_line to attract toward the shadows of the leaf, low alpha and max_px_move to limit how it could compress, and this way the active contour was capable of somewhat ignoring the shadows while keeping most of the relevant part (the leaf) intact. 

In summary, this processing pipeline starts by reducing noise and unwanted details using a Gaussian filter, followed by enhancing edges with a Sobel filter, and finally employing an active contour algorithm to precisely outline the shape of a leaf in the image. The combined effect of these steps helps in effectively detecting and characterizing the leaf's structure and form within the given image.

The final step was to cut the image considering the edges found by the active countours, but due to the lack of available solutions online to perform this action, the result was not 100% effective, we encounter an rotation issue when cutting the image.

# Deep Learning and Classification

Data augmentation is a technique used to increase the size of a dataset by applying various transformations to the existing data, such as rotation, scaling, translation, and flipping. In this context, data augmentation is performed to balance the number of images for each class in the dataset. By generating augmented versions of the images, you ensure that each class has a more uniform representation, which is crucial for training a balanced and effective classification model. Data augmentation is typically done before any further preprocessing or model training to enhance the diversity and robustness of the dataset.

After data augmentation and the Pre-processing and segmentation actions were applied, the dataset is split into two subsets: a training set and a testing set. The training set, constituting 70% of the data, is used to train the model. The testing set, comprising 30% of the data, is reserved for evaluating the model's performance after training. This separation helps in assessing the model's generalization and how well it performs on unseen data.

A Convolutional Neural Network (CNN) is designed to handle and learn features from images effectively. The architecture of this CNN includes:

2D Convolution Layer: this layer creates a convolution kernel that is multiplied with the nodes of the layers input, which helps produce a tensor of outputs (nodes). A kernel is essentially a matrix or mask that is  used for blurring, sharpening, embossing, edge detection, and more by doing a convolution (series of multiplications) between the kernel matrix and the image (represented as a matrix).

2D Max Pooling Layer: It essentially summarises the features present in a region of the feature map generated by a convolution layer, so the model becomes more generic.This makes the model more robust to variations in the position of the features in the input image. This layer is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map

Flatten Layer: Is a layer that is used to make the multidimensional input one-dimensional, commonly used in the transition from the convolution layer to the fully connected layer.

Dense Layer: Is a layer that is deeply connected with its preceding layer, which means that all the neurons of the layer are connected to every neuron of its preceding layer. This layer is the most commonly used layer in artificial neural network networks. 

Dropout Layer: Is layer that is a mask that nullifies the contribution of some neurons towards the next layer and leaves unmodified all others. It basically removes from the some of nodes from the network. They are important in training CNNs because they prevent overfitting on the training data.

# Results


# Conclusions
Active contours has nice features, but only in very specialised datasets, with good control environments and conditions​.

There is another approach for the active countours which is to perform balloning, essentially this means that we start from the center and then continue unti the edges of the image, but we chose not to do ballooning due to the high frequency gradients, high contrast and texture of the nerves and components of leaves which led the active contour to not perform optimally. 

There are many frameworks and libraries to work more efficiently and more easily with deep learning, each one with their own advantages and disadvantages. Most are easy to manipulate and tune to your specific requirements and are reasonably well documented. ​The task of image classification is very computationally and time demanding, specially with limited personal hardware and knowledge about deep learning networks.
