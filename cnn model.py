# Import libraries and dependencies
import tensorflow as tf
import numpy
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import pickle
from matplotlib import pyplot as plt

print("[INFO] Defining parameters")
# Parameters
batch_size = 32
epochs = 5
image_size = (255, 255)

print("[INFO] Load data from the directory")
# Loading the data from the directory
data_dir = "/content/gdrive/MyDrive/plantvillage/color"
#data_dir = "plantvillage/color"

print("[INFO] Define Data Generator")
datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20
)

print("[INFO] Spliting data to train, test and perform Data Augmentation")
# Data augmentation for Training
train_generator = datagen.flow_from_directory(
    data_dir,
    subset='training',
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    target_size=image_size
    #save_to_dir='/content/gdrive/MyDrive/plantvillage/training1_augmented'
    #save_to_dir = "plantvillage/training1_augmented"
)

# Data augmentation for Validation
val_generator = datagen.flow_from_directory(
    data_dir,
    subset='validation',
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    target_size=image_size
    #save_to_dir='/content/gdrive/MyDrive/plantvillage/validation1_augmented'
    #save_to_dir = "plantvillage/validation1_augmented"
)

print("[INFO] Define the classes")
# Define the classes
classes = numpy.unique(train_generator.classes)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Define the model
print("[INFO] Define the model")
# Model architechture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(255, 255, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.24))
model.add(Dense(nClasses, activation='softmax'))

print("[INFO] Compile the model")
model.compile(loss=keras.losses.categorical_crossentropy(from_logits = True),
              optimizer=keras.optimizers.Adam(), 
              metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.Precision()])

print("[INFO] Summary of the model")
model.summary()

# Train the data
print("[INFO] Train the model")
train = model.fit(
    train_generator, 
    batch_size=batch_size,
    epochs=epochs,  
    validation_data=val_generator)

print("[INFO] Save the history of the model")
#Saving the history as a dictionary
history = train.history

# Evaluate the data
print("[INFO] Evaluate the model")
score = model.evaluate(val_generator)

# Show results
print("[INFO] Intial results of the model")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("[INFO] Save history results of the model")
# Pickle the history to file
with open('/content/gdrive/MyDrive/plantvillage/training1_history.pkl', 'wb') as f:
    pickle.dump(history, f)