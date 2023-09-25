# Imports
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, losses
from matplotlib import pyplot as plt

epochs = 5

#Load the history with pickle as well:
with open('/content/gdrive/MyDrive/plantvillage/training1_history.pkl', 'rb') as f:
    history = pickle.load(f)

print("[INFO] Plotting the history results of the model")
# Plot the results
accuracy = history[tf.keras.metrics.Accuracy()]
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
false_negatives = history[tf.keras.metrics.FalseNegatives()]
false_positives = history[tf.keras.metrics.FalsePositives()]
precision = history[tf.keras.metrics.Precision()]

epochs_range = range(epochs)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, accuracy, label = 'Training Accuracy')
plt.plot(epochs_range, val_accuracy, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylim(0,1)
plt.title('Training and Validation Accuracy', fontsize = 15)

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.ylim(bottom = 0)
plt.title('Training and Validation Loss', fontsize = 15)

plt.show()

plt.figure()
plt.plot(epochs_range, false_negatives, label='False negatives')
plt.plot(epochs_range, false_positives, label='False positives')
plt.plot(epochs_range, precision, label='Precision')
plt.title('False negatives/positives and precision')
plt.legend()
plt.show()
