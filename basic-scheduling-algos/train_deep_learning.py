import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.keras.models import Sequential
class DeepLearningJob:
    def __init__(self, epochs=5, job_name="DeepLearningJob"):
        """
        Initialize the deep learning job with specific parameters.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        """
        self.epochs = epochs
        self.job_name = job_name
    def get_command(self):
        return f"python train_deep_learning.py --epochs {self.epochs}"
    
dlJob = DeepLearningJob(epochs=5,job_name="DL_Job1")
# Train a simple neural network on the MNIST dataset and measure the time taken.
print(f"Job {dlJob.job_name} started.")
        
# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
# Preprocess the data
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

def resize_images(images, new_size=(64, 64)):
    return np.array([tf.image.resize(img, new_size).numpy() for img in images])

start_time = time.time()
with tf.device('/GPU:0'):
    train_images = resize_images(train_images)
    test_images = resize_images(test_images)

# Ensure labels are integers
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)
        
# Build the neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (2, 2), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
        
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
# Train the model and measure time taken
model.fit(train_images, train_labels, epochs=dlJob.epochs, batch_size=64, validation_data=(test_images, test_labels))
end_time = time.time()
        
print(f"Job {dlJob.job_name} completed. Time taken: {end_time - start_time} seconds.")