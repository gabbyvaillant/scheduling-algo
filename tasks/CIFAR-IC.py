import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

class CIFAR10Task:
    def __init__(self, epochs=50, job_name="CIFAR10Task"):
        """
        Initialize the CIFAR-10 task with specific parameters.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        """
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python cifar10_task.py --epochs {self.epochs}"

# Initialize CIFAR-10 task
cifar10_task = CIFAR10Task(epochs=50, job_name="CIFAR10_Task1")

# Start job
print(f"Job {cifar10_task.job_name} started.")

# See the available GPUs on CloudLab
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is enabled.")
    except RuntimeError as e:
        print(e)

# CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Define a deeper CNN model with Batch Normalization and Dropout
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model and measure time taken
start_time = time.time()
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=cifar10_task.epochs,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr]
)
end_time = time.time()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("cnn_cifar10_improved_task.keras")

print(f"Job {cifar10_task.job_name} completed. Time taken: {end_time - start_time} seconds.")
