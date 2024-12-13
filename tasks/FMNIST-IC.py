import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

#Using FMNIST because we can easily get it from the tensorflow datasets!
class FMNISTTask:
    def __init__(self, epochs=10, job_name="FMNISTTask"):
        
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python fmnist_task.py --epochs {self.epochs}"

#Choose large number of epochs for stress testing of GPUs!!
fmnist_task = FMNISTTask(epochs=50, job_name="FMNIST_Task1")


print(f"Job {fmnist_task.job_name} started.")

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]  # Add channel dimension
x_test = x_test[..., tf.newaxis]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN model for Fashion MNIST
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Measure time taken as an extra metric to examine!
start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=fmnist_task.epochs,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr],
    batch_size=64
)
end_time = time.time()


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_acc}")

print(f"Job {fmnist_task.job_name} completed. Time taken: {end_time - start_time} seconds.")

model.save("cnn_fashion_mnist.keras")
