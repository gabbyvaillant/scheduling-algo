import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

class MNISTTask:
    def __init__(self, epochs=10, job_name="MNISTTask"):
        """
        Initialize the MNIST task with specific parameters.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        """
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python mnist_task.py --epochs {self.epochs}"

# Initialize MNIST task
mnist_task = MNISTTask(epochs=10, job_name="MNIST_Task1")

# Start job
print(f"Job {mnist_task.job_name} started.")

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

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]  # Add channel dimension
x_test = x_test[..., tf.newaxis]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define CNN model for MNIST
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

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model and measure time taken
start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=mnist_task.epochs,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr],
    batch_size=64
)
end_time = time.time()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("cnn_mnist_task.keras")

print(f"Job {mnist_task.job_name} completed. Time taken: {end_time - start_time} seconds.")

print(f"Test accuracy: {test_acc}")

# Save the model
model.save("cnn_mnist.keras")
