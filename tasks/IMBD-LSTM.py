"""

LSTM task:
Stress testing GPU:
    - Large batch size 512
    - 50 epochs to increase training time

To implement in the reinforcement learning code or FCFS code: 
    -RL code you must keep it as is, in it's own .py file
    -Baseline algorithm you must implement it in the same file
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

class LSTMTextClassification:
    def __init__(self, epochs=50, batch_size=512, max_features=20000, maxlen=500, job_name="LSTMTextClassificationStressTest"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_features = max_features
        self.maxlen = maxlen
        self.job_name = job_name

    def enable_gpu(self):
        # Check for GPUs on CloudLab:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU is enabled.")
            except RuntimeError as e:
                print(e)

    def load_data(self):
        """
        Using the IMDB dataset which is reviews of Movies!
        """
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        x_train = pad_sequences(x_train, maxlen=self.maxlen)
        x_test = pad_sequences(x_test, maxlen=self.maxlen)
        return x_train, y_train, x_test, y_test

    def build(self):
        """
        Build the LSTM model for stress testing.
        :return: Compiled model.
        """
        model = models.Sequential([
            layers.Embedding(self.max_features, 128, input_length=self.maxlen),
            layers.LSTM(128, return_sequences=True),  # Increased number of units
            layers.Dropout(0.2),
            layers.LSTM(64),  # Increased number of units
            layers.Dense(128, activation='relu'),  # Additional dense layer
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, x_train, y_train, x_test, y_test):
        model = self.build()

        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        # Measure time before training starts
        start_time = time.time()

        # Train the model
        history = model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test),
            callbacks=[reduce_lr]
        )

        # Measure time after training ends
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")

        return history, model

    def evaluate_model(self, model, x_test, y_test):
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        return test_acc


# Example of running the stress test
lstm_job = LSTMTextClassification(epochs=50, batch_size=512, job_name="LSTM_Text_Classification")
print(f"Job {lstm_job.job_name} started.")

# Enable GPU
lstm_job.enable_gpu()

# Load and preprocess data
x_train, y_train, x_test, y_test = lstm_job.load_data()

# Train the model
history, model = lstm_job.train_model(x_train, y_train, x_test, y_test)

# Evaluate the model
test_acc = lstm_job.evaluate_model(model, x_test, y_test)
print(f"Test accuracy: {test_acc}")

