import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
class LSTMJob:
    def __init__(self, sequence_length=50, epochs=10, job_name="LSTMJob"):
        """
        Initialize the LSTM job with specific parameters.
        :param sequence_length: Length of input sequences.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.job_name = job_name

    def get_command(self):
        return f"python train_lstm.py --sequence_length {self.sequence_length} --epochs {self.epochs}"
    
    def generate_data(self, num_samples=10000):
        """
        Generate a synthetic dataset of sine wave sequences.
        :param num_samples: Number of sequences to generate.
        :return: Tuple of (X,y) where X is the input sequence and y is the target sequence.
        """
        X = []
        y = []
        for i in range(num_samples):
            start = np.random.rand() * 2 * np.pi
            sequence = np.sin(np.linspace(start, start + 10 * np.pi, self.sequence_length + 1))
            X.append(sequence[:-1].reshape(self.sequence_length, 1))
            y.append(sequence[1:])
        return np.array(X), np.array(y)

lstmJob = LSTMJob(sequence_length=200, epochs=10, job_name="LSTM_Job1")
# Train an LSTM model on the synthetic dataset and measure the time taken.
print(f"Job {lstmJob.job_name} started.")

# Generate dataset
X, y = lstmJob.generate_data()

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(lstmJob.sequence_length, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model and measure time taken
start_time = time.time()
model.fit(X, y, epochs=lstmJob.epochs, batch_size=64, verbose=1)
end_time = time.time()
print(f"Job {lstmJob.job_name} completed. Time taken: {end_time - start_time} seconds.")