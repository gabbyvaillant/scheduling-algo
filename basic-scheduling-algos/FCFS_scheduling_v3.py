#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

class FCFS:
    def __init__(self):
        self.job_queue = []
        self.execution_log = []
    
    def submit_job(self, job):
        """
        Submit a job to the FCFS queue.
        :param job: Job to be submitted.
        """
        self.job_queue.append(job)
        print(f"Job {job.job_name} submitted.")

    def render_gantt_chart(self, filename="FCFS_gantt_chart.png"):
        """Generate a Gantt chart for the logged execution times."""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab20.colors

        for idx, log in enumerate(self.execution_log):
            job_name = log["job_name"]
            start_time = log["start_time"]
            end_time = log["end_time"]
            ax.barh(job_name, end_time - start_time, left=start_time, color=colors[idx % len(colors)])

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Jobs")
        ax.set_title("FCFS Scheduling Gantt Chart")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename) # save the chart to a file
        print(f"Gantt chart saved as {filename}")
    
    def run(self):
        """
        Execute jobs in the order they were submitted.
        """
        current_time = 0
        while self.job_queue:
            current_job = self.job_queue.pop(0)
            start_time = current_time
            print(f"Running job: {current_job.job_name}")
            job_runtime = current_job.run()
            end_time = start_time + job_runtime
            # Log execution details
            self.execution_log.append({
                "job_name": current_job.job_name,
                "start_time": start_time,
                "end_time": end_time,
            })
            # Update current time
            current_time = end_time
        print("All jobs have been completed.")

class MatrixMultiplicationJob:
    def __init__(self, matrix_size=5000, job_name="MatrixMultiplication"):
        """
        Initialize the matrix multiplication job with a specific matrix size.
        :param matrix_size: The size of the square matrices to multiply.
        :param job_name: Name of the job for identification.
        """
        self.matrix_size = matrix_size
        self.job_name = job_name
    
    def run(self):
        """
        Perform matrix multiplication and measure the time taken.
        """
        print(f"Job {self.job_name} started.")
        
        # Generate two random matrices and perform matrix multiplication
        start_time = time.time()
        with tf.device('/GPU:0'):
            A = tf.random.uniform((self.matrix_size, self.matrix_size))
            B = tf.random.uniform((self.matrix_size, self.matrix_size))
            result = tf.matmul(A, B)
        
        # measure the time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Job {self.job_name} completed. Time taken: {end_time - start_time} seconds.")
        return execution_time
    

class DeepLearningJob:
    def __init__(self, epochs=5, job_name="DeepLearningJob"):
        """
        Initialize the deep learning job with specific parameters.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        """
        self.epochs = epochs
        self.job_name = job_name
    
    def run(self):
        """
        Train a simple neural network on the CIFAR-10 dataset and measure the time taken.
        """
        print(f"Job {self.job_name} started.")
        
        # Load the CIFAR-10 dataset
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        # Preprocess the data
        train_images = train_images.astype("float32") / 255
        test_images = test_images.astype("float32") / 255

        def resize_images(images, new_size=(64, 64)):
            return np.array([tf.image.resize(img, new_size).numpy() for img in images])

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
        start_time = time.time()
        model.fit(train_images, train_labels, epochs=self.epochs, batch_size=64, validation_data=(test_images, test_labels))
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Job {self.job_name} completed. Time taken: {end_time - start_time} seconds.")
        return execution_time

class XGBoostJob:
    def __init__(self, n_estimators=100, max_depth=6, job_name="XGBoostJob"):
        """
        Initialize the XGBoost job with specific parameters.
        :param n_estimators: Number of boosting rounds.
        :param max_depth: Maximum depth of each tree.
        :param job_name: Name of the job for identification.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.job_name = job_name

    def run(self):
        """
        Train an XGBoost model on synthetic data and measure the time taken.
        """
        print(f"Job {self.job_name} started.")

        # Generate synthetic dataset
        X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to DMatrix and load onto the GPU
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Set up XGBoost parameters for GPU usage
        params = {
            "objective": "binary:logistic",
            "max_depth": self.max_depth,
            "tree_method": "hist",
            "device": "cuda"
        }

        # Measure time taken for training
        start_time = time.time()
        model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        end_time = time.time()
        execution_time = end_time - start_time
        # Test model accuracy
        predictions = (model.predict(dtest) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Job {self.job_name} completed. Time taken: {end_time - start_time} seconds.")
        print(f"Accuracy of {self.job_name}: {accuracy}")
        return execution_time

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

    def run(self):
        """
        Train an LSTM model on the synthetic dataset and measure the time taken.
        """
        print(f"Job {self.job_name} started.")

        # Generate dataset
        X, y = self.generate_data()

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1)),
            Dense(1)
        ])

        # Compile model
        model.compile(optimizer='adam', loss='mse')

        # Train model and measure time taken
        start_time = time.time()
        model.fit(X, y, epochs=self.epochs, batch_size=64, verbose=1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Job {self.job_name} completed. Time taken: {end_time - start_time} seconds.")
        return execution_time

# Example usage with matrix multiplication and ML jobs
if __name__ == "__main__":
    fcfs_scheduler = FCFS()
    
    # Create and submit jobs
    job1 = MatrixMultiplicationJob(matrix_size=30000, job_name="MatrixMultiplication_Job1")
    job2 = DeepLearningJob(epochs=5,job_name="DL_Job1")
    job3 = XGBoostJob(n_estimators=1200, max_depth=30, job_name="XGBoost_Job1")
    job4 = LSTMJob(sequence_length=200, epochs=10, job_name="LSTM_Job1")
    
    fcfs_scheduler.submit_job(job1)
    fcfs_scheduler.submit_job(job2)
    fcfs_scheduler.submit_job(job3)
    fcfs_scheduler.submit_job(job4)
    
    # Run the scheduler
    fcfs_scheduler.run()
    fcfs_scheduler.render_gantt_chart()


# In[ ]:
