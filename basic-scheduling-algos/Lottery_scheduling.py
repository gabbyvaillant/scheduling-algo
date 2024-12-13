import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import time
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
class MatrixMultiplicationJob:
    def __init__(self, matrix_size=5000, job_name="MatrixMultiplication", tickets=1):
        """
        Initialize the matrix multiplication job with a specific matrix size.
        :param matrix_size: The size of the square matrices to multiply.
        :param job_name: Name of the job for identification.
        :param tickets: Number of tickets (priority) of the the job
        """
        self.matrix_size = matrix_size
        self.job_name = job_name
        self.tickets = tickets
    
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
    def __init__(self, epochs=5, job_name="DeepLearningJob", tickets=2):
        """
        Initialize the deep learning job with specific parameters.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        :param tickets: Number of tickets (priority) of the the job
        """
        self.epochs = epochs
        self.job_name = job_name
        self.tickets = tickets
    
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
    def __init__(self, n_estimators=100, max_depth=6, job_name="XGBoostJob", tickets=3):
        """
        Initialize the XGBoost job with specific parameters.
        :param n_estimators: Number of boosting rounds.
        :param max_depth: Maximum depth of each tree.
        :param job_name: Name of the job for identification.
        :param tickets: Number of tickets (priority) of the the job
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.job_name = job_name
        self.tickets = tickets

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
    def __init__(self, sequence_length=50, epochs=10, job_name="LSTMJob", tickets=4):
        """
        Initialize the LSTM job with specific parameters.
        :param sequence_length: Length of input sequences.
        :param epochs: Number of epochs for training.
        :param job_name: Name of the job for identification.
        :param tickets: Number of tickets (priority) of the the job
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.job_name = job_name
        self.tickets = tickets
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

# Lottery Scheduler
class LotteryScheduler:
    def __init__(self):
        self.task_pool = []
        self.execution_log = []

    def add_task(self, task):
        """Add a task to the task pool."""
        self.task_pool.append(task)

    def run(self, iterations=10):
        """Run the lottery scheduling for a certain number of iterations."""
        current_time = 0
        for _ in range(iterations):
            if not self.task_pool:
                print("No tasks left to schedule.")
                break
            
            # Create a lottery pool
            ticket_pool = []
            for task in self.task_pool:
                ticket_pool.extend([task] * task.tickets)

            # Pick a random task based on tickets
            winner_task = random.choice(ticket_pool)
            print(f"\nSelected Task: {winner_task.job_name}")
            
            start_time = current_time
            # Run the selected task
            job_runtime = winner_task.run()
            end_time = start_time + job_runtime
            # Log the execution details
            self.execution_log.append({
                "job_name": winner_task.job_name,
                "start_time": start_time,
                "end_time": end_time,
            })
            # Update current time
            current_time = end_time
            # Optionally, remove the task after execution if it's a one-time task
            self.task_pool = [task for task in self.task_pool if task != winner_task]

    def render_gantt_chart(self, filename="LS_gantt_chart.png"):
        # Create a Gantt chart to show job schedules
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab20.colors
        for idx, log in enumerate(self.execution_log):
            job_name = log["job_name"]
            start_time = log["start_time"]
            end_time = log["end_time"]
            ax.barh(job_name, end_time - start_time, left=start_time, color=colors[idx % len(colors)])

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Jobs")
        ax.set_title("Lottery Scheduling Gantt Chart")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename) # Save the chart to a file
        print(f"Gantt chart saved as {filename}")


# Example Usage
if __name__ == "__main__":
    # Create tasks with different ticket counts (priority)
    task1 = MatrixMultiplicationJob(30000,"MatrixMult", tickets=5)
    task2 = DeepLearningJob(5, "DLJob", tickets=10)
    task3 = XGBoostJob(1200, 30, "XGBoostJob", tickets=7)
    task4 = LSTMJob(200, 10, "LSTMJob", tickets=15)

    # Initialize scheduler and add tasks
    scheduler = LotteryScheduler()
    scheduler.add_task(task1)
    scheduler.add_task(task2)
    scheduler.add_task(task3)
    scheduler.add_task(task4)

    # Run the scheduler
    print("Starting Lottery Scheduler...")
    scheduler.run(iterations=10)
    scheduler.render_gantt_chart()
