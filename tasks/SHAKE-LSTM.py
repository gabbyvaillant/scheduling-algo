import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time

class LSTMTextGeneration:
    def __init__(self, text_file='shakespeare.txt', seq_length=100, batch_size=512, epochs=50, job_name="LSTMTextGenerationTask"):
        self.text_file = text_file
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.job_name = job_name

    def enable_gpu(self):
        #Enable GPU usage for CloudLab!
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU is enabled.")
            except RuntimeError as e:
                print(e)

    def load_data(self):
        # Load and preprocess the text (Shakespeare's text)
        
        text = open(self.text_file, 'r').read()

        #Create character-to-index and index-to-character mappings
        chars = sorted(list(set(text)))
        char_to_index = {char: index for index, char in enumerate(chars)}
        index_to_char = {index: char for index, char in enumerate(chars)}

        #Convert to numerical representation
        text_as_int = np.array([char_to_index[char] for char in text])

        #Prepare input-output pairs
        X, y = [], []
        for i in range(0, len(text_as_int) - self.seq_length, self.seq_length):
            X.append(text_as_int[i:i + self.seq_length])
            y.append(text_as_int[i + self.seq_length])

        X = np.array(X)
        y = np.array(y)

        #Reshape X to [samples, time steps, features] for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1) / float(len(chars))
        
        self.chars = chars
        return X, y

    def build(self):
        #Define LSTM model for text generation
        model = models.Sequential([
            layers.LSTM(256, input_shape=(self.seq_length, 1), return_sequences=True),
            layers.LSTM(256),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.chars), activation='softmax')  # Output is the size of the character set
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    def train_model(self, X, y):
        model = self.build()

        # Measure time before training starts
        start_time = time.time()

        # Train the model
        history = model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        # Measure time after training ends
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")

        return history, model

#Running:
lstm_job = LSTMTextGeneration()
print(f"Job {lstm_job.job_name} started.")

lstm_job.enable_gpu()
X, y = lstm_job.load_data()
history, model = lstm_job.train_model(X, y)

