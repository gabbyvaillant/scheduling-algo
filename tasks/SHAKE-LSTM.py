import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Enable GPU usage
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is enabled.")
    except RuntimeError as e:
        print(e)

# Load and preprocess the text (Shakespeare's works)
text = open('shakespeare.txt', 'r').read()

# Create character-to-index and index-to-character mappings
chars = sorted(list(set(text)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Convert text to numerical representation
text_as_int = np.array([char_to_index[char] for char in text])

# Prepare input-output pairs
seq_length = 100
X, y = [], []
for i in range(0, len(text_as_int) - seq_length, seq_length):
    X.append(text_as_int[i:i+seq_length])
    y.append(text_as_int[i+seq_length])

X = np.array(X)
y = np.array(y)

# Reshape X to [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1) / float(len(chars))

# Define LSTM model for text generation
model = models.Sequential([
    layers.LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True),
    layers.LSTM(256),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(chars), activation='softmax')  # Output is the size of the character set
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
history = model.fit(
    X, y,
    epochs=50,
    batch_size=64
)

# Save the model
model.save("lstm_text_generation_shakespeare.keras")
