import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Load Reuters dataset
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)

# Preprocess data
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define Transformer model
model = models.Sequential([
    layers.Embedding(max_features, 128),
    layers.MultiHeadAttention(num_heads=4, key_dim=128),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(46, activation='softmax')  # 46 categories in Reuters dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("transformer_reuters.keras")
