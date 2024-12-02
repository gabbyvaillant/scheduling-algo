import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

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

# Load and preprocess stock data (example: Apple stock)
data = pd.read_csv('AAPL.csv', date_parser=True, index_col='Date')
data = data[['Close']]  # Using only 'Close' price for prediction

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for time series forecasting
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape input to be compatible with LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM model for stock price prediction
model = models.Sequential([
    layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    layers.LSTM(100, return_sequences=False),
    layers.Dense(50),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=64
)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {test_loss}")

# Save the model
model.save("lstm_stock_prediction.keras")
