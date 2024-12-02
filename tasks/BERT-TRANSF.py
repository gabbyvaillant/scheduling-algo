import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

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

# Load IMDB dataset from TensorFlow Datasets
imdb_data, info = tfds.load('imdb', with_info=True, as_supervised=True)
train_data, test_data = imdb_data['train'], imdb_data['test']

# Preprocess the data (tokenization)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_review(text, label):
    encoding = tokenizer(text.numpy().decode('utf-8'), padding='max_length', truncation=True, max_length=512)
    return encoding['input_ids'], label

def tf_encode_review(text, label):
    return tf.py_function(encode_review, [text, label], [tf.int32, tf.int64])

# Prepare datasets
train_dataset = train_data.map(tf_encode_review, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_data.map(tf_encode_review, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

# Load BERT model for sentiment classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("bert_sentiment_analysis_imdb")
