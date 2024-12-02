import tensorflow as tf
from transformers import MarianMTModel, MarianTokenizer

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

# Load MarianMT model and tokenizer for translation (English to French)
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate a sentence from English to French
sentence = "Hello, how are you?"
inputs = tokenizer(sentence, return_tensors="pt", padding=True)

# Generate translation
translated = model.generate(**inputs)
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(f"Original: {sentence}")
print(f"Translated: {translated_text}")

# Save the model
model.save("marianmt_translation_en_fr")
