import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

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

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# Encode a prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='tf')

# Generate text
generated_text = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.7)

# Decode and print the generated text
generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(generated_text)

# Save the model
model.save("gpt2_text_generation")


