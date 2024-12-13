import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

class GPT2TextGeneration:
    def __init__(self, model_name='gpt2', max_length=200, temperature=0.7):
        """
        Initialize the GPT2 text generation model and tokenizer.
        :param model_name: Name of the GPT2 model to use (default is 'gpt2').
        :param max_length: Maximum length of the generated text (default is 200).
        :param temperature: Sampling temperature to control randomness (default is 0.7).
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = TFGPT2LMHeadModel.from_pretrained(model_name)
        
        # Enable GPU usage
        print("Available GPUs:", tf.config.list_physical_devices('GPU'))
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU is enabled.")
            except RuntimeError as e:
                print(e)

    def generate_text(self, prompt):
        """
        Generate text based on the input prompt.
        :param prompt: The text prompt to start text generation from.
        :return: The generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='tf')
        generated_text = self.model.generate(input_ids, 
                                            max_length=self.max_length, 
                                            num_return_sequences=1, 
                                            temperature=self.temperature)
        
        generated_text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return generated_text

    def save_model(self, save_path="gpt2_text_generation"):
        """
        Save the trained model to a specified path.
        :param save_path: Path where the model will be saved (default is 'gpt2_text_generation').
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

# Example usage:
if __name__ == "__main__":
    # Initialize the GPT2TextGeneration class
    gpt2_gen = GPT2TextGeneration()

    # Generate text with a prompt
    prompt = "Once upon a time"
    generated_text = gpt2_gen.generate_text(prompt)
    print("Generated Text:", generated_text)

    # Save the model
    gpt2_gen.save_model()


