import time
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()

# Load the tokenizer and model from local directory
tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')
model = GPT2LMHeadModel.from_pretrained('./models/gpt2')
model.eval()

end_time = time.time()

print(f"Model loaded in {end_time - start_time:.2f} seconds")
