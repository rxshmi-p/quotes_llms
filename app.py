import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2')
    model = GPT2LMHeadModel.from_pretrained('finetuned_gpt')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def validate_input(text):
    if len(text) > 20:
        return False
    if ' ' in text:
        return False
    if not all(c.isalnum() or c == '-' for c in text):
        return False
    return True

def generate_quote(user_input):
    input_text = user_input + ":"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id, 
        temperature=0.9, 
        top_k=50, 
        do_sample=True
    )
    generated_quote = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_quote

def main(category1, category2, category3, category4, category5):
    inputs = [category1, category2, category3, category4, category5]
    
    valid_inputs = True
    input_data = []
    for text in inputs:
        if text and not validate_input(text):
            return f"Invalid input: '{text}'"
        input_data.append(text if text else '')

    user_input = ", ".join(input_data)
    result = generate_quote(user_input)
    return result

categories = ["category1", "category2", "category3", "category4", "category5"]

input_fields = [gr.Textbox(label=category.capitalize()) for category in categories]
output_field = gr.Textbox()

gr.Interface(
    fn=main,
    inputs=input_fields,
    outputs=output_field,
    title="What are the Top 5 Words to Describe How You're Feeling?",
    description="Example Input for Fine-tuned Model: 'Motivation, Love, Life, Work, Health'"
).launch(share=True)
