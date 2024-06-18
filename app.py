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

def main(category1, category2, category3):
    inputs = [category1, category2, category3]
    
    valid_inputs = True
    input_data = []
    for text in inputs:
        if text and not validate_input(text):
            return f"Invalid input: '{text}'"
        if text:
            input_data.append(text)
    
    user_input = ", ".join(input_data) if input_data else ""
    result = generate_quote(user_input)
    return result

categories = ["Word #1", "Word #2", "Word #3"]

input_fields = [
    gr.Textbox(label=f"{category}", placeholder=f"Enter {category.lower()}")
    for category in categories
]
output_field = gr.Textbox(label="Generated Quote", lines=5, interactive=True)

css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body { 
    background: linear-gradient(to right, #6A9113, #141517);
    color: white;
    font-family: 'Poppins', sans-serif;
}
input[type="text"], textarea {
    color: white;
    background-color: #2C3E50;
    border: 1px solid #2980B9;
    font-family: 'Poppins', sans-serif;
}
label {
    font-weight: bold;
    font-size: 16px;
    font-family: 'Poppins', sans-serif;
}
button.primary {
    background-color: #3498DB !important;
    color: white;
    border: none;
    font-size: 16px;
    padding: 10px 20px;
    cursor: pointer;
    font-family: 'Poppins', sans-serif;
}
button.primary:hover {
    background-color: #2980B9 !important;
}
"""

gr.Interface(
    fn=main,
    inputs=input_fields,
    outputs=output_field,
    title="Custom Quote Generator",
    description="Enter up to 3 words to generate a customized quote. Example words: 'Motivation, Love, Life, Work, Health'",
    css=css
).launch(share=True)
