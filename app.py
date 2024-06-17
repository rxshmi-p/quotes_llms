import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./models')
model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='./models')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    return tokenizer, model

# Streamlit UI
st.title("GPT-2 Text Generation")

# Display loading messages outside the cached function
if 'model_loaded' not in st.session_state:
    st.info("Loading model...")
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
    st.session_state['model_loaded'] = True
else:
    tokenizer, model = load_model()

def generate_text(prompt, max_length=50):
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=max_length)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        return f"Error generating text: {e}"

prompt = st.text_input("Enter a prompt:")
max_length = st.slider("Max length of generated text", 10, 100, 50)

if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        result = generate_text(prompt, max_length)
        st.write(result)
