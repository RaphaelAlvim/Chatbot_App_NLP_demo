import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import os

# Path to save the model
MODEL_PATH = "./models/gpt2-small"

# Ensure the model directory exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Download model if not already available
def download_model(model_name, path):
    # Download model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=path)
    return model, tokenizer

# Load the GPT-2 Small model and tokenizer
@st.cache_resource
def load_model():
    # Check if model files exist
    if not os.listdir(MODEL_PATH):
        download_model("gpt2", MODEL_PATH)
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

gpt2_pipeline = load_model()

st.title("RM DataScience Chatbot Demo - GPT-2 Small")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt from user
if prompt := st.chat_input("How can I help you? / Como posso ajudar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from GPT-2 Small
    with st.chat_message("assistant"):
        response = gpt2_pipeline(prompt, max_length=150, num_return_sequences=1)
        response_text = response[0]['generated_text']
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})