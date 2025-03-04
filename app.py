import streamlit as st
import requests

# Streamlit App
st.title("LLaMA 3.2 Text Generator ")

prompt = st.text_area("Enter your prompt:", "Explain quantum mechanics in short")
max_tokens = st.slider("Max Tokens", 50, 500, 100)

if st.button("Generate"):
    response = requests.post("http://127.0.0.1:8000/generate/", json={"prompt": prompt, "max_new_tokens": max_tokens})
    
    if response.status_code == 200:
        data = response.json()
        st.markdown(f"### Response:\n\n{data['response']}", unsafe_allow_html=True)
    else:
        st.error("Error generating response. Please check the API.")

