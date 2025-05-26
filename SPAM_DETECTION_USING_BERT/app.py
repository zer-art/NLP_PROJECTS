import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

st.set_page_config(page_title="Spam Mail Detector", layout="centered")

# Load the trained BERT model
@st.cache_resource
def loader():
    model = load_model("model/spam_detector.h5"  , custom_objects={"KerasLayer": hub.KerasLayer})
    return  model

model = loader()

# App title and description
st.title("📧 Spam Mail Detector with BERT")
st.markdown("Detect whether a message is **Spam** or **Not Spam** using a fine-tuned BERT model.")

# Input method
st.subheader("Enter email message")
text_input = st.text_area("✍️ Type your message below:", height=200, placeholder="Write or paste your message here...")

st.subheader("Or upload a `.txt` file")
uploaded_file = st.file_uploader("📂 Upload text file", type=["txt"])

# Combine input logic
def get_text():
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            return content
        except:
            st.error("Error reading the file. Please ensure it's a valid .txt file encoded in UTF-8.")
            return None
    elif text_input.strip():
        return text_input
    else:
        return None

# Detect button
if st.button("🔍 Detect"):
    input_text = get_text()

    if input_text:
        prediction = model.predict([input_text])[0][0]
        label = "🚫 Spam" if prediction > 0.5 else "✅ Not Spam"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Result: **{label}**")
        st.progress(int(confidence * 100))
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.warning("Please enter text or upload a .txt file first.")

# Optional info
with st.expander("ℹ️ About this App"):
    st.write("""
    This app uses a fine-tuned BERT model from TensorFlow Hub to analyze messages and detect if they are spam.
    
    - You can either type a message or upload a `.txt` file.
    - BERT understands the context of the language for accurate detection.
    """)
