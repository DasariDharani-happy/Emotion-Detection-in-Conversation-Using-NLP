import streamlit as st
import numpy as np
from transformers import pipeline

# Title and description
st.title("Emotion Detection in Conversation")
st.write("This app detects the **emotion** from a given conversational sentence using NLP techniques.")

# Load emotion classification pipeline (you can replace this with your own fine-tuned model)
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_model = load_emotion_model()

# Input from user
user_input = st.text_area("Enter a conversation or dialogue:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        results = emotion_model(user_input)[0]
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_emotion = sorted_results[0]

        st.subheader("Predicted Emotion:")
        st.markdown(f"**{top_emotion['label']}** with confidence {top_emotion['score']:.2f}")

        st.subheader("All Emotion Scores:")
        for res in sorted_results:
            st.write(f"{res['label']}: {res['score']:.2f}")
