import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset

def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return df

data = load_data()

# Load the embedding model

def load_embedding_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

model = load_embedding_model()

st.title("Heart, Lung, and Blood Health Q&A")

# User question input
user_question = st.text_input("Enter your question:")

def find_best_answer(user_question, data, model):
    user_question_embedding = model.encode([user_question])[0]
    similarities = cosine_similarity([user_question_embedding], data['Question_Embedding'].tolist())[0]
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    similarity_threshold = 0.7

    if best_match_score >= similarity_threshold:
        return data['Answer'].iloc[best_match_idx], best_match_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", best_match_score

if st.button('Get Answer'):
    if user_question:
        answer, score = find_best_answer(user_question, data, model)
        st.write(f"Answer: {answer} (Similarity Score: {score:.2f})")
    else:
        st.write("Please enter a question.")
