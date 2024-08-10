import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    # Convert the 'Question_Embedding' column from strings to numpy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return df

data = load_data()

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

model = load_embedding_model()

st.title("Heart, Lung, and Blood Health Q&A")

# User question input
user_question = st.text_input("Enter your question:")

# Button to trigger the answer search
if st.button('Get Answer'):
    if user_question:
        # To be implemented: Question answering logic
        pass
    else:
        st.write("Please enter a question.")

def find_best_answer(user_question, data, model):
    # Generate the embedding for the user's question
    user_question_embedding = model.encode([user_question])[0]

    # Calculate the cosine similarity between user_question_embedding and all question embeddings in the dataset
    similarities = cosine_similarity([user_question_embedding], data['Question_Embedding'].tolist())[0]

    # Find the index of the highest similarity score
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]

    # Define a similarity threshold (this value can be adjusted)
    similarity_threshold = 0.7

    if best_match_score >= similarity_threshold:
        # Return the answer corresponding to the best match
        return data['Answer'].iloc[best_match_idx], best_match_score
    else:
        # Return a message indicating no relevant answer was found
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", best_match_score

# Implement the Question Answering Logic
if st.button('Get Answer'):
    if user_question:
        answer, score = find_best_answer(user_question, data, model)
        st.write(f"Answer: {answer} (Similarity Score: {score:.2f})")
    else:
        st.write("Please enter a question.")
