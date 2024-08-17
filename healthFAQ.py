import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = st.secrets["mykey"]

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('qa_dataset_with_embeddings.csv')

# Convert the Question_Embedding from string to numpy array
df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(eval(x)))

# Function to generate embeddings using OpenAI
def generate_embedding_openai(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

# Function to find the most similar question and return the corresponding answer
def find_most_similar_answer(user_question, df):
    # Generate embedding for the user's question using OpenAI
    user_embedding = generate_embedding_openai(user_question)

    # Calculate cosine similarities between user question and stored questions
    stored_embeddings = np.stack(df['Question_Embedding'].values)
    similarities = cosine_similarity([user_embedding], stored_embeddings)

    # Find the index of the most similar question
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[0][most_similar_idx]

    # Define a threshold for relevance
    threshold = 0.7

    # Return the corresponding answer if similarity is above the threshold
    if similarity_score > threshold:
        return df['Answer'].iloc[most_similar_idx], similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", similarity_score

# Streamlit Interface
st.title("Health Q&A System")

# User input
user_question = st.text_input("Ask a question about heart, lung, or blood-related health topics:")

# Button to trigger search
if st.button("Get Answer"):
    if user_question.strip() != "":
        answer, similarity = find_most_similar_answer(user_question, df)
        st.write(f"**Answer:** {answer}")
        st.write(f"**Similarity Score:** {similarity:.2f}")
    else:
        st.write("Please enter a question.")

# Clear button (resets the text input)
if st.button("Clear"):
    st.text_input("Ask a question about heart, lung, or blood-related health topics:", value="", key="new")

