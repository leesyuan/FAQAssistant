import openai
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = st.secrets["mykey"]

def generate_embedding(text):
  response = openai.Embedding.create(
    input=[text],
    model="text-embedding-ada-002"
  )
  return response['data'][0]['embedding'] 


# Load the dataset
df = pd.read_csv("qa_dataset_with_embeddings.csv")

def generate_answer(user_question):
  # Generate embedding for user question
  user_embedding = generate_embedding(user_question)

  # Convert question embeddings to NumPy array
  question_embeddings = np.array(df['Question_Embedding'].tolist())

  # Calculate cosine similarity
  similarities = cosine_similarity(np.array(user_embedding).reshape(1, -1), question_embeddings)

  # Find the most similar question
  most_similar_index = np.argmax(similarities)
  similarity_score = similarities[0][most_similar_index]

  # Set a similarity threshold (you can adjust this)
  threshold = 0.7

  if similarity_score > threshold:
    answer = df['Answer'][most_similar_index]
    return answer
  else:
    return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

def main():
  st.title("Health Question Answering")

  user_question = st.text_input("Ask your health question")
  if st.button("Submit"):
    answer = generate_answer(user_question)
    st.text_area("Answer:", value=answer)

if __name__ == "__main__":
  main()
