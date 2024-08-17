import pandas as pd
import streamlit as st
import openai
import os

# Set up OpenAI API key
openai.api_key = st.secrets["mykey"]

def load_data():
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    df['Question_Embedding'] = df['Question_Embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return df

data = load_data()

st.title("Heart, Lung, and Blood Health Q&A")

# User question input
user_question = st.text_input("Enter your question:")

def get_openai_answer(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if st.button('Get Answer'):
    if user_question:
        answer = get_openai_answer(user_question)
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question.")
