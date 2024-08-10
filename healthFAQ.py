import streamlit as st
import openai
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set this in your Streamlit environment

st.title("Heart, Lung, and Blood Health Q&A")

# User question input
user_question = st.text_input("Enter your question:")

def get_openai_answer(question):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use a different engine if preferred
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
