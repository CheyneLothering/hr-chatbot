import streamlit as st
from openai import OpenAI
from hr_guardrails import is_hr_related
from rag import retrieve_context
from prompts import SYSTEM_PROMPT, build_prompt
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

st.title("🇨🇦 HR Assistant")

question = st.text_input("Ask an HR question:")

if question:
    if not is_hr_related(question):
        st.warning("I can only help with HR-related questions.")
    else:
        context = retrieve_context(question)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context, question)}
            ],
            temperature=0.3
        )

        st.markdown(response.choices[0].message.content)