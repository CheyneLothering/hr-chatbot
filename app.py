import re
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from hr_guardrails import is_hr_related
from prompts import SYSTEM_PROMPT, build_prompt
from rag import retrieve_context

load_dotenv()
client = OpenAI()

MAX_TURNS = 6  # keep only the last few user/assistant messages for short-term memory


def anonymize(text: str) -> str:
    """Mask common personal data patterns before storing anything locally."""
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[redacted email]", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted phone]", text)
    return text


def add_to_history(role: str, content: str) -> None:
    """Store only short-term, HR-safe snippets."""
    sanitized = anonymize(content)
    st.session_state.chat_history.append({"role": role, "content": sanitized})
    st.session_state.chat_history = st.session_state.chat_history[-MAX_TURNS:]


if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

st.title("HR Assistant")

# Display existing history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR question")

if question:
    user_message = anonymize(question)
    with st.chat_message("user"):
        st.markdown(user_message)

    hr_ok, _reason = is_hr_related(question, client)
    if not hr_ok:
        with st.chat_message("assistant"):
            st.markdown("I can only help with HR-related questions.")
    else:
        add_to_history("user", user_message)

        recent_history: List[Dict[str, str]] = st.session_state.chat_history[:-1]
        context = retrieve_context(question)

        message_window: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Relevant HR policy context:\n{context}"},
            *recent_history,
            {"role": "user", "content": build_prompt(context, user_message)},
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=message_window,
                temperature=0.3,
            )
            assistant_reply = response.choices[0].message.content
        except Exception as exc:
            assistant_reply = (
                "I'm having trouble responding right now. Please try again in a moment."
            )
            st.error(f"Request error: {exc}")

        add_to_history("assistant", assistant_reply)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)