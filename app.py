import re
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from hr_guardrails import is_hr_related
from prompts import SYSTEM_PROMPT, build_prompt
from rag import retrieve_context

st.set_page_config(page_title="HR Assistant", page_icon="briefcase", layout="centered")

load_dotenv()
client = OpenAI()

MAX_TURNS = 6  # keep only the last few user/assistant messages for short-term memory


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Manrope:wght@400;600;700&display=swap');

:root {
  --ink: #0f172a;
  --muted: #475569;
  --accent: #0f766e;
  --accent-2: #f59e0b;
  --panel: rgba(255, 255, 255, 0.92);
  --panel-border: rgba(15, 23, 42, 0.14);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #dbeafe 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #fef3c7 0%, transparent 55%),
    linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  color: var(--ink);
}

.main .block-container {
  padding-top: 2rem;
  max-width: 900px;
}

div[data-testid="stVerticalBlock"]:has(.hero-marker) {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 18px;
  padding: 1.5rem 1.75rem;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  backdrop-filter: blur(6px);
  animation: fadeIn 600ms ease-out;
  margin-bottom: 1.25rem;
}

.hero-top {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  align-items: center;
  justify-content: space-between;
}

.hero .eyebrow {
  font-family: 'Manrope', sans-serif;
  font-size: 0.85rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
  font-weight: 700;
}

.hero h1 {
  font-family: 'Fraunces', serif;
  font-size: 2.2rem;
  margin: 0 0 0.4rem 0;
  letter-spacing: -0.02em;
}

.hero p {
  font-family: 'Manrope', sans-serif;
  color: var(--muted);
  margin: 0.2rem 0 0.8rem 0;
  font-size: 1rem;
}

.hero-avatar {
  width: 88px;
  height: 88px;
  border-radius: 20px;
  background: #ffffff;
  border: 1px solid var(--panel-border);
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.12);
  display: grid;
  place-items: center;
}

.hero-avatar svg {
  width: 64px;
  height: 64px;
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}

div[data-testid="stButton"] > button {
  font-family: 'Manrope', sans-serif;
  font-size: 0.7rem;
  padding: 0.22rem 0.5rem;
  border-radius: 999px;
  background: rgba(15, 118, 110, 0.08);
  border: 1px solid rgba(15, 118, 110, 0.2);
  color: #0f766e;
  box-shadow: none;
}

div[data-testid="stButton"] > button:hover {
  background: rgba(15, 118, 110, 0.16);
  border-color: rgba(15, 118, 110, 0.35);
}

div[data-testid="stButton"] > button[kind="primary"] {
  background: rgba(245, 158, 11, 0.18);
  border-color: rgba(245, 158, 11, 0.45);
  color: #b45309;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: rgba(245, 158, 11, 0.28);
  border-color: rgba(245, 158, 11, 0.6);
}

div[data-testid="stChatMessageContent"] {
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.18);
  color: #0f172a;
  border-radius: 16px;
  padding: 0.75rem 1rem;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

div[data-testid="stChatMessageContent"] p,
div[data-testid="stChatMessageContent"] span,
div[data-testid="stChatMessageContent"] li {
  color: inherit;
}

div[data-testid="stChatInput"] {
  background: #ffffff;
  border-radius: 16px;
  border: 1px solid rgba(15, 23, 42, 0.14);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)


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

pill_labels = [
    ("Benefits and leave", "secondary"),
    ("Payroll and compensation", "secondary"),
    ("Policies and compliance", "secondary"),
    ("Manager guidance", "primary"),
    ("Employee relations", "primary"),
]

selected_prompt = None
hero_container = st.container()
with hero_container:
    st.markdown('<div class="hero-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="hero-top">
  <div>
    <div class="eyebrow">HR Assistant</div>
    <h1>Workplace answers with a human touch</h1>
    <p>
      Ask about benefits, leave, payroll, policies, onboarding, and workplace issues.
      I can only answer HR-related topics and I do not store personal data long-term.
    </p>
  </div>
  <div class="hero-avatar" aria-hidden="true">
    <svg viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="18" y="22" width="60" height="52" rx="14" fill="#0f172a"/>
      <rect x="24" y="30" width="48" height="34" rx="10" fill="#e2e8f0"/>
      <circle cx="38" cy="47" r="6" fill="#0f766e"/>
      <circle cx="58" cy="47" r="6" fill="#0f766e"/>
      <rect x="36" y="58" width="24" height="6" rx="3" fill="#0f172a"/>
      <rect x="44" y="12" width="8" height="10" rx="4" fill="#f59e0b"/>
      <circle cx="48" cy="10" r="6" fill="#f59e0b"/>
    </svg>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pill-row">', unsafe_allow_html=True)
    cols = st.columns(len(pill_labels))
    for idx, (label, style) in enumerate(pill_labels):
        if cols[idx].button(label, key=f"pill-{idx}", type=style):
            selected_prompt = label
    st.markdown("</div>", unsafe_allow_html=True)

# Display existing history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR question")
input_text = selected_prompt or question

if input_text:
    user_message = anonymize(input_text)
    with st.chat_message("user"):
        st.markdown(user_message)

    hr_ok, _reason = is_hr_related(input_text, client)
    if not hr_ok:
        with st.chat_message("assistant"):
            st.markdown("I can only help with HR-related questions.")
    else:
        add_to_history("user", user_message)

        recent_history: List[Dict[str, str]] = st.session_state.chat_history[:-1]
        context = retrieve_context(input_text)

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
