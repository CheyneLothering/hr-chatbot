import re
import io
import base64
import os
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

from hr_guardrails import is_hr_related
from prompts import SYSTEM_PROMPT, build_prompt
from rag import retrieve_context

st.set_page_config(page_title="HR Assistant", page_icon="briefcase", layout="centered")

load_dotenv()
client = OpenAI()

MAX_TURNS = 6  # keep only the last few user/assistant messages for short-term memory
COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "components", "voice_chat_input")
voice_chat_input = components.declare_component(
    "voice_chat_input",
    path=COMPONENT_PATH,
)


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

div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] {
  margin-right: 0.4rem;
}

div[data-testid="stButton"] > button {
  font-family: 'Manrope', sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1.15;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.14), rgba(15, 118, 110, 0.04));
  border: 1px solid rgba(15, 118, 110, 0.35);
  color: #0b5f5a;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
  transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
  width: 100%;
  white-space: normal;
  text-align: center;
}

div[data-testid="stButton"] > button:hover {
  transform: translateY(-1px);
  background: linear-gradient(180deg, rgba(15, 118, 110, 0.24), rgba(15, 118, 110, 0.08));
  border-color: rgba(15, 118, 110, 0.5);
  box-shadow: 0 10px 18px rgba(15, 23, 42, 0.12);
}

div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(180deg, rgba(245, 158, 11, 0.3), rgba(245, 158, 11, 0.12));
  border-color: rgba(245, 158, 11, 0.6);
  color: #92400e;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: linear-gradient(180deg, rgba(245, 158, 11, 0.42), rgba(245, 158, 11, 0.2));
  border-color: rgba(245, 158, 11, 0.75);
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
if "last_input_id" not in st.session_state:
    st.session_state.last_input_id = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

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
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(pill_labels))
    for idx, (label, style) in enumerate(pill_labels):
        if cols[idx].button(label, key=f"pill-{idx}", type=style):
            selected_prompt = label

# Display existing history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

voice_reply = True
payload = voice_chat_input(
    placeholder="Ask an HR question",
    key="voice_chat_input",
    height=70,
)

input_text = None
if selected_prompt:
    input_text = selected_prompt
elif payload and payload.get("id") != st.session_state.last_input_id:
    st.session_state.last_input_id = payload.get("id")
    kind = payload.get("kind")
    if kind == "text":
        input_text = (payload.get("text") or "").strip()
    elif kind == "audio":
        audio_b64 = payload.get("data") or ""
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "voice.webm"
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=audio_file,
                )
                st.session_state.last_transcript = transcription.text.strip()
                input_text = st.session_state.last_transcript
                if input_text:
                    st.caption(f"Transcribed: {input_text}")
            except Exception as exc:
                st.session_state.last_transcript = ""
                st.error(f"Transcription error: {exc}")

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
            if voice_reply:
                try:
                    speech = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice="sage",
                        input=assistant_reply,
                    )
                    st.audio(speech.content, format="audio/mp3")
                except Exception as exc:
                    st.error(f"Voice reply error: {exc}")
