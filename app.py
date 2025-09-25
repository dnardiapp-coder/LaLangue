import os, io, re, json
import streamlit as st
from pydub import AudioSegment
from pydub.generators import Silence
from typing import List, Dictt
from pydub.generators import 
from openai import OpenAI

# --------- CONFIG ---------
OPENAI_MODEL = "gpt-5"   # or "gpt-5"
TTS_VOICE_COACH = "alloy"
TTS_VOICE_NATIVE = "verse"
MAX_INPUT_CHARS = 8000

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI setup
st.set_page_config(page_title="LaLangue", page_icon="🎧", layout="centered")
st.title("🎧 LaLangue – Audio Lesson Generator")

with st.sidebar:
    st.header("Lesson Settings")
    lang = st.selectbox("Target language", ["Spanish", "French", "Italian", "Portuguese", "Mandarin Chinese", "Russian"])
    level = st.selectbox("Level (CEFR)", ["A1", "A2", "B1"])
    duration_min = st.slider("Lesson duration target (minutes)", 8, 30, 15)
    n_items = st.slider("New items to introduce", 4, 10, 8)
    seed = st.number_input("Randomness seed (optional)", value=0, step=1)
    show_transcript = st.checkbox("Show generated script", value=True)

st.write("Type a **topic/goal** or **upload material** to base your lesson on:")

topic = st.text_input("Topic / Goal (e.g., 'order coffee and introduce myself')", "")
uploaded = st.file_uploader("Upload text or PDF (we’ll extract text)", type=["txt", "pdf"])
raw_text = st.text_area("Or paste text here", height=200)

# --- helper functions (extract PDF, clamp text, etc.) ---
def extract_text_from_pdf(file) -> str:
    try:
        import pypdf
    except ImportError:
        st.error("Install pypdf: pip install pypdf")
        return ""
    try:
        reader = pypdf.PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""

def clamp_len(s: str, n: int = MAX_INPUT_CHARS) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]

base_text = ""
if uploaded:
    if uploaded.type == "application/pdf":
        base_text = extract_text_from_pdf(uploaded)
    else:
        base_text = uploaded.read().decode("utf-8", errors="ignore")
elif raw_text:
    base_text = raw_text
base_text = clamp_len(base_text)

# ---- Lesson planning prompt ----
SYSTEM_INSTR = f"""
You are LaLangue, a lesson planner that produces AUDIO-based language lessons.

Principles:
- Anticipation: always pose the task BEFORE revealing the answer.
- Spaced Recall: resurface items later in the lesson.
- Core vocabulary in real contexts (short dialogues, not lists).
- Teach grammar organically through chunks and recombination.
- Use level {level} (short, simple sentences).
Return STRICT JSON in the schema below. Do not include explanations.

Schema:
{{
  "metadata": {{
    "language": "string",
    "level": "string",
    "duration_minutes": int
  }},
  "items": [
     {{"id":"it1","orth":"...", "phon":"IPA/pinyin if relevant","gloss_en":"..."}}
  ],
  "dialogues": [
     {{"id":"d1","situation":"cafe","turns":[{{"speaker":"A","text_tl":"..."}}]}}
  ],
  "timeline": [
     {{
       "t": 0.0,
       "action": "prompt" | "answer" | "recall",
       "voice": "coach" | "native",
       "text_en": "Say: I'd like coffee", 
       "text_tl": "Vorrei un caffè.",
       "expect_tl": "Vorrei un caffè.",
       "pause_sec": 4.5,
       "item_ids": ["it1"]
     }}
  ]
}}
"""

def make_user_prompt(topic: str, base_text: str) -> str:
    ref = f"\nREFERENCE TEXT:\n{base_text}\n" if base_text else ""
    return f"""
Create one {duration_min}-minute {lang} lesson for {level}.
Introduce {n_items} new items, include 10–14 recalls.
Topic/goal: "{topic or 'basic conversation'}".
Contexts must be natural (e.g., cafe, greeting).
{ref}
"""

def plan_lesson():
    messages = [
        {"role":"system","content": SYSTEM_INSTR},
        {"role":"user","content": make_user_prompt(topic, base_text)}
    ]
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        temperature=0.7,
        seed=seed or None,
        text_format={"type":"json_object"}
    )
    return json.loads(resp.output_text)

def tts_to_wav(text: str, voice: str) -> AudioSegment:
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    audio_bytes = speech.read()
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

def assemble_audio(timeline: List[Dict]) -> (AudioSegment, List[str]):
    lesson = AudioSegment.silent(duration=250)
    captions = []
    for step in timeline:
        action, voice = step.get("action"), step.get("voice")
        text_en, text_tl = (step.get("text_en") or "").strip(), (step.get("text_tl") or "").strip()
        pause_sec = float(step.get("pause_sec", 0))

        if action in ("prompt","recall") and voice=="coach" and text_en:
            seg = tts_to_wav(text_en, TTS_VOICE_COACH)
            lesson += seg + Silence(duration=int(pause_sec*1000)).to_audio_segment()
            captions.append(f"COACH: {text_en} [pause {pause_sec:.1f}s]")

        elif action=="answer" and voice=="native" and text_tl:
            seg = tts_to_wav(text_tl, TTS_VOICE_NATIVE)
            lesson += seg
            captions.append(f"NATIVE: {text_tl}")

        if action=="prompt" and step.get("expect_tl"):
            ans = step["expect_tl"].strip()
            seg = tts_to_wav(ans, TTS_VOICE_NATIVE)
            lesson += seg
            captions.append(f"NATIVE (answer): {ans}")

        lesson += AudioSegment.silent(duration=200)
    return lesson, captions

if st.button("Generate Lesson", type="primary"):
    if not (topic or base_text):
        st.warning("Please enter a topic or provide text.")
        st.stop()

    with st.spinner("Creating lesson..."):
        plan = plan_lesson()

    if show_transcript:
        st.subheader("Lesson JSON (preview)")
        st.json(plan)

    with st.spinner("Synthesizing audio..."):
        audio, caps = assemble_audio(plan["timeline"])

    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)

    st.audio(buf, format="audio/mp3")
    st.download_button("Download MP3", buf, file_name="lesson.mp3", mime="audio/mpeg")

    if show_transcript:
        st.subheader("Script")
        st.write("\n".join(caps))
