import os, io, re, json, math, time
import streamlit as st
from typing import List, Dict, Tuple
from pydub import AudioSegment
from openai import OpenAI

# =================== KEYS & CLIENT ===================
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("❌ OPENAI_API_KEY not set. Add it in Streamlit Secrets or your local environment.")
    st.stop()
client = OpenAI(api_key=API_KEY)

# =================== APP CONFIG ======================
st.set_page_config(page_title="LaLangue", page_icon="🎧", layout="centered")
st.title("🎧 LaLangue – Audio Lesson Generator")

# =================== SIDEBAR =========================
with st.sidebar:
    st.header("Lesson Settings")
    lang = st.selectbox("Target language", ["Spanish", "French", "Italian", "Portuguese", "Mandarin Chinese", "Russian"])
    level = st.selectbox("Level (CEFR)", ["A1", "A2", "B1"])
    duration_min = st.slider("Target duration (minutes)", 5, 45, 30)
    base_pause = st.slider("Base pause after prompts (seconds)", 2.0, 8.0, 4.5, 0.5)
    coach_wpm = st.slider("Coach speaking pace (words/min)", 110, 180, 140, 5)
    native_wpm = st.slider("Native speaking pace (words/min)", 120, 200, 160, 5)
    richness = st.selectbox("Content richness", ["standard", "detailed", "very_detailed"])
    n_items = st.slider("New lexical items", 6, 16, 10)
    seed = st.number_input("Randomness seed (optional)", value=0, step=1)
    show_transcript = st.checkbox("Show generated JSON & script", value=True)

st.write("Enter a **topic/goal** and/or **upload material** to base your lesson on:")

topic = st.text_input("Topic / Goal (e.g., ‘order coffee and small talk’)", "")
uploaded = st.file_uploader("Upload text or PDF (we’ll extract text)", type=["txt", "pdf"])
raw_text = st.text_area("Or paste text here", height=180)

# =================== HELPERS =========================
MAX_INPUT_CHARS = 8000

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

# =================== PROMPTS =========================
OPENAI_MODEL = "gpt-4o"       # text planner
TTS_MODEL    = "gpt-4o-mini-tts"  # TTS

def system_instructions() -> str:
    richness_hint = {
        "standard": "Keep scenes concise but natural.",
        "detailed": "Use richer context, short follow-ups, and occasional alternatives.",
        "very_detailed": "Add micro-variations, quick reformulations, and brief cultural notes (1 sentence)."
    }[richness]

    return f"""
You are LaLangue, an expert designer of AUDIO language lessons.

You will output STRICT JSON matching the provided schema. Do not include commentary.

Pedagogy:
- Anticipation: always pose the task BEFORE revealing the answer.
- Spaced Recall: recycle earlier items later; include multiple recall cycles.
- Real contexts: dialogs (cafe, street, store, phone), micro-situations.
- Chunks over rules: teach grammar through pattern examples & recombinations.
- Level: {level} → short clear sentences; avoid rare vocabulary.
- Native lines must be in {lang}. English only for instructions to the learner.

Structure:
- Sections: Warm-up → Core Scenes (2–4) → Variations → Recall Cycles → Wrap-up.
- Each 'prompt' by COACH in English; learner gets a pause; then NATIVE provides model answer.
- Insert occasional 'recall' tasks referencing items from earlier.

Duration targeting:
- Aim for {duration_min} minutes total audio.
- Base pause default: {base_pause} seconds after prompts.
- Provide enough steps to reach the target. err on the side of slightly longer.

Richness mode: {richness.upper()} — {richness_hint}

Output JSON schema:
{{
  "metadata": {{
    "language": "{lang}",
    "level": "{level}",
    "duration_target_min": {duration_min},
    "sections": ["warmup","scene1","scene2","variations","recalls","wrapup"]
  }},
  "items": [
    {{"id": "it1", "orth": "...", "phon": "IPA/Pinyin if relevant", "gloss_en": "..."}}
  ],
  "timeline": [
    {{
      "t": 0.0,                         # cumulative seconds, non-decreasing
      "section": "warmup|scene1|scene2|variations|recalls|wrapup",
      "action": "prompt|answer|recall",
      "voice": "coach|native",
      "text_en": "Say: I'd like a coffee",   # English for COACH lines
      "text_tl": "Vorrei un caffè.",         # Target language for native lines
      "expect_tl": "Vorrei un caffè.",       # Expected learner answer (when applicable)
      "pause_sec": {base_pause},              # base pause, may vary per step
      "item_ids": ["it1","it3"]               # referenced lexical items if relevant
    }}
  ]
}}
Important:
- Ensure at least {n_items} new lexical items appear in 'items' and are used in 'timeline'.
- Provide at least ~ (duration_min * 60) / (base_pause + 2.5) prompt/answer units.
- Keep 'timeline' long enough to meet duration target. 
- Never include non-JSON text.
"""

def make_user_prompt(topic: str, base_text: str) -> str:
    ref = f"\nREFERENCE TEXT (optional signals, style & key phrases):\n{base_text}\n" if base_text else ""
    return f"""
Design a {duration_min}-minute {lang} lesson for level {level}.
Topic/goal: "{topic or 'basic daily conversation'}".
Introduce {n_items} new items and recycle them via spaced recall.
Include at least 2 core scenes with natural dialogs, plus recall cycles and a short wrap-up.
{ref}
"""

# =================== DURATION ESTIMATION =============
def count_words(s: str) -> int:
    return len([w for w in re.findall(r"[\\w\\-’']+", s or "")])

def estimate_timeline_seconds(timeline: List[Dict]) -> float:
    total = 0.0
    for step in timeline:
        a = (step.get("action") or "").lower()
        v = (step.get("voice") or "").lower()
        pause = float(step.get("pause_sec", 0.0))

        text = (step.get("text_en") if v == "coach" else step.get("text_tl")) or ""
        words = count_words(text)
        wpm = coach_wpm if v == "coach" else native_wpm
        speech = (words / max(wpm, 60)) * 60.0  # seconds
        total += speech
        # pauses only matter after prompts/recalls (anticipation)
        if a in ("prompt", "recall"):
            total += pause
        # small spacer between steps (~0.2s) will be added during assembly
        total += 0.2
    return total

def need_more(plan: Dict) -> Tuple[bool, float, float]:
    secs = estimate_timeline_seconds(plan.get("timeline", []))
    target = duration_min * 60.0
    return (secs < target * 0.98), secs, target

# =================== OPENAI CALLS ====================
def call_plan(messages) -> Dict:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        temperature=0.6,
        seed=seed or None,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.output_text)

def initial_plan() -> Dict:
    messages = [
        {"role":"system","content": system_instructions()},
        {"role":"user","content": make_user_prompt(topic, base_text)},
    ]
    return call_plan(messages)

def expand_timeline(plan: Dict, short_by_sec: float) -> Dict:
    # ask the model to append more steps continuing from last t to reach target
    last_t = 0.0
    if plan.get("timeline"):
        try:
            last_t = float(plan["timeline"][-1].get("t", 0.0))
        except:
            last_t = 0.0

    messages = [
        {"role":"system","content": system_instructions()},
        {"role":"user","content":
         f"""Your previous JSON was too short by ~{int(short_by_sec)} seconds.
Continue the SAME JSON timeline ONLY by APPENDING steps (do not repeat previous items).
Constraints:
- Start at t >= {last_t:.1f}
- Keep sections flowing: add more variations & recall cycles, then wrap-up.
- Maintain anticipation (prompt/recall → pause → native answer).
- Keep {lang} for native lines, English for coach.
Return STRICT JSON with ONLY:
{{"timeline": [ ... only new steps ... ]}}"""
        }
    ]
    add = call_plan(messages)
    if "timeline" in add and isinstance(add["timeline"], list):
        plan["timeline"].extend(add["timeline"])
    return plan

# =================== TTS & ASSEMBLY ==================
def tts_to_seg(text: str, voice: str) -> AudioSegment:
    speech = client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text)
    audio_bytes = speech.read()
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

def assemble_audio(timeline: List[Dict]) -> Tuple[AudioSegment, List[str]]:
    lesson = AudioSegment.silent(duration=250)
    captions = []
    for step in timeline:
        action = (step.get("action") or "").lower()
        voice  = (step.get("voice") or "").lower()
        text_en, text_tl = (step.get("text_en") or "").strip(), (step.get("text_tl") or "").strip()
        pause_sec = float(step.get("pause_sec", base_pause))

        if action in ("prompt","recall") and voice == "coach" and text_en:
            seg = tts_to_seg(text_en, "alloy")
            lesson += seg + AudioSegment.silent(duration=int(pause_sec * 1000))
            captions.append(f"COACH: {text_en} [pause {pause_sec:.1f}s]")

        elif action == "answer" and voice == "native" and text_tl:
            seg = tts_to_seg(text_tl, "verse")
            lesson += seg
            captions.append(f"NATIVE: {text_tl}")

        # expected answer after a prompt (model readout)
        if action == "prompt" and step.get("expect_tl"):
            ans = step["expect_tl"].strip()
            if ans:
                seg = tts_to_seg(ans, "verse")
                lesson += seg
                captions.append(f"NATIVE (answer): {ans}")

        # small spacer
        lesson += AudioSegment.silent(duration=200)
    return lesson, captions

# =================== BUTTON ACTION ===================
if st.button("Generate Lesson", type="primary"):
    if not (topic or base_text):
        st.warning("Please enter a topic or provide text.")
        st.stop()

    with st.spinner("Planning lesson..."):
        plan = initial_plan()
        # grow until duration target is reached
        for _ in range(6):  # up to 6 expansions
            too_short, secs, target = need_more(plan)
            if not too_short:
                break
            plan = expand_timeline(plan, target - secs)

    # sanity checks
    timeline = plan.get("timeline", [])
    if not isinstance(timeline, list) or len(timeline) < 60 and duration_min >= 20:
        st.warning("The first pass seems short. I’ll try to enrich the lesson one more time.")
        plan = expand_timeline(plan, duration_min*60 - estimate_timeline_seconds(timeline))
        timeline = plan.get("timeline", [])

    if show_transcript:
        st.subheader("Lesson JSON (preview)")
        st.json(plan)

    with st.spinner("Synthesizing audio..."):
        audio, caps = assemble_audio(timeline)

    # export
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)

    est_min = estimate_timeline_seconds(timeline) / 60.0
    st.success(f"Estimated duration: ~{est_min:.1f} min (target {duration_min} min)")
    st.audio(buf, format="audio/mp3")
    st.download_button("Download MP3", buf, file_name="lesson.mp3", mime="audio/mpeg")

    if show_transcript:
        st.subheader("Script")
        st.write("\n".join(caps))
