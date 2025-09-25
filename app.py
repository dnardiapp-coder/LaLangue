import streamlit as st
from import os, io, re, json, math, time
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

def need_more(plan: Dict]) -> Tuple[bool, float, float]:
    secs = estimate_timeline_seconds(plan.get("timeline", []))
    target = duration_min * 60.0
    return (secs < target * 0.98), secs, target

# =================== OPENAI CALLS ====================
def call_plan(messages) -> Dict:
    # Use Chat Completions for robust JSON output
    chat = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.6,
        seed=seed or None,
        response_format={"type": "json_object"},
    )
    content = chat.choices[0].message.content
    try:
        return json.loads(content)
    except Exception as e:
        # Surface the raw text to help debugging
        st.error("Model returned non-JSON. Showing raw text below to help diagnose.")
        st.code(content)
        raise

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
openai import OpenAI
import pydub
from pydub import AudioSegment
import io
import PyPDF2
import docx
from typing import List, Dict, Union

# --- UI Configuration ---
st.set_page_config(
    page_title="Pimsleur Audio Lesson Generator",
    page_icon="🗣️",
    layout="wide",
)

st.title("🗣️ Pimsleur-Style Audio Lesson Generator")
st.markdown("""
    Create custom language lessons based on the Pimsleur method. 
    Provide a topic, text, or a document, and the AI will generate a script and an audio lesson for you.
    **Note:** Audio generation for a 5-minute lesson can take 1-2 minutes.
""")

# --- Helper Functions ---

def get_text_from_file(uploaded_file):
    """Extracts text from uploaded file (PDF, DOCX, TXT)."""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def create_pimsleur_prompt(native_language: str, target_language: str, level: str, context: str) -> str:
    """Creates the detailed system prompt for the AI script generator."""
    return f"""
    You are an expert linguist and curriculum designer specializing in the Pimsleur language learning method.
    Your task is to create a script for a 5-minute audio lesson to teach {target_language} to a native {native_language} speaker at a {level} level.

    **Pimsleur Method Core Principles:**
    1.  **Anticipation Principle:** The narrator prompts the student to say a phrase, pauses for the student to respond, and then a native speaker provides the correct answer. This is crucial.
    2.  **Graduated Interval Recall:** Introduce new phrases and then ask the student to recall them at increasing intervals.
    3.  **Core Vocabulary:** Focus on a small number of useful, common phrases. Build upon them.
    4.  **Organic Learning:** Introduce new concepts within a conversational context. Break down longer sentences into parts and have the student build them back up.

    **Lesson Context:**
    The lesson should be based on the following topic or text. Extract key vocabulary and phrases from it:
    ---
    {context}
    ---

    **Output Format:**
    You MUST output the script as a JSON array of objects. Each object represents a segment of the lesson and must have two keys: "speaker" and "line".
    - `speaker`: Can be "narrator", "{target_language.lower()}", or "pause".
    - `line`: The text to be spoken. For "pause", the line should be the duration of the pause in seconds (e.g., "3").

    **Example JSON Structure:**
    [
      {{"speaker": "narrator", "line": "Welcome to your {target_language} lesson. Let's begin."}},
      {{"speaker": "pause", "line": "2"}},
      {{"speaker": "narrator", "line": "Listen to the phrase for 'Hello, how are you?'"}},
      {{"speaker": "{target_language.lower()}", "line": "你好，你好吗？"}},
      {{"speaker": "pause", "line": "4"}},
      {{"speaker": "narrator", "line": "Now, imagine you're greeting a friend. How do you say 'Hello, how are you?'"}},
      {{"speaker": "pause", "line": "5"}},
      {{"speaker": "{target_language.lower()}", "line": "你好，你好吗？"}}
    ]

    **Instructions:**
    - The lesson must be progressive and easy to follow for a {level}.
    - The narrator speaks only in {native_language}.
    - The `{target_language.lower()}` speaker speaks only in {target_language}.
    - Include plenty of pauses (3-5 seconds) after prompts to give the student time to think and speak.
    - Ensure the total lesson feels like a cohesive 5-minute learning experience.
    - Start the generation now.
    """


def generate_audio_segment(client: OpenAI, text: str, voice: str) -> AudioSegment:
    """Generates an audio segment for a line of text using OpenAI's TTS API."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        audio_data = response.read()
        return AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    except Exception as e:
        # Fallback to a silent segment on error
        st.warning(f"Could not generate audio for: '{text}'. Error: {e}")
        return AudioSegment.silent(duration=100)


def combine_audio_segments(
    client: OpenAI,
    script: List[Dict[str, str]],
    narrator_voice: str,
    target_voice: str,
    target_language_code: str
) -> io.BytesIO:
    """Combines individual audio clips and pauses into a single audio file."""
    
    full_lesson = AudioSegment.empty()
    total_segments = len(script)

    progress_bar = st.progress(0, text="Generating audio segments...")

    for i, segment in enumerate(script):
        speaker = segment.get("speaker", "").lower()
        line = segment.get("line", "")
        
        progress_text = f"Generating audio... Segment {i+1}/{total_segments}: {speaker.capitalize()}"
        progress_bar.progress((i + 1) / total_segments, text=progress_text)

        if speaker == "narrator":
            full_lesson += generate_audio_segment(client, line, narrator_voice)
        elif speaker == target_language_code:
            full_lesson += generate_audio_segment(client, line, target_voice)
        elif speaker == "pause":
            try:
                duration_ms = int(float(line) * 1000)
                full_lesson += AudioSegment.silent(duration=duration_ms)
            except ValueError:
                full_lesson += AudioSegment.silent(duration=2000) # Default 2s pause
        
        # Add a tiny silence between clips to prevent them from sounding too abrupt
        full_lesson += AudioSegment.silent(duration=150)

    progress_bar.progress(1.0, text="Audio generation complete! Exporting file...")
    
    # Export the final audio to an in-memory file
    final_audio = io.BytesIO()
    full_lesson.export(final_audio, format="mp3")
    final_audio.seek(0)
    
    progress_bar.empty()
    return final_audio

# --- Streamlit UI Layout ---

with st.sidebar:
    st.header("⚙️ Lesson Configuration")
    
    # Use session state to persist API key
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''

    api_key_input = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state.openai_api_key,
        help="Your API key is stored securely for this session only."
    )
    
    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    native_language = st.selectbox("Your Native Language", ["English", "Spanish", "German", "French", "Mandarin"])
    target_language = st.selectbox("Language to Learn", ["French", "Spanish", "Japanese", "German", "Italian", "Mandarin", "Korean"])
    level = st.selectbox("Proficiency Level", ["Beginner", "Intermediate", "Advanced"])
    
    st.header("🎙️ Voice Selection")
    narrator_voice = st.selectbox("Narrator Voice (Native Language)", ["nova", "echo", "onyx", "shimmer"], index=0)
    target_voice = st.selectbox("Target Language Voice", ["alloy", "fable", "onyx", "shimmer"], index=1)

st.header("📝 Lesson Content")
st.markdown("Provide the source material for your lesson.")

input_method = st.radio("Choose Input Method", ["Topic / Text", "Upload Document"], horizontal=True)

context = ""
if input_method == "Topic / Text":
    context = st.text_area(
        "Enter a topic, conversation, or text to base the lesson on.",
        "A conversation about ordering food in a restaurant.",
        height=150
    )
else:
    uploaded_file = st.file_uploader("Upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if uploaded_file:
        with st.spinner("Reading document..."):
            context = get_text_from_file(uploaded_file)
            st.success("Document loaded successfully!")
            st.text_area("Document Content (first 500 chars):", context[:500] + "...", height=100, disabled=True)

generate_button = st.button("Generate Audio Lesson", type="primary", use_container_width=True)


# --- Main Application Logic ---
if generate_button:
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not context.strip():
        st.warning("Please provide some content for the lesson (either text or a document).")
    else:
        try:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            
            with st.spinner("Step 1/3: Crafting the lesson prompt..."):
                prompt = create_pimsleur_prompt(native_language, target_language, level, context)

            with st.spinner("Step 2/3: Generating lesson script with GPT-4o... This may take a moment."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Please generate the Pimsleur lesson script now."}
                    ],
                    response_format={"type": "json_object"}
                )
                # The response is a string of JSON, so we need to load it.
                # The actual content is nested under response.choices[0].message.content
                script_text = response.choices[0].message.content
                # The model often wraps the list in a root object, e.g. {"script": [...]}, so we find the list.
                import json
                script_data = json.loads(script_text)
                
                # Handle cases where JSON is nested, e.g., {"script": [...]}
                if isinstance(script_data, dict):
                    # Find the first value in the dict that is a list
                    found_list = False
                    for key, value in script_data.items():
                        if isinstance(value, list):
                            lesson_script = value
                            found_list = True
                            break
                    if not found_list:
                         raise ValueError("JSON response does not contain a list of script segments.")
                elif isinstance(script_data, list):
                    lesson_script = script_data
                else:
                    raise ValueError("Could not parse the script from the AI response.")


            st.success("Lesson script generated successfully!")

            with st.expander("📖 View Generated Script", expanded=False):
                st.json(lesson_script)

            with st.spinner("Step 3/3: Generating and combining audio... This is the longest step."):
                final_audio_file = combine_audio_segments(
                    client,
                    lesson_script,
                    narrator_voice,
                    target_voice,
                    target_language.lower()
                )
            
            st.success("🎉 Your audio lesson is ready!")
            st.audio(final_audio_file, format='audio/mp3')

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("This could be due to an invalid API key, server issues, or a problem with the generated script format. Try again or check your key.")
