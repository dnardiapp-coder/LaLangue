import os, io, re, json, math, random
import streamlit as st
from pydub import AudioSegment
from typing import List, Dict
from openai import OpenAI

# ───────── Config & API key ─────────
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("❌ OPENAI_API_KEY not set. Add it in Streamlit Secrets or your local environment.")
    st.stop()

PLANNER_MODEL = "gpt-4o-mini"      # JSON-capable
TTS_MODEL     = "gpt-4o-mini-tts"  # text-to-speech
VOICE_COACH   = "alloy"
VOICE_NATIVE  = "verse"

# Heuristic: ~10 seconds per prompt→pause→answer cycle
SEC_PER_CYCLE = 10.0
MAX_INPUT_CHARS = 9000
MAX_OUTPUT_TOKENS = 4000  # allow larger JSON

client = OpenAI(api_key=API_KEY)

st.set_page_config(page_title="LaLangue", page_icon="🎧", layout="centered")
st.title("🎧 LaLangue – Audio Lesson Generator")

with st.sidebar:
    st.header("Lesson Settings")
    lang = st.selectbox("Target language", ["Spanish", "French", "Italian", "Portuguese", "Mandarin Chinese", "Russian"])
    level = st.selectbox("Level (CEFR)", ["A1", "A2", "B1"])
    duration_min = st.slider("Lesson duration target (minutes)", 8, 30, 15)
    n_items = st.slider("New items to introduce", 5, 12, 8)
    show_transcript = st.checkbox("Show generated script", value=True)

st.write("Type a **topic/goal** or **upload material** to base your lesson on:")

topic = st.text_input("Topic / Goal (e.g., 'order coffee and introduce myself')", "")
uploaded = st.file_uploader("Upload text or PDF (we’ll extract text)", type=["txt", "pdf"])
raw_text = st.text_area("Or paste text here", height=200)

# ───────── Helpers ─────────
def extract_text_from_pdf(file) -> str:
    try:
        import pypdf
    except ImportError:
        st.error("Install pypdf: pip install pypdf")
        return ""
    try:
        reader = pypdf.PdfReader(file)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
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

# ───────── Prompting ─────────
def target_steps_for(minutes: int) -> int:
    return max(20, math.ceil((minutes * 60.0) / SEC_PER_CYCLE))

SYSTEM_INSTR = """
You are LaLangue, designing AUDIO language lessons faithful to these principles:

- Anticipation: prompt the learner BEFORE revealing the answer.
- Spaced Recall: frequently resurface earlier items later in the timeline.
- Natural micro-scenes: short role-plays (café, market, greeting, asking directions), not isolated word lists.
- Chunking: teach useful phrases & collocations; avoid single isolated words unless necessary.
- Organic grammar: expose patterns in context (I’d like…, Could I…, Where is…?).
- Level control: keep sentences short, high-utility, CEFR appropriate.

STRICT JSON ONLY (no markdown, no comments). Schema:

{
  "metadata": { "language": "string", "level": "string", "duration_minutes": 15 },
  "items": [ { "id":"it1", "orth":"...", "phon":"IPA/pinyin if relevant", "gloss_en":"..." } ],
  "dialogues": [
    { "id":"d1", "situation":"cafe", "turns":[ { "speaker":"A","text_tl":"..."}, {"speaker":"B","text_tl":"..."} ] }
  ],
  "timeline": [
    {
      "t": 0.0,
      "action": "prompt" | "recall",            // coach instructs learner in English
      "voice": "coach",
      "scene": "cafe/greeting/market/etc",
      "text_en": "Say: I'd like a small coffee, please.",
      "expect_tl": "Vorrei un caffè piccolo, per favore.",
      "pause_sec": 4.5,
      "item_ids": ["it3","it5"]
    },
    {
      "t": 6.0,
      "action": "answer",                       // native model answer
      "voice": "native",
      "scene": "cafe",
      "text_tl": "Vorrei un caffè piccolo, per favore.",
      "item_ids": ["it3","it5"]
    }
  ]
}

Rules:
- Most cycles are 2-step: (1) coach prompt with pause + expected answer; (2) native answer. Use this pattern consistently.
- 35–45% of total steps must be "recall" prompts that reuse earlier items in new contexts.
- Use situation labels in "scene" for every step.
- Avoid dictionary-style "What is X?" Replace with functional prompts (“Ask for the price”, “Say you’re in a hurry”).
- Dialogues should coincide with scenes used in the timeline (brief, snappy, 2–6 turns).
"""

def make_user_prompt(lang, level, duration_min, n_items, topic, base_text):
    steps = target_steps_for(duration_min)
    ref = f"\nREFERENCE TEXT (optional):\n{base_text}\n" if base_text else ""
    return f"""
Language: {lang}
Level: {level}
Target duration: {duration_min} minutes.
Target steps in "timeline": {steps} (±10%). Ensure enough content to fill the time at ~10 seconds per prompt→pause→answer cycle.

Introduce {n_items} NEW high-utility items (phrases/chunks). NO lists of unrelated single words.
Ensure 35–45% of the timeline are "recall" steps resurfacing earlier items in varied scenes.

Topic/goal: {topic or "basic everyday conversation in natural scenes (greeting, ordering, paying, parting)"}.

Produce STRICT JSON matching the schema and rules. Do not include any extra text outside JSON.
{ref}
""".strip()

def extract_json_block(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    if m: return m.group(0)
    return s

def plan_lesson(lang, level, duration_min, n_items, topic, base_text):
    messages = [
        {"role": "system", "content": SYSTEM_INSTR},
        {"role": "user", "content": make_user_prompt(lang, level, duration_min, n_items, topic, base_text)},
    ]
    resp = client.responses.create(
        model=PLANNER_MODEL,
        input=messages,
        temperature=0.7,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    content = resp.output_text or ""
    json_str = extract_json_block(content)
    return json.loads(json_str)

# ───────── Post-processing & Quality Guards ─────────
def ensure_two_step_cycles(timeline: List[Dict]) -> List[Dict]:
    """If a prompt/recall lacks an immediate native answer step, inject one."""
    fixed = []
    for step in timeline:
        fixed.append(step)
        if step.get("voice") == "coach" and step.get("action") in ("prompt","recall"):
            exp = (step.get("expect_tl") or "").strip()
            if exp:
                # If next already an answer for same items, keep; else add answer.
                need_answer = True
                if len(fixed) >= 2:
                    prev = fixed[-2]
                # Peek upcoming in original timeline
                # Simpler: always append an answer; duplicates are rare and harmless.
                answer = {
                    "t": float(step.get("t", 0.0)) + 6.0,
                    "action": "answer",
                    "voice": "native",
                    "scene": step.get("scene", "generic"),
                    "text_tl": exp,
                    "item_ids": step.get("item_ids", []),
                }
                fixed.append(answer)
    return fixed

def expand_recalls(plan: Dict, target_steps: int) -> Dict:
    """Append additional recall cycles sampling earlier items until we meet target steps."""
    timeline = plan.get("timeline", [])
    items = plan.get("items", [])
    ids = [i["id"] for i in items] if items else []
    scenes = ["greeting","cafe","market","directions","checkout","smalltalk","transport"]

    def recall_prompt(for_ids, scene):
        gloss = ""
        # Make an English functional cue that uses one or two known items
        cue = random.choice([
            "Politely ask for it again.",
            "Say you’re in a hurry.",
            "Ask the price.",
            "Ask for a smaller one.",
            "Greet and ask how they are.",
            "Say you’d like to pay by card.",
            "Ask where the station is.",
            "Say you didn’t catch that.",
        ])
        return {
            "t": 0.0,
            "action": "recall",
            "voice": "coach",
            "scene": scene,
            "text_en": f"{cue}",
            "expect_tl": "",         # let the model’s earlier 'expect_tl' be recycled via answer step below
            "pause_sec": 4.5,
            "item_ids": for_ids,
        }

    # Add recall pairs until >= target
    while len(timeline) < target_steps and ids:
        chosen = random.sample(ids, k=min(2, len(ids)))
        scene = random.choice(scenes)
        p = recall_prompt(chosen, scene)
        a = {
            "t": 0.0,
            "action": "answer",
            "voice": "native",
            "scene": scene,
            "text_tl": "",   # will be empty; but our audio builder only speaks when text is present
            "item_ids": chosen
        }
        timeline.extend([p, a])

    plan["timeline"] = timeline
    return plan

def repair_and_pad_plan(plan: Dict, minutes: int) -> Dict:
    """Apply guards: two-step cycles + reach target steps with recalls."""
    steps_target = target_steps_for(minutes)
    tl = plan.get("timeline", [])
    # 1) Force two-step cycles
    tl = ensure_two_step_cycles(tl)
    # 2) Expand (recalls) if under target
    plan["timeline"] = tl
    if len(tl) < steps_target:
        plan = expand_recalls(plan, steps_target)
    return plan

# ───────── TTS & Assembly ─────────
def tts_to_seg(text: str, voice: str) -> AudioSegment:
    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text
    )
    audio_bytes = speech.read()
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

def assemble_audio(timeline: List[Dict], target_minutes: int) -> (AudioSegment, List[str]):
    lesson = AudioSegment.silent(duration=250)
    captions = []
    for step in timeline:
        action  = step.get("action")
        voice   = step.get("voice")
        text_en = (step.get("text_en") or "").strip()
        text_tl = (step.get("text_tl") or "").strip()
        expect  = (step.get("expect_tl") or "").strip()
        pause   = float(step.get("pause_sec", 0))

        if action in ("prompt","recall") and voice == "coach" and text_en:
            lesson += tts_to_seg(text_en, VOICE_COACH)
            if pause > 0:
                lesson += AudioSegment.silent(duration=int(pause * 1000))
            captions.append(f"COACH: {text_en} [pause {pause:.1f}s]")

            # If prompt includes expected TL, add the native answer immediately (Pimsleur rhythm)
            if expect:
                lesson += tts_to_seg(expect, VOICE_NATIVE)
                captions.append(f"NATIVE (answer): {expect}")

        elif action == "answer" and voice == "native" and text_tl:
            lesson += tts_to_seg(text_tl, VOICE_NATIVE)
            captions.append(f"NATIVE: {text_tl}")

        lesson += AudioSegment.silent(duration=200)

    # If still short, top up with a brief closing silence so player reaches target
    target_ms = int(target_minutes * 60_000)
    if len(lesson) < int(target_ms * 0.95):   # don’t over-pad; aim ~95% minimum
        lesson += AudioSegment.silent(duration=max(0, target_ms - len(lesson)))

    return lesson, captions

# ───────── Run ─────────
if st.button("Generate Lesson", type="primary"):
    if not (topic or base_text):
        st.warning("Please enter a topic or provide text.")
        st.stop()

    with st.spinner("Planning lesson (scenes, items, timeline)…"):
        try:
            plan = plan_lesson(lang, level, duration_min, n_items, topic, base_text)
        except Exception as e:
            st.error(f"Failed to plan lesson: {e}")
            st.stop()

    # Repair + ensure enough steps/duration
    plan = repair_and_pad_plan(plan, duration_min)

    if show_transcript:
        st.subheader("Lesson JSON (preview)")
        st.json(plan)

    with st.spinner("Synthesizing audio…"):
        audio, caps = assemble_audio(plan.get("timeline", []), duration_min)

    # Deliver
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)

    st.audio(buf, format="audio/mp3")
    st.download_button("Download MP3", buf, file_name="lesson.mp3", mime="audio/mpeg")

    if show_transcript:
        st.subheader("Script")
        st.write("\n".join(caps))
