import os, io, re, json, math, random
import streamlit as st
from typing import List, Dict, Tuple
from pydub import AudioSegment
from openai import OpenAI

# ───────────────── Config & API ─────────────────
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("❌ OPENAI_API_KEY não definido em Secrets ou variável de ambiente.")
    st.stop()

PLANNER_MODEL = "gpt-4o-mini"       # para seed (JSON)
TTS_MODEL     = "gpt-4o-mini-tts"   # TTS
VOICE_COACH   = "alloy"
VOICE_NATIVE  = "verse"

SEC_PER_PROMPT = 10.0               # média por ciclo coach→pausa→nativo
MAX_INPUT_CHARS = 9000
MAX_OUTPUT_TOKENS = 4000

client = OpenAI(api_key=API_KEY)

st.set_page_config(page_title="LaLangue", page_icon="🎧", layout="centered")
st.title("🎧 LaLangue — Lições em Áudio")

with st.sidebar:
    st.header("Configuração")
    lang = st.selectbox("Língua-alvo", ["Portuguese", "Spanish", "French", "Italian", "Mandarin Chinese", "Russian"])
    level = st.selectbox("Nível (CEFR)", ["A1", "A2", "B1"])
    duration_min = st.slider("Duração (min)", 8, 30, 15)
    n_items = st.slider("Novos itens (chunks)", 5, 12, 8)
    show_transcript = st.checkbox("Mostrar script", True)

st.write("Diga um **tópico/objetivo** ou **envie material** (TXT/PDF).")
topic = st.text_input("Ex.: pedir café, perguntar o preço, pagar e se despedir", "")
uploaded = st.file_uploader("Enviar TXT/PDF", type=["txt", "pdf"])
raw_text = st.text_area("Ou cole um texto", height=180)

# ───────────────── Helpers ─────────────────
def extract_text_from_pdf(file) -> str:
    try:
        import pypdf
    except ImportError:
        st.error("Instale pypdf no requirements.txt")
        return ""
    try:
        reader = pypdf.PdfReader(file)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception as e:
        st.warning(f"PDF não lido: {e}")
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

def target_steps(minutes: int) -> int:
    return max(18, math.ceil((minutes * 60.0) / SEC_PER_PROMPT))

# ───────────────── Seed prompt ─────────────────
SYSTEM_INSTR = """
You are LaLangue, planning audio lessons with this macro-structure:

A) Welcome/scene (coach, English/Portuguese brief).
B) Natural dialogue (20–30 seconds) between two natives, in the target language only.
C) Shadowing: replay the same dialogue line-by-line.
D) Anticipation drills (coach cue in English, learner answers in TL).
E) Spaced recall (reuse earlier chunks in new micro-scenes).
F) Role-play (coach gives situational cues; learner speaks).

IMPORTANT:
- Focus on useful phrases/chunks (not isolated words).
- Keep CEFR level constraints (short, clear, high-frequency patterns).
- Dialogue must sound natural, contextual, and short (20–30s total).
- Return STRICT JSON ONLY with the schema below. No markdown.

Schema:
{
  "metadata": { "language": "string", "level": "string" },
  "items": [
    {"id":"it1","orth":"chunk in TL","gloss_en":"English gloss","scene":"cafe/greeting/etc"}
  ],
  "dialogue": {
    "scene":"cafe/greeting/etc",
    "approx_seconds": 25,
    "turns": [
      {"speaker":"A","text_tl":"..."},
      {"speaker":"B","text_tl":"..."}
    ]
  },
  "cue_bank": {
    "anticipation": [
      {"cue_en":"Say you'd like a small coffee, please.","expect_tl":"...","scene":"cafe"},
      {"cue_en":"Ask how much it is.","expect_tl":"...","scene":"market"}
    ],
    "roleplay": [
      "Greet, ask how they are, then order quickly.",
      "Say you’re in a hurry, ask for the price, pay by card, and thank them."
    ]
  }
}
"""

def make_user_prompt(lang, level, n_items, topic, base_text):
    ref = f"\nREFERENCE (optional):\n{base_text}\n" if base_text else ""
    return f"""
Language: {lang}
Level: {level}
Introduce {n_items} NEW items (phrases/chunks) bound to scenes.
Topic/goal: {topic or "basic everyday conversation in natural scenes (cafe/market/greeting)"}.
The dialogue must last about 20–30 seconds total when read naturally.

Output STRICT JSON matching the schema. No extra text.
{ref}
""".strip()

def extract_json_block(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        json.loads(s); return s
    except Exception:
        pass
    m = re.search(r"\{.*\}\s*$", s, flags=re.DOTALL)
    return m.group(0) if m else s

def get_seed(lang, level, n_items, topic, base_text) -> Dict:
    messages = [
        {"role":"system","content": SYSTEM_INSTR},
        {"role":"user","content": make_user_prompt(lang, level, n_items, topic, base_text)}
    ]
    resp = client.responses.create(
        model=PLANNER_MODEL,
        input=messages,
        temperature=0.7,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    content = resp.output_text or ""
    return json.loads(extract_json_block(content))

# ───────────────── TTS ─────────────────
def tts_mp3(text: str, voice: str) -> AudioSegment:
    speech = client.audio.speech.create(model=TTS_MODEL, voice=voice, input=text)
    data = speech.read()
    return AudioSegment.from_file(io.BytesIO(data), format="mp3")

# ───────────────── Timeline builder ─────────────────
def section_welcome(scene: str, minutes: int) -> List[Dict]:
    msg = f"Welcome! You’ll practice {scene}. Listen, repeat, and answer in {lang}."
    return [{
        "action":"coach","voice":"coach","text_en":msg,"pause_sec":2.5,"scene":scene
    }]

def section_dialogue(dialogue: Dict) -> List[Dict]:
    steps=[]
    # Play full dialogue once (native only)
    full = " ".join([t["text_tl"].strip() for t in dialogue.get("turns", []) if t.get("text_tl")])
    if full:
        steps.append({"action":"native","voice":"native","text_tl":full,"scene":dialogue.get("scene","generic")})
    return steps

def section_shadowing(dialogue: Dict) -> List[Dict]:
    steps=[]
    for t in dialogue.get("turns", []):
        line = (t.get("text_tl") or "").strip()
        if not line: continue
        steps.append({"action":"coach","voice":"coach","text_en":"Repeat:", "pause_sec":0.8, "scene":dialogue.get("scene","generic")})
        steps.append({"action":"native","voice":"native","text_tl":line, "scene":dialogue.get("scene","generic")})
        steps.append({"action":"pause","pause_sec":3.8})
    return steps

def section_anticipation(cues: List[Dict], max_pairs: int) -> List[Dict]:
    steps=[]
    for c in cues[:max_pairs]:
        cue = c.get("cue_en","").strip()
        exp = c.get("expect_tl","").strip()
        scene = c.get("scene","generic")
        if not cue or not exp: continue
        steps.append({"action":"coach","voice":"coach","text_en":cue,"pause_sec":4.5,"scene":scene})
        steps.append({"action":"native","voice":"native","text_tl":exp,"scene":scene})
    return steps

def section_spaced_recall(items: List[Dict], count: int) -> List[Dict]:
    steps=[]
    if not items: return steps
    sample = random.sample(items, k=min(count, len(items)))
    recall_cues = [
        "Say it more politely.",
        "Say it faster, you’re in a hurry.",
        "Ask them to repeat.",
        "Ask for a smaller one.",
        "Say you’d like to pay by card.",
        "Ask where it is.",
    ]
    for it in sample:
        cue = random.choice(recall_cues)
        steps.append({"action":"coach","voice":"coach","text_en":cue,"pause_sec":4.5,"scene":it.get("scene","generic")})
        # Sem resposta explícita — deixa para o aluno produzir
    return steps

def section_roleplay(role_cues: List[str], max_cues: int) -> List[Dict]:
    steps=[]
    for cue in role_cues[:max_cues]:
        steps.append({"action":"coach","voice":"coach","text_en":cue,"pause_sec":6.0,"scene":"roleplay"})
    return steps

def build_timeline(seed: Dict, minutes: int) -> List[Dict]:
    dialogue = seed.get("dialogue", {})
    items = seed.get("items", [])
    cues = (seed.get("cue_bank") or {}).get("anticipation", [])
    role_cues = (seed.get("cue_bank") or {}).get("roleplay", [])

    # 1) welcome
    tl = section_welcome(dialogue.get("scene","greeting"), minutes)
    # 2) natural dialogue (20–30s)
    tl += section_dialogue(dialogue)
    # 3) shadowing
    tl += section_shadowing(dialogue)
    # 4) anticipation drills (limit para evitar exagero)
    tl += section_anticipation(cues, max_pairs=8)
    # 5) spaced recall (curto)
    tl += section_spaced_recall(items, count=6)
    # 6) role-play final
    tl += section_roleplay(role_cues, max_cues=4)

    # limpeza: remove passos vazios e mescla pausas contíguas
    final=[]
    for step in tl:
        if step.get("action")=="pause" and step.get("pause_sec",0)<=0: 
            continue
        if step.get("action") in ("coach","native") and not (step.get("text_en") or step.get("text_tl")):
            continue
        final.append(step)

    return final

# ───────────────── Áudio a partir da timeline ─────────────────
def render_audio(timeline: List[Dict], target_minutes: int) -> Tuple[AudioSegment, List[str]]:
    audio = AudioSegment.silent(duration=250)
    caps=[]
    for s in timeline:
        act = s.get("action")
        if act=="coach":
            txt=s.get("text_en","").strip()
            if txt:
                audio += tts_mp3(txt, VOICE_COACH)
                caps.append(f"COACH: {txt}")
            pause = float(s.get("pause_sec",0))
            if pause>0: audio += AudioSegment.silent(duration=int(pause*1000))
        elif act=="native":
            tl=s.get("text_tl","").strip()
            if tl:
                audio += tts_mp3(tl, VOICE_NATIVE)
                caps.append(f"NATIVE: {tl}")
        elif act=="pause":
            audio += AudioSegment.silent(duration=int(float(s.get("pause_sec",1))*1000))
        audio += AudioSegment.silent(duration=120)

    # padding leve para atingir duração
    target_ms = int(target_minutes*60_000)
    if len(audio) < int(target_ms*0.95):
        audio += AudioSegment.silent(duration=max(0, target_ms - len(audio)))
    return audio, caps

# ───────────────── Run ─────────────────
if st.button("Gerar Lição", type="primary"):
    if not (topic or base_text):
        st.warning("Informe um tópico ou material.")
        st.stop()

    with st.spinner("Montando seed (itens + diálogo natural)…"):
        seed = get_seed(lang, level, n_items, topic, base_text)

    # Construímos a timeline estilo Pimsleur
    timeline = build_timeline(seed, duration_min)

    if show_transcript:
        st.subheader("Seed (JSON)")
        st.json(seed)
        st.subheader("Timeline construída")
        st.json(timeline)

    with st.spinner("Sintetizando áudio…"):
        audio, caps = render_audio(timeline, duration_min)

    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)
    st.audio(buf, format="audio/mp3")
    st.download_button("Baixar MP3", buf, file_name="lesson.mp3", mime="audio/mpeg")

    if show_transcript:
        st.subheader("Script")
        st.write("\n".join(caps))
