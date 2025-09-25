import streamlit as st
from openai import OpenAI
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
