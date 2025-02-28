import streamlit as st
import os

# ---------------------------
# Load environment variables from st.secrets
# ---------------------------
if "env_vars" in st.secrets:
    for key, value in st.secrets["env_vars"].items():
        os.environ[key] = value

import asyncio
import tempfile

# ---------------------------
# Initialize pyttsx3 (with error handling)
# ---------------------------
try:
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Map characters to specific voice IDs. Adjust indices based on your system.
    character_voices = {
        "Sherlock Holmes": voices[0].id if voices else None,
        "William Shakespeare": voices[1].id if len(voices) > 1 else (voices[0].id if voices else None),
    }
except RuntimeError as e:
    st.warning("Text-to-speech is not available on this platform. Voice output is disabled.")
    engine = None
    voices = []
    character_voices = {}

from getpass import getpass
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

# ---------------------------
# Ensure an asyncio event loop is running
# ---------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def init_async():
        return
    loop.run_until_complete(init_async())

# ---------------------------
# API & Audio Setup Functions
# ---------------------------
def setup_openai_api():
    if "OPENAI_API_KEY" not in os.environ:
        if "env_vars" in st.secrets and "OPENAI_API_KEY" in st.secrets["env_vars"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["env_vars"]["OPENAI_API_KEY"]
        else:
            st.error("Missing OpenAI API Key in environment variables. Please add it to your Streamlit Secrets.")

def generate_audio(character, text):
    """
    Generate audio using pyttsx3 for the given character and text.
    Returns the audio bytes in WAV format, or None if TTS is unavailable.
    """
    if engine is None:
        return None

    # Set voice for the character if available, otherwise default to the first voice.
    if character in character_voices and character_voices[character]:
        engine.setProperty('voice', character_voices[character])
    else:
        engine.setProperty('voice', voices[0].id)
    
    # Reduce the voice speed by setting a lower rate (default is typically around 200)
    engine.setProperty('rate', 150)  # Adjust as needed
    
    # Create a temporary file for the audio output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tmp_filename = f.name

    engine.save_to_file(text, tmp_filename)
    engine.runAndWait()  # Wait for speech synthesis to complete
    
    # Read the audio data from the file and then remove it
    with open(tmp_filename, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_filename)
    return audio_bytes

# ---------------------------
# Utility & Pipeline Setup Functions
# ---------------------------
def load_and_prepare_documents(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split on double newlines and remove any empty chunks.
    documents = [Document(content=chunk.strip()) for chunk in text.split("\n\n") if chunk.strip()]
    # Deduplicate documents by ID.
    unique_documents = list({doc.id: doc for doc in documents}.values())
    return unique_documents

def initialize_document_store():
    return InMemoryDocumentStore()

def build_indexing_pipeline(document_store, documents, model="sentence-transformers/all-MiniLM-L6-v2"):
    pipeline = Pipeline()
    pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model=model),
        name="doc_embedder"
    )
    pipeline.add_component(
        instance=DocumentWriter(document_store=document_store),
        name="doc_writer"
    )
    pipeline.connect("doc_embedder.documents", "doc_writer.documents")
    pipeline.run({"doc_embedder": {"documents": documents}})
    return pipeline

def build_rag_pipeline(document_store, persona_prompt: str, text_embedder_model="sentence-transformers/all-MiniLM-L6-v2", llm_model="gpt-4o-mini"):
    template = [
        ChatMessage.from_system(
            f"""
This is the ground rule: You are {persona_prompt}. Answer the questions based on the provided context and respond in your distinctive style.

Context:
{{% for document in documents %}}
    {{% raw %}}{{{{ document.content }}}}{{% endraw %}}
{{% endfor %}}
Question: {{{{ question }}}}
Answer:
"""
        )
    ]
    
    rag_pipe = Pipeline()
    rag_pipe.add_component("embedder", SentenceTransformersTextEmbedder(model=text_embedder_model))
    rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
    rag_pipe.add_component("llm", OpenAIChatGenerator(model=llm_model))
    
    rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
    rag_pipe.connect("retriever", "prompt_builder.documents")
    rag_pipe.connect("prompt_builder.prompt", "llm.messages")
    
    return rag_pipe

def rag_pipeline_func(query: str, rag_pipe) -> str:
    result = rag_pipe.run({"embedder": {"text": query}, "prompt_builder": {"question": query}})
    return result["llm"]["replies"][0].text

def create_tool(rag_pipe, persona: str):
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": f"Answer in the style of {persona}.",
            }
        },
        "required": ["query"],
    }
    def tool_function(query: str):
        return {"reply": rag_pipeline_func(query, rag_pipe)}
    
    return Tool(
        name=f"{persona.lower().replace(' ', '_')}_rag_tool",
        description=f"Get responses in the style of {persona}.",
        parameters=parameters,
        function=tool_function,
    )

# ---------------------------
# Pipeline Initialization (Cached for Streamlit)
# ---------------------------
@st.cache_resource(show_spinner=False)
def initialize_sherlock_pipeline():
    setup_openai_api()
    # Path to your Sherlock text file
    sherlock_file = "The Adventures of Sherlock Holmes_clean.txt"  
    documents = load_and_prepare_documents(sherlock_file)
    document_store = initialize_document_store()
    build_indexing_pipeline(document_store, documents)
    rag_pipe = build_rag_pipeline(document_store, persona_prompt="Sherlock Holmes, the astute detective")
    sherlock_tool = create_tool(rag_pipe, "Sherlock Holmes")
    return sherlock_tool

@st.cache_resource(show_spinner=False)
def initialize_shakespeare_pipeline():
    setup_openai_api()
    # Path to your combined Shakespeare text file
    shakespeare_file = "combined_shakespeare.txt"  
    documents = load_and_prepare_documents(shakespeare_file)
    document_store = initialize_document_store()
    build_indexing_pipeline(document_store, documents)
    rag_pipe = build_rag_pipeline(document_store, persona_prompt="William Shakespeare, the eloquent playwright")
    shakespeare_tool = create_tool(rag_pipe, "William Shakespeare")
    return shakespeare_tool

# ---------------------------
# Conversation Logic
# ---------------------------
def initialize_conversation(topic: str = None):
    if topic:
        initial_message = f"Let's discuss {topic}. What are your thoughts?"
    else:
        initial_message = "Good day. Who might you be?"
    return [("Sherlock Holmes", initial_message)]

def get_reply(tool, query):
    result = tool.function(query)
    return result["reply"]

def next_turn(conversation, sherlock_tool, shakespeare_tool):
    last_speaker, last_message = conversation[-1]
    if last_speaker == "Sherlock Holmes":
        next_speaker = "William Shakespeare"
        query = last_message
        reply = get_reply(shakespeare_tool, query)
    else:
        next_speaker = "Sherlock Holmes"
        query = last_message
        reply = get_reply(sherlock_tool, query)
    conversation.append((next_speaker, reply))
    return conversation

# ---------------------------
# Streamlit App
# ---------------------------
# -- Show both characters' images side by side at the top
top_cols = st.columns(2)
with top_cols[0]:
    st.image("sherlock_face.png", caption="Sherlock Holmes", width=200)
with top_cols[1]:
    st.image("shakespeare_face.png", caption="William Shakespeare", width=200)

st.write("FileWatcherType is set to:", os.environ.get("STREAMLIT_SERVER_FILEWATCHERTYPE"))

st.title("Sherlock Holmes vs. William Shakespeare")
# st.write("This app deploys two RAG pipelines in distinct personas that converse on a topic of your choice with voice output using pyttsx3.")

# Sidebar inputs for topic and enabling voice
topic = st.sidebar.text_input("Enter a topic for conversation", value="technology")
voice_enabled = st.sidebar.checkbox("Enable Voice", value=True)
if st.sidebar.button("Reset Conversation with Topic"):
    st.session_state.conversation = initialize_conversation(topic)
    st.rerun()

# Load or initialize pipelines (cached)
sherlock_tool = initialize_sherlock_pipeline()
shakespeare_tool = initialize_shakespeare_pipeline()

if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_conversation(topic)

# Define image paths for each persona (adjust file paths as needed)
persona_images = {
    "Sherlock Holmes": "sherlock_face.png",
    "William Shakespeare": "shakespeare_face.png"
}

st.markdown("### Conversation History")
for speaker, message in st.session_state.conversation:
    cols = st.columns([1, 5])
    with cols[0]:
        if speaker in persona_images:
            st.image(persona_images[speaker], width=64)
        else:
            st.write("")
    with cols[1]:
        st.markdown(f"""<div style="padding:10px; background-color: #000000; border-radius:10px;">
            <strong>{speaker}:</strong> {message}
            </div>""", unsafe_allow_html=True)

# Play voice for the latest message if voice is enabled.
if voice_enabled:
    last_speaker, last_message = st.session_state.conversation[-1]
    st.markdown("**Voice Output:**")
    audio_bytes = generate_audio(last_speaker, last_message)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.write("Voice output unavailable.")

if st.button("Next Turn"):
    st.session_state.conversation = next_turn(
        st.session_state.conversation, sherlock_tool, shakespeare_tool
    )
    st.rerun()
