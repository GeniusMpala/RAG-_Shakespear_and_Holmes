import streamlit as st
import os
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
# Utility & Pipeline Setup Functions
# ---------------------------

def setup_openai_api():
    os.environ["OPENAI_API_KEY"] = "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"


def load_and_prepare_documents(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split into chunks on double newlines and remove empty chunks.
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
    # Create a prompt template using the persona instructions.
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
    # Wrap the RAG function.
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
def get_reply(tool, query):
    # Call the tool function with the query.
    result = tool.function(query)
    return result["reply"]

def initialize_conversation():
    # Start conversation with an initial message from Sherlock.
    return [("Sherlock Holmes", "Good day. Who might you be?")]

def next_turn(conversation, sherlock_tool, shakespeare_tool):
    # Determine the last speaker and route the query to the other persona.
    last_speaker, last_message = conversation[-1]
    if last_speaker == "Sherlock Holmes":
        next_speaker = "William Shakespeare"
        # Use Sherlock's message as query for Shakespeare.
        query = last_message
        reply = get_reply(shakespeare_tool, query)
    else:
        next_speaker = "Sherlock Holmes"
        # Use Shakespeare's message as query for Sherlock.
        query = last_message
        reply = get_reply(sherlock_tool, query)
    conversation.append((next_speaker, reply))
    return conversation

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Conversational RAG: Sherlock Holmes vs. William Shakespeare")
st.write("This app deploys two RAG pipelines in distinct personas. They converse with each other autonomously.")

# Load or initialize pipelines (cached)
sherlock_tool = initialize_sherlock_pipeline()
shakespeare_tool = initialize_shakespeare_pipeline()

# Initialize conversation in session state if not present.
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_conversation()

# Display conversation history
st.markdown("### Conversation History")
for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {message}")

# Button to continue the conversation.
if st.button("Next Turn"):
    st.session_state.conversation = next_turn(
        st.session_state.conversation, sherlock_tool, shakespeare_tool
    )
    st.rerun()

# Button to reset conversation.
if st.button("Reset Conversation"):
    st.session_state.conversation = initialize_conversation()
    st.rerun()
