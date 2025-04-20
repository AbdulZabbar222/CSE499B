import streamlit as st
import fitz
import numpy as np
import pandas as pd
import time
import psutil
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def query_ollama(prompt: str, context: str, model: str = "gemma2:2b") -> str:
    url = "http://localhost:11434/api/generate"
    
    system_prompt = """You are a helpful bank assistant. Your task is to:
    1. Answer questions based ONLY on the provided context
    2. Keep responses clear and professional
    3. Use bullet points for multiple items
    4. Include relevant numbers/data when available
    5. Say \"I don't have enough information\" if context is insufficient"""

    full_prompt = f"""[SYSTEM]
    {system_prompt}

    [CONTEXT]
    {context}

    [QUESTION]
    {prompt}

    [INSTRUCTIONS]
    - Answer directly based on the above context
    - If numbers or specific data are mentioned, include them
    - Format lists as bullet points
    - Keep explanations concise but complete
    - Cite relevant sections if possible

    [RESPONSE FORMAT]
    Begin your response here..."""

    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    response_time = end_time - start_time
    
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
    
    if response.status_code == 200:
        return response.json()['response'], response_time, memory_usage
    else:
        return f"Error: {response.status_code}", response_time, memory_usage

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def compute_similarity(query, corpus_embeddings):
    """Computes cosine similarity between user query and document embeddings."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    return similarities[0]

# Streamlit UI
st.title("PDF Chatbot with Local LLM and Cosine Similarity")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Extract text and generate embeddings
    document_text = extract_text_from_pdf(uploaded_file)
    document_sentences = document_text.split(".\n")  # Split into sentences
    corpus_embeddings = model.encode(document_sentences)

    st.success("PDF successfully processed! Start asking questions.")
    
    user_input = st.text_input("Ask something about the document:")
    if user_input:
        similarities = compute_similarity(user_input, corpus_embeddings)
        best_match_idx = np.argmax(similarities)
        best_match_sentence = document_sentences[best_match_idx]
        similarity_score = similarities[best_match_idx]
        
        # Use local LLM for a better response
        llm_response, response_time, memory_usage = query_ollama(user_input, context=best_match_sentence)
        
        st.write(f"**LLM Response:** {llm_response}")
        # st.write(f"**Best Match from PDF:** {best_match_sentence}")
        st.write(f"**Cosine Similarity:** {similarity_score:.4f}")
        st.write(f"**Response Time:** {response_time:.4f} seconds")
        st.write(f"**Memory Usage:** {memory_usage:.2f} MB")