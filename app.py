import streamlit as st
import faiss
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import uuid
import time
from datetime import datetime
import threading
from typing import Dict, List, Any
from llama_cpp import Llama
import os
import gradio as gr
os.environ['CURL_CA_BUNDLE'] = ''

# Configure Streamlit for multi-user

st.set_page_config(
    page_title="Hipotronics Bot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ConcurrentRAGSystem:
    """Thread-safe RAG system for multiple concurrent users"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.active_users = {}
        self.query_count = 0
        

    def load_models(self):
        """Load models once and cache them (shared across all users)"""
        # Load embedding model
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load qa generation model

        generator = Llama(model_path = "models/tinyllama.gguf", n_ctx=512)


        
        return embedding_model, generator
    
    def load_knowledge_base(self):
        """Load pre-built FAISS index and documents (shared across users)"""
        try:
            # Load FAISS index
            index = faiss.read_index("troubleshooting_index.bin")
            
            # Load document chunks
            with open("troubleshooting_chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
                
            # Load metadata
            with open("troubleshooting_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                
            return index, chunks, metadata
            
        except FileNotFoundError:
            st.error("Knowledge base not found. Please upload troubleshooting documents first.")
            return None, None, None
    
    def get_session_info(self):
        """Get or create session information for current user"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.user_name = f"User_{st.session_state.session_id}"
            st.session_state.messages = []
            st.session_state.query_history = []
            st.session_state.session_start = datetime.now()
            
        return st.session_state.session_id
    
    def update_active_users(self, session_id: str):
        """Track active users for monitoring"""
        with self.lock:
            self.active_users[session_id] = {
                'last_activity': datetime.now(),
                'query_count': len(st.session_state.get('query_history', []))
            }
            
            # Clean up inactive users (older than 30 minutes)
            cutoff_time = datetime.now()
            inactive_users = [
                uid for uid, info in self.active_users.items() 
                if (cutoff_time - info['last_activity']).seconds > 1800
            ]
            for uid in inactive_users:
                del self.active_users[uid]
    
    def search_documents(self, query: str, top_k: int = 3, top_k_candidates: int = 10):
        embedding_model, _ = self.load_models()
        index, chunks, metadata = self.load_knowledge_base()
    
        if index is None:
            return []

        query_embedding = embedding_model.encode([query])
        scores, indices = index.search(query_embedding.astype('float32'), top_k_candidates)

        candidate_chunks = []
        candidate_meta = []
        valid_indices = []

        for idx in indices[0]:
            if idx < len(chunks):
                candidate_chunks.append(chunks[idx])
                candidate_meta.append(metadata[idx])
                valid_indices.append(idx)

        if not candidate_chunks:
            return []

        # Re-rank using cosine similarity
        chunk_embeddings = embedding_model.encode(candidate_chunks)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            idx = valid_indices[i]
            results.append({
                'content': chunks[idx],
                'metadata': metadata[idx] if idx < len(metadata) else {},
                'similarity': float(similarities[i]),
                'source': metadata[idx].get('source', 'Unknown') if idx < len(metadata) else 'Unknown'
            })

        return results
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
      """Generate response using retrieved context"""
      if not context_docs:
        return "I couldn't find relevant information in the troubleshooting guide for your question. Please try rephrasing or contact technical support."

      # Use top 2 most relevant context docs
      context = "\n\n".join([
          f"From {doc['source']}: {doc['content']}" 
          for doc in context_docs[:2]
      ])

      # Create instruction-style prompt for FLAN-T5
      prompt = f"""
      ###Instruction:
      You are a trouble shooting assistant. Your goal is to provide Response as
      accurately as possible based on the instructions and Context provided using the Question.
      
      ###Context:
      {context}

      ###Question: {query}

      ###Response:
      """

      # Load generator
      _, generator = self.load_models()

      print(context)

      try:
          result = generator(prompt, max_tokens=1000, stop=["###"])
          print(result)
          answer = result["choices"][0]["text"].strip()
          print(answer)
          response_template = f"""**Answer:** {answer}

  **Sources:** {', '.join([doc['source'] for doc in context_docs[:2]])}
  """
      except Exception as e:
          print(e)
          st.exception(e)
          response_template = "Sorry, I couldn't generate a response due to an internal error."

      return response_template

    
    def is_repeated_query(self, query: str, threshold: float = 0.85) -> bool:
        """Check if query is too similar to recent queries"""
        if not st.session_state.get('query_history'):
            return False
            
        embedding_model, _ = self.load_models()
        query_embedding = embedding_model.encode([query])
        
        # Check against last 5 queries
        recent_queries = st.session_state.query_history[-5:]
        for prev_query in recent_queries:
            prev_embedding = embedding_model.encode([prev_query])
            similarity = np.dot(query_embedding[0], prev_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(prev_embedding[0])
            )
            
            if similarity > threshold:
                return True
        
        return False

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    return ConcurrentRAGSystem()

rag_system = get_rag_system()

def chatbot_fn(message, history):
    # Check for repeated query
    if rag_system.is_repeated_query(message):
        return "You've asked a similar question recently. Please rephrase."

    # Search documents
    relevant_docs = rag_system.search_documents(message)

    # Generate response
    response = rag_system.generate_response(message, relevant_docs)
    
    return response

# Launch Gradio chat interface
chat = gr.ChatInterface(fn=chatbot_fn, title="Hipotronics Assistant")
chat.launch()