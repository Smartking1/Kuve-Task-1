import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.chatbot import LangChainChatbot
from src.vector_store import LangChainVectorStore
from config.config import VECTOR_STORE_PATH, CHATBOT_NAME, DOMAIN


# Page configuration
st.set_page_config(
    page_title=f"{CHATBOT_NAME} - {DOMAIN}",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_chatbot():
    """
    Initialize chatbot components (cached for performance)
    """
    # Load vector store
    vector_store_manager = LangChainVectorStore(store_path=VECTOR_STORE_PATH)
    
    if vector_store_manager.load():
        vectorstore = vector_store_manager.get_vectorstore()
        chatbot = LangChainChatbot(vectorstore=vectorstore)
        rag_enabled = True
    else:
        chatbot = LangChainChatbot(vectorstore=None)
        rag_enabled = False
    
    return chatbot, rag_enabled


def main():
    """
    Main Streamlit application
    """
    # Title and description
    st.title(f"🤖 {CHATBOT_NAME}")
    st.markdown(f"*Your {DOMAIN} Assistant powered by Kuve*")
    
    # Initialize chatbot
    chatbot, rag_enabled = initialize_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # RAG status
        if rag_enabled:
            st.success("✓ RAG Enabled")
            st.info("Chatbot uses your knowledge base to answer questions.")
        else:
            st.warning("⚠ RAG Disabled")
            st.info("Run `python scripts/prepare_data.py` to enable RAG.")
        
        st.divider()
        
        # Clear memory button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            chatbot.clear_memory()
            st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()
        
        st.divider()
        
        # View history
        if st.button("📜 View History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
            st.rerun()
        
        # Display history in sidebar
        if st.session_state.get('show_history', False):
            st.subheader("Conversation History")
            history = chatbot.get_chat_history()
            if history:
                for entry in history:
                    st.text(entry[:60] + "..." if len(entry) > 60 else entry)
            else:
                st.info("No history yet")
        
        st.divider()
        st.markdown("### About")
        st.markdown(f"""
        **Technology Stack:**
        - 🦜 LangChain for RAG
        - ⚡ Groq for fast LLM inference
        - 🔍 FAISS for vector search
        - 🤗 HuggingFace embeddings
        
        **Domain:** {DOMAIN}
        """)
    
    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in chatbot.chat_stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })


if __name__ == "__main__":
    main()