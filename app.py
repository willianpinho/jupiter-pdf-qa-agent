"""Streamlit application for PDF QA agent.

This is the main entry point for the application.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.agent import PDFQAAgent

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found. Please create a .env file with your API key.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Jupiter PDF QA Agent",
    page_icon="🪐",
    layout="wide",
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = PDFQAAgent()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# App header
st.title("🪐 Project Jupiter — PDF QA Agent")
st.markdown(
    """
    **Multi-turn question answering over uploaded PDFs** using LangGraph + Streamlit.
    
    Upload a PDF document and ask questions about its content!
    """
)

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF document to ask questions about",
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    # Read file bytes
                    file_bytes = uploaded_file.read()
                    
                    # Process with document store
                    result = st.session_state.agent.document_store.process_pdf(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name
                    )
                    
                    # Update uploaded files list
                    if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                            "pages": result["num_pages"],
                            "chunks": result["num_chunks"]
                        })
                        st.success(
                            f"✅ Processed: {uploaded_file.name}\n\n"
                            f"📄 {result['num_pages']} pages → {result['num_chunks']} chunks"
                        )
                    else:
                        st.info("File already uploaded")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    st.divider()
    
    # Display uploaded documents
    st.subheader("Uploaded Documents")
    
    if st.session_state.uploaded_files:
        for doc in st.session_state.uploaded_files:
            st.write(f"📄 {doc['name']}")
            st.caption(f"Size: {doc['size']:,} bytes")
    else:
        st.info("No documents uploaded yet")
    
    st.divider()
    
    # Session controls
    st.subheader("Session Controls")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("Reset Session"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.rerun()


# Main chat interface
st.header("💬 Chat")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass conversation history to maintain context
                response = st.session_state.agent.run(
                    prompt,
                    conversation_history=st.session_state.chat_history
                )
                st.markdown(response)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
