"""
app.py
Streamlit front-end for the RAG-based PDF Q&A system.
Split-screen layout: Left = PDF preview, Right = Chat with PDF
"""

import os
import html as html_lib
import tempfile
import base64

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import process_pdf, get_answer

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A · Split View",
    page_icon="📄",
    layout="wide",  # Use wide layout for split screen
)

# ─── Custom CSS to hide sidebar and style split view ─────────────────────────
# Replace your existing CSS section (from line ~30 to ~120) with this:

st.markdown(
    """
    <style>
        /* Hide the default Streamlit sidebar completely */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Remove sidebar margin/padding from main content */
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        
        /* Make main container full height */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }
        
        /* Fix column heights */
        .stColumns {
            height: calc(100vh - 120px);
            align-items: stretch;
        }
        
        /* Individual column styling */
        .stColumn {
            height: 100%;
            overflow-y: auto;
        }
        
        /* Custom styling for split containers */
        .split-container {
            height: calc(100vh - 80px);
            overflow: hidden;
        }
        
        /* PDF preview container styling - FIXED SCROLLING */
        .pdf-preview-box {
            background: #1e1b2e;
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid #334155;
            height: calc(100vh - 180px);
            overflow-y: auto;
        }
        
        /* Chat container styling - FIXED SCROLLING */
        .chat-container-custom {
            background: #0f0e1a;
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid #334155;
            height: calc(100vh - 180px);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }
        
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        /* Force columns to have consistent height */
        [data-testid="column"] {
            height: calc(100vh - 150px);
            overflow-y: auto;
            scrollbar-width: thin;
        }
        
        /* Message bubbles */
        .user-msg {
            background: linear-gradient(135deg, #6d28d9, #8b5cf6);
            color: white;
            padding: 10px 16px;
            border-radius: 18px;
            border-bottom-right-radius: 4px;
            margin: 8px 0;
            max-width: 85%;
            margin-left: auto;
            word-wrap: break-word;
        }
        
        .bot-msg {
            background: #2d2a3e;
            color: #e2e8f0;
            padding: 10px 16px;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            margin: 8px 0;
            max-width: 85%;
            word-wrap: break-word;
        }
        
        .msg-label {
            font-size: 0.7rem;
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        /* Upload area styling */
        .upload-area {
            background: #1e1b2e;
            border: 2px dashed #8b5cf6;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1e1b4b, #6d28d9, #c084fc);
            padding: 1rem 2rem;
            width: 80%;
            max-width: 1300px;
            margin-left: auto;
            margin-right: auto;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 1.8rem;
            
        }
        
        .main-header p {
            color: #ffffff;
            margin: 0;
        }
        
        /* Process button */
        .stButton > button {
            background: linear-gradient(135deg, #6d28d9, #a855f7);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            width: 100%;
        }
        
        .stButton > button:hover {
            opacity: 0.9;
            color: white;
        }
        
        /* Clear chat button */
        .clear-btn > button {
            background: #2d2a3e;
            border: 1px solid #5b4b8a;
        }
        
        /* Scrollbar styling - consistent for both columns */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1e1b2e;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #8b5cf6;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a855f7;
        }
        
        /* PDF iframe container */
        .pdf-iframe-container {
            height: calc(100vh - 280px);
            min-height: 400px;
            background: #100e1a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #334155;
        }
        
        /* Make preview area scrollable */
        .preview-scroll-area {
            max-height: calc(100vh - 250px);
            overflow-y: auto;
            padding-right: 5px;
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state ────────────────────────────────────────────────────────────
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ─── Helper function to render PDF preview ───────────────────────────────────
def render_pdf_preview(pdf_bytes):
    """Display PDF preview using base64 iframe (simple, reliable)"""
    if pdf_bytes is None:
        return None
    
    # Convert PDF to base64 for embedding
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        style="width:100%; height:100%; border:none; border-radius:8px;"
        type="application/pdf"
    >
    </iframe>
    """
    return pdf_display

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📄 PDF Q&A · System</h1>
        <p>Upload PDF · Ask questions · AI answers from your document only</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── SPLIT SCREEN: LEFT (PDF Preview) + RIGHT (Chat) ─────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ======================== LEFT COLUMN: PDF PREVIEW ============================
with col_left:
    st.markdown("### 📑 PDF Preview")
    st.markdown("*Upload a PDF to see it here*")
    
    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed",
        key="pdf_uploader"
    )
    
    if uploaded_file:
        # Store PDF bytes if new file
        if st.session_state.pdf_name != uploaded_file.name:
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_bytes = uploaded_file.read()
            st.session_state.pipeline_state = None
            st.session_state.history = []
            st.session_state.pdf_processed = False
        
        # Show file info
        st.markdown(
            f"""
            <div style="background:#1e1b2e; padding:8px 12px; border-radius:10px; margin-bottom:12px;">
                📄 <strong>{st.session_state.pdf_name}</strong> · 
                {len(st.session_state.pdf_bytes) / 1024:.1f} KB
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Process button
        process_col1, process_col2 = st.columns([3, 1])
        with process_col1:
            if st.button("⚡ Process PDF", key="process_btn", use_container_width=True):
                if st.session_state.pdf_bytes:
                    with st.spinner("Processing PDF... indexing for Q&A..."):
                        try:
                            # Save bytes to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(st.session_state.pdf_bytes)
                                tmp_path = tmp.name
                            
                            state = process_pdf(tmp_path)
                            st.session_state.pipeline_state = state
                            st.session_state.pdf_processed = True
                            os.unlink(tmp_path)
                            st.success("✅ PDF processed! You can now ask questions.", icon="🎉")
                        except Exception as e:
                            st.error(f"Error processing PDF: {e}")
        
        # PDF Preview Area
        st.markdown("---")
        st.markdown("**📖 Document Preview**")
        
        preview_container = st.container()
        with preview_container:
            if st.session_state.pdf_bytes:
                # Use iframe for PDF preview (shows actual PDF content)
                base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
                preview_html = f"""
                <div style="height: 500px; background:#100e1a; border-radius:12px; overflow:hidden; border:1px solid #334155;">
                    <iframe
                        src="data:application/pdf;base64,{base64_pdf}"
                        style="width:100%; height:100%; border:none;"
                        type="application/pdf"
                    >
                        <p>Your browser cannot preview PDF. <a href="data:application/pdf;base64,{base64_pdf}">Download</a></p>
                    </iframe>
                </div>
                """
                st.markdown(preview_html, unsafe_allow_html=True)
            else:
                st.info("👈 Upload a PDF file to see preview here")
    else:
        # No file uploaded
        st.markdown(
            """
            <div class="upload-area">
                <i class="fas fa-cloud-upload-alt" style="font-size: 48px; color: #8b5cf6;"></i>
                <p style="margin-top: 12px; font-size: 1.2rem; font-weight: 500;">Drop a PDF Using Upload Button</p>
                <p style="font-size: 0.8rem; color: #a78bfa;">Supports any PDF document</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
       

# ======================== RIGHT COLUMN: CHAT WITH PDF =========================
with col_right:
    st.markdown("### 💬 Chat with your PDF")
    st.markdown("*Hey! Welcome to the conversation—let’s chat!*")
    
    # Check if PDF is processed
    if st.session_state.pipeline_state is None:
        st.warning("⚠️ Please upload and process a PDF first (click 'Process PDF' on the left).")
    else:
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            if st.session_state.history:
                for idx, (question, answer) in enumerate(st.session_state.history):
                    # User message
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                            <div style="max-width: 85%;">
                                <div class="msg-label" style="text-align: right; color: #a78bfa;">You</div>
                                <div class="user-msg">{html_lib.escape(question)}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Bot message
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                            <div style="max-width: 85%;">
                                <div class="msg-label" style="color: #c084fc;">AI</div>
                                <div class="bot-msg">{html_lib.escape(answer).replace(chr(10), '<br>')}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("💡 No questions yet. Ask something about your PDF below!")
        
        # Question input area (fixed at bottom of chat)
        st.markdown("---")
        
        question = st.text_input(
            "Ask a question",
            placeholder="e.g., Ask questions based on the document content",
            key="question_input",
            label_visibility="collapsed"
        )
        
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("📤 Ask", use_container_width=True, key="ask_btn")
        with col_clear:
            clear_button = st.button("🗑 Clear chat", use_container_width=True, key="clear_btn")
        
        if ask_button:
            if question.strip():
                with st.spinner("🤔 Searching document..."):
                    try:
                        answer = get_answer(question, st.session_state.pipeline_state)
                        st.session_state.history.append((question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question.")
        
        if clear_button:
            st.session_state.history = []
            st.rerun()

# Optional footer
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; margin-top: 1rem; color: #5a5270; font-size: 0.75rem;">
        🔒 Answers are generated only from your uploaded PDF document.
    </div>
    """,
    unsafe_allow_html=True,
) 