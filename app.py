import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GMT_THREADS'] = '1'

import streamlit as st
import json
import re
from agent.core import get_agent_executor
from agent.tools import (
    set_current_segy_path,
    set_current_miniseed_path,
    set_current_hdf5_path,
    set_current_sac_path,
    set_current_lang,
    run_continuous_monitoring,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from deploy.i18n import t


def _detect_prompt_lang(text: str) -> str:
    """Heuristic language detection for current user turn."""
    if not text:
        return "en"
    # Any CJK character -> Chinese
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


class SafeStreamlitCallbackHandler(StreamlitCallbackHandler):
    """Streamlit callback handler that degrades token streaming safely."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_stream_enabled = True

    def on_llm_new_token(self, token: str, **kwargs):
        if not self._token_stream_enabled:
            return
        try:
            super().on_llm_new_token(token, **kwargs)
        except RecursionError:
            # Keep the thought containers and tool trace, but stop per-token updates.
            self._token_stream_enabled = False
        except Exception:
            self._token_stream_enabled = False

# Page Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PNG = os.path.join(BASE_DIR, "resources", "QuakeCore.png")
LOGO_SVG = os.path.join(BASE_DIR, "resources", "QuakeCore_Logo.svg")
USER_AVATAR = os.path.join(BASE_DIR, "resources", "chuanjun.jpg")


def render_logo_svg(width=40):
    """Render SVG logo inline."""
    if os.path.exists(LOGO_SVG):
        with open(LOGO_SVG, "r") as f:
            svg_content = f.read()
        return f'<div style="width:{width}px; height:{width}px;">{svg_content}</div>'
    return ""

st.set_page_config(
    page_title="QuakeCore Engine",
    page_icon=LOGO_PNG if os.path.exists(LOGO_PNG) else "🌊",
    layout="wide",
)

# ChatGPT Light Theme CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling - ChatGPT style */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        width: 280px !important;
    }

    [data-testid="stSidebarNav"] {
        padding: 0.5rem !important;
    }

    /* Sidebar content */
    .sidebar-content {
        padding: 1rem !important;
    }

    /* Sidebar buttons - minimal style */
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: var(--bg-tertiary) !important;
    }

    /* Delete button - completely frameless */
    [data-testid="stSidebar"] button[key^="del_conv"] {
        background: transparent !important;
        border: 0px !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 0px !important;
        margin: 0px !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: normal !important;
        border-radius: 0px !important;
        min-width: 0px !important;
    }

    [data-testid="stSidebar"] button[key^="del_conv"]:hover {
        color: #ff4444 !important;
        background: transparent !important;
        border: 0px !important;
    }

    .qc-sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
    }

    /* New chat button */
    .new-chat-btn {
        width: 100% !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    .new-chat-btn:hover {
        background: var(--bg-tertiary) !important;
    }

    /* Chat history list */
    .chat-history {
        margin-top: 1rem !important;
    }

    .chat-history-item {
        padding: 0.6rem 0.75rem !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        cursor: pointer !important;
        transition: background 0.15s ease !important;
        margin-bottom: 0.25rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: 0.9rem !important;
    }

    .chat-history-item:hover {
        background: var(--bg-tertiary) !important;
    }

    .chat-history-item.active {
        background: var(--bg-tertiary) !important;
        font-weight: 500 !important;
    }

    .chat-history-item span {
        display: block !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Delete chat button */
    .delete-chat-btn {
        opacity: 0.5 !important;
        transition: opacity 0.15s !important;
        float: right !important;
        padding: 0.2rem !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
    }

    .delete-chat-btn:hover {
        opacity: 1 !important;
    }

    /* Sidebar footer */
    .sidebar-footer {
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        padding: 1rem !important;
        border-top: 1px solid var(--border-color) !important;
        background: var(--bg-secondary) !important;
    }

    /* ChatGPT Light Theme */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f7f7f8;
        --bg-tertiary: #ececf1;
        --border-color: #e5e5e5;
        --text-primary: #1a1a1a;
        --text-secondary: #6e6e6e;
        --text-muted: #9a9a9a;
        --accent: #10a37f;
        --accent-hover: #0d8a6a;
        --assistant-bubble: #ffffff;
        --user-bubble: #10a37f;
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    .main .block-container {
        max-width: 800px;
        padding: 1rem 1rem 6.5rem;
        margin: 0 auto;
    }

    /* User message - right side */
    [data-testid="stChatMessageUser"] {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        gap: 0.5rem !important;
    }

    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
        background: var(--user-bubble) !important;
        border-radius: 18px 4px 18px 18px !important;
        padding: 0.7rem 1rem !important;
        color: white !important;
        max-width: 80% !important;
        border: none !important;
    }

    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] div {
        color: white !important;
    }

    /* Assistant message - left side */
    [data-testid="stChatMessageAssistant"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        gap: 0.5rem !important;
    }

    [data-testid="stChatMessageAssistant"] [data-testid="stChatMessageContent"] {
        background: var(--assistant-bubble) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px 18px 18px 4px !important;
        padding: 0.7rem 1rem !important;
        max-width: 80% !important;
        color: var(--text-primary) !important;
    }

    /* Chat input - clean design */
    .stChatInput {
        background: #ffffff !important;
    }

    .stChatInput > div {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 24px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: all 0.2s ease !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 2px 16px rgba(16, 163, 127, 0.15) !important;
    }

    .stChatInput textarea {
        color: #1a1a1a !important;
        background: transparent !important;
    }

    .stChatInput textarea::placeholder {
        color: #9a9a9a !important;
        opacity: 1 !important;
    }

    /* Ensure chat input text is always dark */
    [data-testid="stChatInput"] textarea {
        color: #1a1a1a !important;
        background: transparent !important;
    }

    /* Chat input inner container - white background */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] .stTextArea {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Target all possible nested elements */
    .stChatInput div,
    .stChatInput section,
    .stChatInput label {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Streamlit theme override for chat input */
    section[data-testid="stChatInput"] {
        background-color: #ffffff !important;
    }

    section[data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
    }

    /* Chat input send button icon - make it green like ChatGPT */
    [data-testid="stChatInput"] button {
        background: #10a37f !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #0d8a6a !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: white !important;
        color: white !important;
    }

    /* Text area within chat input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] .stTextArea textarea {
        background: transparent !important;
        color: #1a1a1a !important;
        caret-color: #1a1a1a !important;
    }

    /* Buttons - minimal style */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        padding: 0.45rem 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: #d0d0d0 !important;
    }

    /* Status widget - thinking animation */
    div[data-testid="stStatusWidget"] {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="stStatusWidget"] p {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }

    /* Thinking dots animation */
    .thinking-dots {
        display: inline-flex;
        gap: 4px;
        padding: 0.5rem 0;
    }

    .thinking-dots span {
        width: 8px;
        height: 8px;
        background: var(--accent);
        border-radius: 50%;
        animation: thinking 1.4s ease-in-out infinite;
    }

    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes thinking {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-6px);
            opacity: 1;
        }
    }

    /* Fade in animation for responses */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stChatMessageContent"] {
        animation: fadeInUp 0.3s ease-out;
    }

    /* Images in chat */
    .stImage img {
        border-radius: 12px;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Captions */
    [data-testid="stCaption"] {
        color: var(--text-muted) !important;
    }

    /* Expander - clean style */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        transition: all 0.15s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background: var(--bg-tertiary) !important;
    }

    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* Dialog - clean style */
    [data-testid="stDialog"] {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
    }

    [data-testid="stMarkdownContainer"] a {
        color: var(--accent) !important;
    }

    [data-testid="stMarkdownContainer"] code {
        background: var(--bg-secondary);
        color: var(--text-primary) !important;
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.88em;
    }

    [data-testid="stMarkdownContainer"] pre {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        background: var(--bg-secondary);
    }

    [data-testid="stMarkdownContainer"] pre code {
        background: transparent !important;
    }

    [data-testid="stChatMessageAvatarUser"] img,
    [data-testid="stChatMessageAvatarAssistant"] img {
        border-radius: 50% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Scrollbar - minimal */
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {background: #d1d1d1; border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #b0b0b0;}

    /* Selectbox/dropdown - clean style */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    .stSelectbox [data-baseweb="select"] {
        background: #ffffff !important;
    }

    .stSelectbox [data-baseweb="popover"] {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
    }

    .stSelectbox [data-baseweb="option"] {
        background: #ffffff !important;
        color: var(--text-primary) !important;
    }

    .stSelectbox [data-baseweb="option"]:hover {
        background: var(--bg-secondary) !important;
    }

    /* Text input - clean style */
    .stTextInput > div > div {
        background: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }

    .stTextInput input {
        color: var(--text-primary) !important;
        background: transparent !important;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    /* Home hero */
    .qc-home-hero-text {
        padding-top: 0.1rem;
        text-align: center;
    }

    .qc-home-title {
        font-size: 2rem;
        font-weight: 650;
        color: #1a1a1a;
        line-height: 1.2;
        margin: 0 0 0.45rem 0;
    }

    .qc-home-subtitle {
        color: #6e6e6e;
        font-size: 1rem;
        margin: 0 0 0.4rem 0;
        line-height: 1.5;
    }

    .qc-home-formats {
        color: #9a9a9a;
        font-size: 0.86rem;
        margin: 0;
        line-height: 1.45;
    }

    .qc-home-actions-spacer {
        height: 0.45rem;
    }

    @media (max-width: 900px) {
        .qc-home-title {
            font-size: 1.6rem;
        }
    }

    /* Info/success/warning/error boxes */
    .stAlert {
        border-radius: 12px !important;
    }

    /* File upload area */
    [data-testid="stFileUploadDropzone"] {
        background: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 16px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent) !important;
        background: rgba(16, 163, 127, 0.03) !important;
    }

    /* Spinner animation */
    [data-testid="stSpinner"] {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Loading shimmer effect */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .loading-shimmer {
        background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 8px;
    }

    /* Blur effect for processing */
    .processing-blur {
        filter: blur(2px);
        transition: filter 0.3s ease;
    }

    /* Main content area - ensure white background */
    section[data-testid="stMainBlockContainer"] {
        background: #ffffff !important;
    }

    /* Bottom chat area background */
    .stApp > div:first-child {
        background: #ffffff !important;
    }

    /* Dialog / Modal styling */
    [data-testid="stDialog"] {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
    }

    [data-testid="stDialog"] h1,
    [data-testid="stDialog"] h2,
    [data-testid="stDialog"] h3,
    [data-testid="stDialog"] h4,
    [data-testid="stDialog"] p,
    [data-testid="stDialog"] span,
    [data-testid="stDialog"] label {
        color: #1a1a1a !important;
    }

    [data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
        color: #1a1a1a !important;
    }

    /* App body background */
    .stApp {
        background-color: #ffffff !important;
    }

    /* Main content background */
    .main {
        background: #ffffff !important;
    }

    /* Streamlit form and input backgrounds */
    .stForm,
    .stForm > div {
        background: #ffffff !important;
    }

    /* Multi-select and dropdown styling */
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }

    /* Date input and number input */
    .stNumberInput > div > div,
    .stDateInput > div > div {
        background: #ffffff !important;
        border-color: #e5e5e5 !important;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: #e5e5e5 !important;
    }

    /* Checkbox styling */
    .stCheckbox > label > div {
        background: #ffffff !important;
    }

    /* Radio button styling */
    .stRadio > div {
        background: #ffffff !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: #e5e5e5 !important;
    }

    /* Metric styling */
    .stMetric {
        background: #ffffff !important;
    }

    /* Table styling */
    .stTable {
        background: #ffffff !important;
    }

    /* JSON styling */
    .stJson {
        background: #ffffff !important;
    }

    /* Code block styling */
    .stCodeBlock {
        background: #f7f7f8 !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 10px !important;
    }

    /* Divider */
    hr {
        border-color: #e5e5e5 !important;
    }

    /* Tooltip */
    .stTooltipIcon {
        color: #6e6e6e !important;
    }

    /* Download button */
    .stDownloadButton {
        background: #ffffff !important;
    }

    /* Color picker */
    .stColorPicker > div > div {
        background: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_config" not in st.session_state:
    st.session_state.agent_config = {"provider": "deepseek", "model_name": "deepseek-v4-flash", "api_key": os.getenv("DEEPSEEK_API_KEY", ""), "base_url": "https://api.deepseek.com"}
if "agent_code_fingerprint" not in st.session_state:
    st.session_state.agent_code_fingerprint = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "active_loaded_conversation_id" not in st.session_state:
    st.session_state.active_loaded_conversation_id = None
if "pending_continuous_request" not in st.session_state:
    st.session_state.pending_continuous_request = None
if "pending_turn" not in st.session_state:
    st.session_state.pending_turn = None
lang = st.session_state.lang
set_current_lang(lang)


def _format_conversation_title(messages, fallback="新对话"):
    """Build a stable sidebar title from the first user message."""
    for msg in messages or []:
        if msg.get("role") == "user":
            title = str(msg.get("content", "")).replace("\n", " ").strip()
            if not title:
                break
            return title[:30] + ("..." if len(title) > 30 else "")
    return fallback


def _save_current_conversation():
    """Persist the current in-memory message list into the active conversation."""
    conv_id = st.session_state.current_conversation_id
    if not conv_id:
        return
    if conv_id not in st.session_state.conversations:
        st.session_state.conversations[conv_id] = {"title": "新对话", "messages": []}
    st.session_state.conversations[conv_id]["messages"] = list(st.session_state.messages)
    st.session_state.conversations[conv_id]["title"] = _format_conversation_title(
        st.session_state.messages,
        st.session_state.conversations[conv_id].get("title", "新对话"),
    )


def _ensure_active_conversation():
    """Create an active conversation if none exists, preserving current messages."""
    if st.session_state.current_conversation_id:
        return st.session_state.current_conversation_id
    import uuid

    new_id = str(uuid.uuid4())[:8]
    st.session_state.current_conversation_id = new_id
    st.session_state.conversations[new_id] = {
        "title": _format_conversation_title(st.session_state.messages, "新对话"),
        "messages": list(st.session_state.messages),
    }
    st.session_state.active_loaded_conversation_id = new_id
    return new_id


def _looks_like_confirmation(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return normalized in {
        "继续", "确认", "确认执行", "继续执行", "继续吧", "yes", "y", "ok", "okay",
        "confirm", "go ahead", "proceed"
    }


def _parse_tool_input(tool_input):
    if isinstance(tool_input, dict):
        return dict(tool_input)
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _build_continuous_resume_input(payload, tool_input=None):
    tool_input = _parse_tool_input(tool_input)
    resume = {}

    region = payload.get("region") if isinstance(payload, dict) else {}
    if not isinstance(region, dict):
        region = {}

    start_time = payload.get("start_time")
    end_time = payload.get("end_time")
    if start_time:
        resume["start"] = start_time
    if end_time:
        resume["end"] = end_time

    for key in ("min_lat", "max_lat", "min_lon", "max_lon", "network", "client", "catalog"):
        value = region.get(key)
        if value is not None:
            resume[key] = value

    if region.get("mode") == "place" and isinstance(region.get("place"), dict):
        place = region["place"]
        place_name = place.get("name") or place.get("en_name")
        if place_name:
            resume["place"] = place_name

    passthrough_keys = (
        "channel",
        "data_dir",
        "compare_with_catalog",
        "min_magnitude",
        "time_tolerance",
        "min_picks",
        "merge_window",
        "peak_threshold",
        "date",
        "hours",
        "duration_hours",
    )
    for key in passthrough_keys:
        value = tool_input.get(key)
        if value not in (None, ""):
            resume[key] = value

    resume["confirm"] = True
    resume["force"] = True
    return resume


def _store_pending_continuous_request(response):
    pending = None
    for action, obs in response.get("intermediate_steps", []) or []:
        obs_text = obs if isinstance(obs, str) else ""
        if "requires_confirmation" not in obs_text and "status" not in obs_text:
            continue
        try:
            obs_data = json.loads(obs_text) if obs_text else {}
        except Exception:
            obs_data = {}
        if isinstance(obs_data, dict) and obs_data.get("status") == "requires_confirmation":
            tool_input = _parse_tool_input(getattr(action, "tool_input", {}))
            pending = {
                "tool": getattr(action, "tool", ""),
                "tool_input": tool_input,
                "prompt": obs_data.get("task_summary") or "",
                "payload": obs_data,
                "resume_input": _build_continuous_resume_input(obs_data, tool_input),
            }
            break
    st.session_state.pending_continuous_request = pending


def _run_pending_continuous_request():
    pending = st.session_state.get("pending_continuous_request")
    if not pending:
        return None
    tool_input = dict(pending.get("resume_input") or pending.get("tool_input") or {})
    if not tool_input:
        payload = pending.get("payload") or {}
        tool_input = _build_continuous_resume_input(payload, pending.get("tool_input"))
    result = run_continuous_monitoring.invoke(tool_input)
    st.session_state.pending_continuous_request = None
    return result


def _format_continuous_resume_result(raw_result):
    """Turn a continuous monitoring tool result into a short assistant response."""
    if isinstance(raw_result, str):
        try:
            data = json.loads(raw_result)
        except Exception:
            return raw_result
    elif isinstance(raw_result, dict):
        data = raw_result
    else:
        return str(raw_result)

    if not isinstance(data, dict):
        return str(raw_result)

    lines = []
    if data.get("status") == "success":
        lines.append("已继续执行上一轮连续监测任务。")
        if data.get("task_summary"):
            lines.append(f"- {data['task_summary']}")
        if data.get("n_stations") is not None:
            lines.append(f"- 台站数：{data['n_stations']}")
        if data.get("n_picks") is not None:
            lines.append(f"- 拾取数：{data['n_picks']}")
        if data.get("n_events_detected") is not None:
            lines.append(f"- 检测事件数：{data['n_events_detected']}")
        if data.get("suggested_request"):
            lines.append(f"- 建议：{data['suggested_request']}")
        if data.get("recommendation"):
            lines.append(f"- 说明：{data['recommendation']}")
        if data.get("location_map"):
            lines.append(f"![定位图]({data['location_map']})")
        return "\n".join(lines)

    if data.get("status") == "requires_confirmation":
        lines.append("上一轮任务仍需要确认。")
        if data.get("task_summary"):
            lines.append(f"- {data['task_summary']}")
        if data.get("recommendation"):
            lines.append(f"- {data['recommendation']}")
        if data.get("suggested_request"):
            lines.append(f"- {data['suggested_request']}")
        return "\n".join(lines)

    return json.dumps(data, ensure_ascii=False, indent=2)


def _invoke_agent_turn(content, lang):
    """Invoke the agent for one user turn and return a structured result."""
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    status = st.status(t("thinking", lang), expanded=True)
    callback = SafeStreamlitCallbackHandler(
        status.container(),
        max_thought_containers=20,
        expand_new_thoughts=True,
        collapse_completed_thoughts=False,
    )
    text_lower = str(content or "").lower()
    if any(k in text_lower for k in ("monitor", "continuous", "download", "waveform", "连续", "监测", "下载", "波形")):
        status.container().markdown(
            "⏳ 连续监测任务执行中：下载波形 → AI拾取 → 事件关联 → 事件定位 → 绘图（台站多时可能需要几分钟）..."
        )

    try:
        response = st.session_state.agent.invoke(
            {"input": content, "chat_history": chat_history},
            config={"callbacks": [callback]},
        )
    except Exception:
        status.update(label=t("error", lang), state="error")
        raise

    _store_pending_continuous_request(response)

    answer = response["output"]
    # Filter parser-retry artifacts from ReAct output.
    if isinstance(answer, str):
        noisy_lines = {
            "Invalid Format: Missing 'Action:' after 'Thought:'",
            "Invalid Format: Missing 'Action:' after 'Thought:' ",
            ":thinking_face: Thinking...",
            "🤔 Thinking...",
        }
        clean_lines = []
        for line in answer.splitlines():
            stripped = line.strip()
            if stripped in noisy_lines:
                continue
            if stripped.startswith("⚠️ Parsing error: Invalid Format:"):
                continue
            clean_lines.append(line)
        answer = "\n".join(clean_lines).strip()

    # If ReAct final text is empty/noisy but the monitoring tool already succeeded,
    # synthesize a stable final answer directly from tool payload.
    payload = _extract_continuous_result_payload(response)
    synthesized = _format_continuous_payload_as_answer(payload)
    if synthesized:
        if not answer:
            answer = synthesized
        elif "Missing 'Action:'" in answer or "Parsing error" in answer:
            answer = synthesized
        elif "Agent stopped" in answer or "iteration limit" in answer.lower():
            answer = synthesized
    steps_text = ""
    if response.get("intermediate_steps"):
        for i, (action, obs) in enumerate(response["intermediate_steps"], 1):
            tool_input = action.tool_input
            if isinstance(tool_input, dict):
                tool_input = json.dumps(tool_input, ensure_ascii=False)
            obs_text = obs
            if isinstance(obs, str):
                try:
                    obs_data = json.loads(obs)
                except Exception:
                    obs_data = None
                if isinstance(obs_data, dict):
                    summary_parts = []
                    if obs_data.get("task_summary"):
                        summary_parts.append(str(obs_data["task_summary"]))
                    if obs_data.get("progress_summary"):
                        summary_parts.append(str(obs_data["progress_summary"]))
                    estimate = obs_data.get("estimate")
                    if isinstance(estimate, dict):
                        estimate_parts = []
                        if estimate.get("station_count") is not None:
                            estimate_parts.append(f"台站数 {estimate['station_count']}")
                        if estimate.get("estimated_mb") is not None:
                            estimate_parts.append(f"约 {estimate['estimated_mb']} MB")
                        if estimate.get("estimated_gb") is not None:
                            estimate_parts.append(f"约 {estimate['estimated_gb']} GB")
                        if estimate_parts:
                            summary_parts.append("，".join(estimate_parts))
                    progress_items = obs_data.get("progress")
                    if isinstance(progress_items, list) and progress_items:
                        tail = progress_items[-5:]
                        progress_lines = []
                        for item in tail:
                            if not isinstance(item, dict):
                                continue
                            msg = str(item.get("message", "")).strip()
                            if not msg:
                                continue
                            progress_lines.append(f"- {msg}")
                        if progress_lines:
                            summary_parts.append("下载进度：\n" + "\n".join(progress_lines))
                    if summary_parts:
                        obs_text = "\n".join(summary_parts)
            steps_text += f"{i}. **{action.tool}**\n   → {obs_text}\n\n"

    img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    response_images = [p.strip() for p in img_pattern.findall(answer)]
    if response.get("intermediate_steps"):
        for _, obs in response["intermediate_steps"]:
            if isinstance(obs, str):
                for p in img_pattern.findall(obs):
                    if p.strip() not in response_images:
                        response_images.append(p.strip())
                try:
                    obs_data = json.loads(obs)
                except Exception:
                    obs_data = None
                if isinstance(obs_data, dict):
                    for key in ("location_map", "location_3view"):
                        path = obs_data.get(key)
                        if isinstance(path, str) and path.strip() and path.strip() not in response_images:
                            response_images.append(path.strip())

    text_only = img_pattern.sub('', answer).strip()
    return {
        "response": response,
        "answer": answer,
        "steps_text": steps_text,
        "response_images": response_images,
        "text_only": text_only,
        "status": status,
        "callback": callback,
    }


def _extract_monitor_progress(response):
    """Extract continuous-monitoring progress from intermediate tool observations."""
    if not isinstance(response, dict):
        return None
    steps = response.get("intermediate_steps") or []
    best = None
    for action, obs in steps:
        if getattr(action, "tool", "") not in {"run_continuous_monitoring", "download_continuous_waveforms"}:
            continue
        if not isinstance(obs, str):
            continue
        try:
            payload = json.loads(obs)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        progress_items = payload.get("progress")
        if not isinstance(progress_items, list) or not progress_items:
            continue

        total = None
        downloaded = 0
        failed = 0
        for item in progress_items:
            if not isinstance(item, dict):
                continue
            if item.get("total") is not None:
                total = int(item.get("total"))
            if item.get("downloaded") is not None:
                downloaded = int(item.get("downloaded"))
            if item.get("failed") is not None:
                failed = int(item.get("failed"))
        if total is None:
            # Fallback: infer total from estimate if available.
            estimate = payload.get("estimate") or {}
            if isinstance(estimate, dict) and estimate.get("station_count") is not None:
                total = int(estimate.get("station_count"))

        best = {
            "downloaded": downloaded,
            "failed": failed,
            "total": total,
            "summary": payload.get("progress_summary"),
            "tail": progress_items[-5:],
        }
    return best


def _extract_continuous_result_payload(response):
    """Get the latest continuous-monitoring JSON payload from intermediate steps."""
    if not isinstance(response, dict):
        return None
    steps = response.get("intermediate_steps") or []
    payload = None
    for action, obs in steps:
        if getattr(action, "tool", "") != "run_continuous_monitoring":
            continue
        if not isinstance(obs, str):
            continue
        try:
            data = json.loads(obs)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("status") in {"success", "error"}:
            payload = data
    return payload


def _format_continuous_payload_as_answer(payload):
    """Build a concise markdown answer from continuous monitoring payload."""
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    if status not in {"success", "error"}:
        return None

    lines = []
    if status == "success":
        lines.append("连续监测已完成。")
        if payload.get("task_summary"):
            lines.append(f"- 任务：{payload['task_summary']}")
        if payload.get("n_stations") is not None:
            lines.append(f"- 台站数：{payload['n_stations']}")
        if payload.get("n_picks") is not None:
            lines.append(f"- 拾取数：{payload['n_picks']}")
        if payload.get("n_events_detected") is not None:
            lines.append(f"- 检测事件数：{payload['n_events_detected']}")
        if payload.get("progress_summary"):
            lines.append(f"- 进度：{payload['progress_summary']}")
        if payload.get("location_map"):
            lines.append(f"![地震定位图]({payload['location_map']})")
        return "\n".join(lines)

    err = payload.get("error") or "连续监测执行失败"
    lines.append(f"连续监测失败：{err}")
    if payload.get("hint"):
        lines.append(f"- 提示：{payload['hint']}")
    return "\n".join(lines)

# ==================== Layout ====================
# Sidebar content - chat history
with st.sidebar:
    st.markdown('<div class="qc-sidebar-brand">QuakeCore</div>', unsafe_allow_html=True)

    st.divider()

    # New Chat button
    if st.button("➕ 新对话", key="new_chat_btn", width="stretch"):
        _save_current_conversation()
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[new_id] = {"title": "新对话", "messages": []}
        st.session_state.current_conversation_id = new_id
        st.session_state.active_loaded_conversation_id = new_id
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Chat history
    st.markdown("**历史对话**")
    if st.session_state.conversations:
        conv_list = list(reversed(list(st.session_state.conversations.items())))
        for i, (conv_id, conv_data) in enumerate(conv_list, 1):
            is_active = st.session_state.current_conversation_id == conv_id
            label = f"**{i}. {conv_data['title']}**" if is_active else f"{i}. {conv_data['title']}"
            row_cols = st.columns([6.5, 1], gap="small")
            with row_cols[0]:
                if st.button(label, key=f"conv_{conv_id}", width="stretch"):
                    _save_current_conversation()
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.active_loaded_conversation_id = conv_id
                    st.session_state.messages = list(conv_data["messages"])
                    st.rerun()
            with row_cols[1]:
                if st.button("✕", key=f"del_conv_{conv_id}", help="删除对话", type="tertiary", width="stretch"):
                    st.session_state.conversations.pop(conv_id, None)

                    if st.session_state.current_conversation_id == conv_id:
                        if st.session_state.conversations:
                            next_id = list(st.session_state.conversations.keys())[-1]
                            st.session_state.current_conversation_id = next_id
                            st.session_state.active_loaded_conversation_id = next_id
                            st.session_state.messages = list(st.session_state.conversations[next_id]["messages"])
                        else:
                            st.session_state.current_conversation_id = None
                            st.session_state.active_loaded_conversation_id = None
                            st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("暂无历史对话")

    st.divider()

    # Model info
    config = st.session_state.get("agent_config", {})
    model_name = config.get("model_name", "未配置")
    st.caption(f"模型: {model_name}")

# ==================== Load current conversation ====================
if st.session_state.current_conversation_id:
    if st.session_state.active_loaded_conversation_id != st.session_state.current_conversation_id:
        conv = st.session_state.conversations.get(st.session_state.current_conversation_id)
        if conv and conv["messages"]:
            st.session_state.messages = list(conv["messages"])
        st.session_state.active_loaded_conversation_id = st.session_state.current_conversation_id

if not st.session_state.current_conversation_id and st.session_state.messages:
    _ensure_active_conversation()

# ==================== Main content header ====================
is_homepage = not st.session_state.messages
if is_homepage:
    hero_cols = st.columns([0.8, 5.6, 1.6], gap="medium")
    with hero_cols[0]:
        logo_svg = render_logo_svg(52)
        if logo_svg:
            st.markdown(logo_svg, unsafe_allow_html=True)
        elif os.path.exists(LOGO_PNG):
            st.image(LOGO_PNG, width=52)

    with hero_cols[1]:
        _lang = st.session_state.lang
        st.markdown(
            f"""
            <div class="qc-home-hero-text">
                <div class="qc-home-title">{t("app_title", _lang)}</div>
                <p class="qc-home-subtitle">{t("app_subtitle", _lang)}</p>
                <p class="qc-home-formats">{t("app_formats", _lang)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_cols[2]:
        st.markdown('<div class="qc-home-actions-spacer"></div>', unsafe_allow_html=True)
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button(t("lang_toggle", lang), key="lang_btn", help=t("lang_tooltip", lang), width="stretch"):
                st.session_state.lang = "en" if lang == "zh" else "zh"
                st.session_state.agent = None
                st.rerun()
        with action_cols[1]:
            if st.button("⚙️", key="settings_btn", help=t("settings_tooltip", lang), width="stretch"):
                st.session_state.show_settings = True
else:
    cols = st.columns([0.5, 3, 0.7])
    with cols[0]:
        logo_svg = render_logo_svg(40)
        if logo_svg:
            st.markdown(logo_svg, unsafe_allow_html=True)
        elif os.path.exists(LOGO_PNG):
            st.image(LOGO_PNG, width=40)
    with cols[2]:
        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button(t("lang_toggle", lang), key="lang_btn", help=t("lang_tooltip", lang), width="stretch"):
                st.session_state.lang = "en" if lang == "zh" else "zh"
                st.session_state.agent = None
                st.rerun()
        with action_cols[1]:
            if st.button("⚙️", key="settings_btn", help=t("settings_tooltip", lang), width="stretch"):
                st.session_state.show_settings = True

# Settings Dialog
if st.session_state.get("show_settings"):
    _lang = st.session_state.lang
    @st.dialog(t("settings_title", _lang), width="small")
    def settings_dialog():
        st.markdown(f"### {t('select_engine', _lang)}")

        provider_options = {t("deepseek_option", _lang): "deepseek", t("ollama_option", _lang): "ollama"}
        provider_label = st.selectbox(t("engine_label", _lang), list(provider_options.keys()), label_visibility="collapsed")
        provider = provider_options[provider_label]

        current_config = {}
        if provider == "ollama":
            model_name = st.text_input(t("model_name", _lang), value="qwen2.5:3b")
            st.caption(t("ollama_hint", _lang))
            current_config = {"provider": "ollama", "model_name": model_name, "api_key": None, "base_url": None}
        else:
            api_key = st.text_input(t("api_key_label", _lang), value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
            model_name = st.text_input(t("model_label", _lang), value="deepseek-v4-flash")
            current_config = {"provider": "deepseek", "model_name": model_name, "api_key": api_key, "base_url": "https://api.deepseek.com"}

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("save", _lang), width="stretch"):
                st.session_state.agent_config = current_config
                st.session_state.agent = None  # Force re-initialize
                st.session_state.show_settings = False
                st.rerun()
        with col2:
            if st.button(t("cancel", _lang), width="stretch"):
                st.session_state.show_settings = False
                st.rerun()

    settings_dialog()

# Agent initialization
def _get_fingerprint():
    try:
        return os.path.getmtime(os.path.join(os.getcwd(), "agent", "core.py")), os.path.getmtime(os.path.join(os.getcwd(), "agent", "tools.py"))
    except:
        return 0.0, 0.0

current_fingerprint = _get_fingerprint()
if st.session_state.agent_code_fingerprint != current_fingerprint:
    st.session_state.agent = None
    st.session_state.agent_code_fingerprint = current_fingerprint

# Initialize agent
config = st.session_state.agent_config
if config and not st.session_state.agent:
    if config["provider"] == "deepseek" and not config.get("api_key"):
        st.session_state.agent_error = "请先配置 API Key"
    else:
        try:
            st.session_state.agent = get_agent_executor(**config, lang=st.session_state.lang)
            st.session_state.agent_error = None
        except Exception as e:
            st.session_state.agent_error = str(e)

agent_error = st.session_state.agent_error
agent_ready = st.session_state.agent is not None

# Welcome screen
if not st.session_state.messages:
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]

    # Set avatar based on role
    if role == "user":
        avatar = USER_AVATAR if os.path.exists(USER_AVATAR) else "👤"
    else:
        avatar = LOGO_PNG if os.path.exists(LOGO_PNG) else "🤖"

    with st.chat_message(role, avatar=avatar):
        # Display text content
        st.markdown(msg["content"])

        # Display uploaded files info
        if msg.get("files"):
            for f_info in msg["files"]:
                st.caption(f"📎 {f_info['name']}")

        # Display images inline (like GPT)
        if msg.get("images"):
            for img_path in msg["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")

        # Thinking process expander
        if steps := msg.get("steps"):
            with st.expander(f"💭 {t('thinking_process', lang)}"):
                st.markdown(f"<div style='font-size: 0.85rem; color: #b0b0b0;'>{steps}</div>", unsafe_allow_html=True)

# Chat input with file upload
prompt = st.chat_input(
    placeholder=t("chat_placeholder", lang),
    accept_file="multiple",
    file_type=["segy", "sgy", "mseed", "miniseed", "h5", "hdf5", "sac", "png", "jpg", "jpeg", "gif", "json"],
    disabled=not agent_ready,
)

if prompt and agent_ready:
    # Get text and files
    text = prompt.text if hasattr(prompt, 'text') else ""
    files = prompt.files if hasattr(prompt, 'files') else []

    # Process uploaded files
    files_info = []
    if files:
        for f in files:
            data_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(data_dir, exist_ok=True)
            target_path = os.path.join(data_dir, f.name)

            with open(target_path, "wb") as out:
                out.write(f.getvalue())

            files_info.append({"name": f.name, "path": target_path})

            ext = os.path.splitext(f.name)[1].lower().lstrip(".")
            if ext in {"segy", "sgy"}:
                set_current_segy_path(target_path)
            elif ext in {"mseed", "miniseed"}:
                set_current_miniseed_path(target_path)
            elif ext in {"h5", "hdf5"}:
                set_current_hdf5_path(target_path)
            elif ext == "sac":
                set_current_sac_path(target_path)

    # Build content
    content = text or ""
    if files_info:
        file_names = ", ".join([f["name"] for f in files_info])
        if text:
            content = f"[{t('uploaded', lang)}: {file_names}]\n\n{text}"
        else:
            content = f"[{t('uploaded', lang)}: {file_names}]\n\n{t('read_file_default', lang)}"


    # Auto-switch response language by current user query language
    turn_lang = _detect_prompt_lang(content)
    if turn_lang != st.session_state.lang:
        st.session_state.lang = turn_lang
        set_current_lang(turn_lang)
        st.session_state.agent = None
        try:
            st.session_state.agent = get_agent_executor(**st.session_state.agent_config, lang=turn_lang)
            st.session_state.agent_error = None
        except Exception as e:
            st.session_state.agent_error = str(e)

    # Keep local lang variable in sync for this turn UI text
    lang = st.session_state.lang

    # Ensure this turn belongs to a persisted conversation.
    _ensure_active_conversation()

    # Add user message to the current conversation, then rerun so the assistant
    # turn is handled in a clean pass. This avoids Streamlit dropping the fresh
    # prompt when sidebar state also changes.
    st.session_state.messages.append({
        "role": "user",
        "content": content,
        "files": files_info if files_info else None
    })
    _save_current_conversation()
    st.session_state.pending_turn = {"content": content, "lang": lang}
    st.rerun()

if st.session_state.pending_turn and agent_ready:
    turn = dict(st.session_state.pending_turn)
    st.session_state.pending_turn = None
    content = turn.get("content", "")
    lang = turn.get("lang", st.session_state.lang)

    with st.chat_message("assistant", avatar=LOGO_PNG if os.path.exists(LOGO_PNG) else "🤖"):
        placeholder = st.empty()
        try:
            result = _invoke_agent_turn(content, lang)

            answer = result["answer"]
            steps_text = result["steps_text"]
            response_images = result["response_images"]
            text_only = result["text_only"]

            placeholder.markdown(text_only)

            if response_images:
                for img_path in response_images:
                    if os.path.exists(img_path):
                        st.image(img_path, width="stretch")

            progress_info = _extract_monitor_progress(result.get("response"))
            if progress_info:
                total = progress_info.get("total")
                downloaded = int(progress_info.get("downloaded") or 0)
                failed = int(progress_info.get("failed") or 0)
                processed = downloaded + failed
                if total and total > 0:
                    frac = max(0.0, min(1.0, processed / float(total)))
                    st.progress(frac, text=f"下载进度：{processed}/{total} 台站（成功 {downloaded}，失败 {failed}）")
                summary = progress_info.get("summary")
                if summary:
                    st.caption(summary)
                tail = progress_info.get("tail") or []
                if tail:
                    with st.expander("📈 下载阶段明细", expanded=False):
                        for item in tail:
                            if not isinstance(item, dict):
                                continue
                            message = str(item.get("message", "")).strip()
                            if message:
                                st.markdown(f"- {message}")

            if steps_text:
                with st.expander(f"💭 {t('thinking_process', lang)}", expanded=False):
                    st.markdown(
                        f"<div style='font-size: 0.85rem; color: #b0b0b0;'>{steps_text}</div>",
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "images": response_images,
                "steps": steps_text
            })
            _save_current_conversation()
            result["status"].update(label=t("done", lang), state="complete")
        except Exception as e:
            error_msg = f"{t('error', lang)}: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            _save_current_conversation()
