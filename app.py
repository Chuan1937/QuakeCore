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
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from deploy.i18n import t

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
    st.session_state.agent_config = {"provider": "deepseek", "model_name": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY", ""), "base_url": "https://api.deepseek.com"}
if "agent_code_fingerprint" not in st.session_state:
    st.session_state.agent_code_fingerprint = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "lang" not in st.session_state:
    st.session_state.lang = "en"
lang = st.session_state.lang
set_current_lang(lang)

# ==================== Layout ====================
# Sidebar content - chat history
with st.sidebar:
    st.markdown('<div class="qc-sidebar-brand">QuakeCore</div>', unsafe_allow_html=True)

    st.divider()

    # New Chat button
    if st.button("➕ 新对话", key="new_chat_btn", use_container_width=True):
        if st.session_state.current_conversation_id and st.session_state.messages:
            st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[new_id] = {"title": "新对话", "messages": []}
        st.session_state.current_conversation_id = new_id
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
                if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                    if st.session_state.current_conversation_id and st.session_state.messages:
                        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.messages = conv_data["messages"]
                    st.rerun()
            with row_cols[1]:
                if st.button("✕", key=f"del_conv_{conv_id}", help="删除对话", type="tertiary", use_container_width=True):
                    st.session_state.conversations.pop(conv_id, None)

                    if st.session_state.current_conversation_id == conv_id:
                        if st.session_state.conversations:
                            next_id = list(st.session_state.conversations.keys())[-1]
                            st.session_state.current_conversation_id = next_id
                            st.session_state.messages = st.session_state.conversations[next_id]["messages"]
                        else:
                            st.session_state.current_conversation_id = None
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
    conv = st.session_state.conversations.get(st.session_state.current_conversation_id)
    if conv and conv["messages"]:
        st.session_state.messages = conv["messages"]

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
            if st.button(t("lang_toggle", lang), key="lang_btn", help=t("lang_tooltip", lang), use_container_width=True):
                st.session_state.lang = "en" if lang == "zh" else "zh"
                st.session_state.agent = None
                st.rerun()
        with action_cols[1]:
            if st.button("⚙️", key="settings_btn", help=t("settings_tooltip", lang), use_container_width=True):
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
            if st.button(t("lang_toggle", lang), key="lang_btn", help=t("lang_tooltip", lang), use_container_width=True):
                st.session_state.lang = "en" if lang == "zh" else "zh"
                st.session_state.agent = None
                st.rerun()
        with action_cols[1]:
            if st.button("⚙️", key="settings_btn", help=t("settings_tooltip", lang), use_container_width=True):
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
            model_name = st.text_input(t("model_label", _lang), value="deepseek-chat")
            current_config = {"provider": "deepseek", "model_name": model_name, "api_key": api_key, "base_url": "https://api.deepseek.com"}

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("save", _lang), use_container_width=True):
                st.session_state.agent_config = current_config
                st.session_state.agent = None  # Force re-initialize
                st.session_state.show_settings = False
                st.rerun()
        with col2:
            if st.button(t("cancel", _lang), use_container_width=True):
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
                    st.image(img_path, use_container_width=True)

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


    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": content,
        "files": files_info if files_info else None
    })

    # Update conversation title if it's a new conversation
    if st.session_state.current_conversation_id:
        conv = st.session_state.conversations.get(st.session_state.current_conversation_id)
        if conv and conv["title"] == "新对话":
            # Use first 30 chars of user message as title
            title = content[:30].replace("\n", " ").strip()
            if len(content) > 30:
                title += "..."
            st.session_state.conversations[st.session_state.current_conversation_id]["title"] = title
        # Sync messages to conversation storage
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages

    # Display user message immediately
    user_avatar = USER_AVATAR if os.path.exists(USER_AVATAR) else "👤"
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(content)
        if files_info:
            for f_info in files_info:
                st.caption(f"📎 {f_info['name']}")

    # Generate response
    with st.chat_message("assistant", avatar=LOGO_PNG if os.path.exists(LOGO_PNG) else "🤖"):
        placeholder = st.empty()
        try:
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            status = st.status(t("thinking", lang), expanded=True)
            callback = StreamlitCallbackHandler(status.container(), expand_new_thoughts=True)

            response = st.session_state.agent.invoke(
                {"input": content, "chat_history": chat_history},
                config={"callbacks": [callback]}
            )

            answer = response["output"]
            steps_text = ""
            if response.get("intermediate_steps"):
                for i, (action, obs) in enumerate(response["intermediate_steps"], 1):
                    tool_input = action.tool_input
                    if isinstance(tool_input, dict):
                        tool_input = json.dumps(tool_input, ensure_ascii=False)
                    steps_text += f"{i}. **{action.tool}**\n   → {obs}\n\n"

            # Extract images from markdown
            img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
            response_images = [p.strip() for p in img_pattern.findall(answer)]
            
            # Also extract from intermediate steps if LLM omitted them
            if response.get("intermediate_steps"):
                for _, obs in response["intermediate_steps"]:
                    if isinstance(obs, str):
                        for p in img_pattern.findall(obs):
                            if p.strip() not in response_images:
                                response_images.append(p.strip())

            # Remove image markdown from text for cleaner display
            text_only = img_pattern.sub('', answer).strip()

            placeholder.markdown(text_only)

            # Display images inline
            if response_images:
                for img_path in response_images:
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)

            # Store message with images
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "images": response_images,
                "steps": steps_text
            })

            # Sync to conversation storage
            if st.session_state.current_conversation_id:
                st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.messages

            status.update(label=t("done", lang), state="complete")

        except Exception as e:
            error_msg = f"{t('error', lang)}: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
