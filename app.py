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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {display: none !important;}

    .stApp {
        background: #0f0f1a;
    }

    .main .block-container {
        max-width: 800px;
        padding: 1.5rem 1.5rem 6rem;
    }

    /* Chat messages container */
    [data-testid="stChatMessage"] {
        padding: 0.25rem 0 !important;
    }

    /* User message - align right */
    [data-testid="stChatMessageUser"] {
        display: flex !important;
        justify-content: flex-end !important;
    }

    /* User message bubble */
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 10px 14px !important;
        color: white !important;
        display: inline-block !important;
        max-width: 70% !important;
    }

    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] div {
        color: white !important;
    }

    /* Assistant message - align left */
    [data-testid="stChatMessageAssistant"] {
        display: flex !important;
        justify-content: flex-start !important;
    }

    /* Assistant message bubble */
    [data-testid="stChatMessageAssistant"] [data-testid="stChatMessageContent"] {
        background: #1e1e32 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
        display: inline-block !important;
        max-width: 85% !important;
    }

    /* Chat input */
    .stChatInput > div {
        background: #1a1a2e !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .stChatInput textarea {
        color: #e2e8f0 !important;
    }

    .stChatInput textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Status widget */
    div[data-testid="stStatusWidget"] {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Images in chat */
    .stImage img {
        border-radius: 12px;
        margin-top: 0.5rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a2e !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }

    /* Dialog */
    .stDialog {
        background: #1a1a2e !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 16px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {width: 6px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {background: #3b3b5c; border-radius: 3px;}
</style>
""", unsafe_allow_html=True)

# Header with settings + language toggle
if "lang" not in st.session_state:
    st.session_state.lang = "en"
lang = st.session_state.lang
set_current_lang(lang)
header_col1, header_col2, header_col3, header_col4 = st.columns([1, 6, 1, 1])
with header_col1:
    logo_svg = render_logo_svg(40)
    if logo_svg:
        st.markdown(logo_svg, unsafe_allow_html=True)
    elif os.path.exists(LOGO_PNG):
        st.image(LOGO_PNG, width=40)

with header_col2:
    st.markdown("""
    <h1 style="margin: 0; font-size: 1.5rem; font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center;">
        QuakeCore Engine
    </h1>
    """, unsafe_allow_html=True)

with header_col3:
    if st.button(t("lang_toggle", lang), key="lang_btn", help=t("lang_tooltip", lang)):
        st.session_state.lang = "en" if lang == "zh" else "zh"
        st.session_state.agent = None  # Force re-init with new language
        st.rerun()

with header_col4:
    if st.button("⚙️", key="settings_btn", help=t("settings_tooltip", lang)):
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

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
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
    _lang = st.session_state.lang
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <div style="font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, #4A6591 0%, #E64B35 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {t("app_title", _lang)}
        </div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;">
            {t("app_subtitle", _lang)}
        </p>
        <p style="color: #6b7280; font-size: 0.8rem;">
            {t("app_formats", _lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)

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
                st.markdown(f"<div style='font-size: 0.85rem; color: #9ca3af;'>{steps}</div>", unsafe_allow_html=True)

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

            status.update(label=t("done", lang), state="complete")

        except Exception as e:
            error_msg = f"{t('error', lang)}: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
