import streamlit as st
import os
import tempfile
import json
import shutil
import re
from agent.core import get_agent_executor
from agent.tools import (
    set_current_segy_path,
    set_current_miniseed_path,
    set_current_hdf5_path,
    set_current_sac_path,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import StreamlitCallbackHandler


def _agent_code_fingerprint() -> tuple[float, float]:
    """Return a fingerprint for agent code so we can refresh cached executors when code changes."""
    try:
        core_mtime = os.path.getmtime(os.path.join(os.getcwd(), "agent", "core.py"))
    except OSError:
        core_mtime = 0.0
    try:
        tools_mtime = os.path.getmtime(os.path.join(os.getcwd(), "agent", "tools.py"))
    except OSError:
        tools_mtime = 0.0
    return core_mtime, tools_mtime


def format_intermediate_steps(intermediate_steps):
    """Turn LangChain intermediate steps into markdown for display."""
    if not intermediate_steps:
        return "（本次没有调用工具）"

    blocks = []
    for idx, (action, observation) in enumerate(intermediate_steps, start=1):
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input_str = json.dumps(tool_input, ensure_ascii=False)
        else:
            tool_input_str = str(tool_input)

        block = (
            f"{idx}. **工具** `{action.tool}`\n"
            f"   - 输入: `{tool_input_str}`\n"
            f"   - 观察: {observation}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def render_message_content(content, msg_idx=0):
    """Render message content, extracting images to the right column if present."""
    # Regex to find markdown images: ![alt](path)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    images = image_pattern.findall(content)
    
    if images:
        # Remove image tags from content to avoid double rendering
        text_without_images = image_pattern.sub('', content)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(text_without_images)
        with col2:
            for idx, (alt, path) in enumerate(images):
                path = path.strip()
                if os.path.exists(path):
                    st.image(path, caption=alt, width='stretch')
                    with open(path, "rb") as file:
                        st.download_button(
                            label="⬇️ 下载图片",
                            data=file,
                            file_name=os.path.basename(path),
                            mime="image/png",
                            key=f"download_{msg_idx}_{idx}_{os.path.basename(path)}"
                        )
                else:
                    st.warning(f"无法加载图片: {path}")
    else:
        st.markdown(content)


# Page Config
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "resources", "QuakeCore.png")
USER_AVATAR_PATH = os.path.join(BASE_DIR, "resources", "chuanjun.jpg")

st.set_page_config(
    page_title="QuakeCore AI Agent",
    page_icon=LOGO_PATH,
    layout="wide"
)

# Custom CSS for "Cool" effects
st.markdown("""
<style>
    :root {
        /* Light Theme Variables from Seismic Design */
        --bg-app: rgba(255, 255, 255, 0.65);
        --bg-sidebar: rgba(255, 255, 255, 0.5);
        --text-primary: #4a4a4a;
        --accent-color: #8ec5fc; /* Light Blue */
        --accent-hover: #ff9a9e; /* Light Pink */
        --message-user-bg: #bde0fe;
        --message-ai-bg: #ffffff;
        --radius-lg: 30px;
        --radius-md: 20px;
    }

    /* Global Background - Pure White */
    .stApp {
        background-color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: var(--text-primary);
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 2px 8px rgba(142, 197, 252, 0.15);
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 20px;
        border: none;
        box-shadow: 0 4px 6px rgba(142, 197, 252, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--accent-hover);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 154, 158, 0.4);
        color: white;
    }

    /* Status Widget (The "Thinking" box) */
    div[data-testid="stStatusWidget"] {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px solid var(--accent-color);
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(142, 197, 252, 0.4);
        animation: neon-pulse 2s infinite alternate;
    }

    @keyframes neon-pulse {
        0% { box-shadow: 0 0 5px var(--accent-color); border-color: var(--accent-color); }
        100% { box-shadow: 0 0 15px var(--accent-color); border-color: #69f0ae; }
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Window Controls Simulation */
    .window-controls {
        display: flex;
        gap: 8px;
        margin-bottom: 10px;
    }
    .control {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .control.red { background-color: #ffadad; }
    .control.yellow { background-color: #ffd6a5; }
    .control.green { background-color: #caffbf; }
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
with col_title:
    st.title("震元引擎 (QuakeCore Engine)")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_data_panel" not in st.session_state:
    st.session_state.show_data_panel = True

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_config" not in st.session_state:
    st.session_state.agent_config = None

if "agent_code_fingerprint" not in st.session_state:
    st.session_state.agent_code_fingerprint = None

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "current_file_ext" not in st.session_state:
    st.session_state.current_file_ext = None

with st.sidebar:
    st.header("数据面板")
    st.session_state.show_data_panel = st.checkbox(
        "显示可视化结果",
        value=bool(st.session_state.show_data_panel),
        key="show_data_panel_checkbox",
    )
    
    # Placeholder for data panel content
    data_panel_placeholder = st.empty()

    st.divider()
    st.header("模型配置")
    provider_options = {
        "DeepSeek API": "deepseek",
        "本地 Ollama": "ollama",
    }
    provider_label = st.selectbox("选择推理引擎", list(provider_options.keys()), index=0, key="provider_select")
    provider = provider_options[provider_label]

    current_agent_config = {}
    if provider == "ollama":
        model_name = st.text_input("本地模型名称 (Ollama)", value="qwen2.5:3b", key="ollama_model_input")
        st.info("请确保本地已安装 Ollama 并运行了对应模型")
        current_agent_config = {
            "provider": "ollama",
            "model_name": model_name,
            "api_key": None,
            "base_url": None,
        }
    else:
        deepseek_model = st.text_input("DeepSeek 模型名称", value="deepseek-chat", key="deepseek_model_input")
        deepseek_api_key = st.text_input(
            "DeepSeek API Key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            key="deepseek_api_key_input",
        )
        deepseek_base_url = st.text_input(
            "DeepSeek Base URL",
            value="https://api.deepseek.com",
            key="deepseek_base_url_input",
        )
        current_agent_config = {
            "provider": "deepseek",
            "model_name": deepseek_model,
            "api_key": deepseek_api_key,
            "base_url": deepseek_base_url,
        }
        st.info("使用 DeepSeek 时需要有效的 API Key，可在环境变量 DEEPSEEK_API_KEY 中配置。")

    st.divider()
    st.markdown(
        "<div style='font-size: 0.8em; color: #7a7a7a; text-align: center;'>"
        "AI 生成的内容可能存在误差，请结合专业工具核实。"
        "</div>",
        unsafe_allow_html=True
    )

current_fingerprint = _agent_code_fingerprint()
if st.session_state.agent_code_fingerprint != current_fingerprint:
    # Agent code changed (e.g., tools/prompt updated) -> rebuild executor
    st.session_state.agent = None
    st.session_state.agent_config = None
    st.session_state.agent_code_fingerprint = current_fingerprint

config_changed = current_agent_config != st.session_state.agent_config
if config_changed:
    if current_agent_config["provider"] == "deepseek" and not current_agent_config.get("api_key"):
        st.session_state.agent = None
        st.session_state.agent_config = None
        st.session_state.agent_error = "DeepSeek 模式需要提供 API Key。"
    else:
        try:
            st.session_state.agent = get_agent_executor(**current_agent_config)
            st.session_state.agent_config = current_agent_config
            st.session_state.agent_error = None
        except Exception as err:
            st.session_state.agent = None
            st.session_state.agent_config = None
            st.session_state.agent_error = str(err)

agent_error = st.session_state.agent_error
agent_ready = st.session_state.agent is not None

if agent_error:
    st.error(agent_error)
elif not agent_ready:
    st.info("请在侧边栏完成模型配置以启动对话。")

# Attachment upload near chat
uploaded_file_main = st.file_uploader(
    "上传数据文件（SEGY/SGY/MiniSEED/HDF5/SAC）",
    type=["segy", "sgy", "mseed", "miniseed","h5","hdf5","sac"],
    help="选择单个文件并复制到 data 目录"
)

if uploaded_file_main:
    filename = uploaded_file_main.name
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, filename)
    with open(target_path, "wb") as f:
        f.write(uploaded_file_main.getvalue())
    st.session_state.current_file_path = target_path
    st.session_state.uploaded_filename = filename
    st.session_state.current_file_ext = ext
    st.success(f"文件已复制到 data: `{filename}`")

if "current_file_path" in st.session_state:
    ext = st.session_state.get("current_file_ext")
    if ext in {"segy", "sgy"}:
        set_current_segy_path(st.session_state.current_file_path)
    elif ext in {"mseed", "miniseed"}:
        set_current_miniseed_path(st.session_state.current_file_path)
    elif ext in {"h5","hdf5"}:
        set_current_hdf5_path(st.session_state.current_file_path)
    elif ext == "sac":
        set_current_sac_path(st.session_state.current_file_path)



def render_message_content_no_image(content):
    """Render message content but strip images (since they are in the right column)."""
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    text_without_images = image_pattern.sub('', content)
    st.markdown(text_without_images)


# Chat Interface (full width)
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = USER_AVATAR_PATH if role == "user" else LOGO_PATH
    with st.chat_message(role, avatar=avatar):
        render_message_content_no_image(message["content"])
        if steps := message.get("steps"):
            with st.expander("思考过程", expanded=False):
                st.markdown(steps)

# User Input
prompt = st.chat_input(
    "输入你的问题（可先在上方上传文件）",
    disabled=not agent_ready,
)

if prompt and agent_ready:
    # Display user message
    st.chat_message("user", avatar=USER_AVATAR_PATH).markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant", avatar=LOGO_PATH):
        message_placeholder = st.empty()
        
        try:
            # Prepare chat history for LangChain
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude current prompt
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Run Agent with a visible status (so the neon animation actually applies)
            status = st.status("正在思考…", expanded=True)
            st_callback = StreamlitCallbackHandler(status.container(), expand_new_thoughts=True)
            
            response = st.session_state.agent.invoke(
                {"input": prompt, "chat_history": chat_history},
                config={"callbacks": [st_callback]}
            )
            
            answer = response["output"]
            steps_markdown = format_intermediate_steps(response.get("intermediate_steps", []))

            message_placeholder.empty()
            render_message_content_no_image(answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "steps": steps_markdown
            })

            status.update(label="完成", state="complete", expanded=False)
            
        except Exception as e:
            active_provider = (st.session_state.agent_config or current_agent_config or {}).get("provider", "ollama")
            provider_hint = "Ollama 本地服务" if active_provider == "ollama" else "DeepSeek API 配置或网络状态"
            error_msg = f"发生错误: {str(e)}\n\n请检查 {provider_hint}。"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Helper for modal image view
if hasattr(st, "dialog"):
    image_dialog = st.dialog
elif hasattr(st, "experimental_dialog"):
    image_dialog = st.experimental_dialog
else:
    image_dialog = None

if image_dialog:
    @image_dialog("可视化详情", width="large")
    def view_image_modal(path, caption):
        st.image(path, caption=caption, width="stretch")

def update_data_panel():
    if not st.session_state.show_data_panel:
        data_panel_placeholder.empty()
        return

    with data_panel_placeholder.container():
        st.caption("最近一次生成的图像：")
        latest_image_path = None
        latest_image_caption = None
        image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
        for msg in reversed(st.session_state.messages):
            if msg.get("role") != "assistant":
                continue
            match = image_pattern.search(msg.get("content", ""))
            if match:
                latest_image_caption = match.group(1)
                latest_image_path = match.group(2).strip()
                break

        if latest_image_path and os.path.exists(latest_image_path):
            st.image(latest_image_path, caption=latest_image_caption or "结果图", width='stretch')
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if image_dialog:
                    if st.button("🔍 放大", key=f"view_{os.path.basename(latest_image_path)}", use_container_width=True):
                        view_image_modal(latest_image_path, latest_image_caption)
            with col2:
                with open(latest_image_path, "rb") as file:
                    st.download_button(
                        label="⬇️ 下载",
                        data=file,
                        file_name=os.path.basename(latest_image_path),
                        mime="image/png",
                        key=f"download_latest_vis_{os.path.basename(latest_image_path)}",
                        use_container_width=True
                    )
        else:
            st.info("暂无可视化结果")

# Update data panel at the end of the script
update_data_panel()
