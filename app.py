import streamlit as st
import os
import tempfile
import json
import shutil
from agent.core import get_agent_executor
from agent.tools import set_current_segy_path, set_current_miniseed_path, set_current_hdf5_path
from langchain_core.messages import HumanMessage, AIMessage


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

# Page Config
st.set_page_config(page_title="QuakeCore AI Agent", layout="wide")

st.title("🌋 QuakeCore AI - 地震数据智能助手")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_config" not in st.session_state:
    st.session_state.agent_config = None

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "current_file_ext" not in st.session_state:
    st.session_state.current_file_ext = None

with st.sidebar:
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
    "上传数据文件（SEGY/SGY/MiniSEED/HDF5）",
    type=["segy", "sgy", "mseed", "miniseed","h5","hdf5"],
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

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
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
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        try:
            # Prepare chat history for LangChain
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude current prompt
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Run Agent
            response = st.session_state.agent.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            answer = response["output"]
            steps_markdown = format_intermediate_steps(response.get("intermediate_steps", []))

            message_placeholder.markdown(answer)
            with st.expander("思考过程", expanded=False):
                st.markdown(steps_markdown)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "steps": steps_markdown
            })
            
        except Exception as e:
            active_provider = (st.session_state.agent_config or current_agent_config or {}).get("provider", "ollama")
            provider_hint = "Ollama 本地服务" if active_provider == "ollama" else "DeepSeek API 配置或网络状态"
            error_msg = f"发生错误: {str(e)}\n\n请检查 {provider_hint}。"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Cleanup on exit (Optional - Streamlit handles temp files differently depending on OS, 
# but explicit cleanup is good practice if we were managing sessions manually)
