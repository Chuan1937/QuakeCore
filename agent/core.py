import os
from typing import Literal, Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

Provider = Literal["deepseek", "ollama"]


def _build_llm(
    provider: Provider,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    streaming: bool = False,
):
    if provider == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DeepSeek 模式需要提供 API Key。")

        return ChatOpenAI(
            api_key=key,
            base_url=base_url or "https://api.deepseek.com",
            model=model_name,
            temperature=0,
            streaming=streaming,
        )

    return ChatOllama(
        model=model_name,
        base_url=base_url or "http://localhost:11434",
        temperature=0,
        num_predict=4096,
    )


def get_agent_executor(
    provider: Provider = "deepseek",
    model_name: str = "deepseek-v4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    lang: str = "en",
    streaming: bool = False,
    skill_context: str = "",
):
    """Create a tool-calling agent with DeepSeek's native function calling.

    This uses DeepSeek's tool_calls API instead of text-based ReAct parsing.
    Benefits:
    - Faster: no text generation for "Thought/Action/Action Input" blocks
    - More reliable: structured JSON output, no parsing errors
    - Better tool selection: model sees all tools in OpenAI function format

    To add a new tool, create a file in ``quakecore_tools/`` with
    ``@register_tool`` — no changes needed here.
    """
    import quakecore_tools
    quakecore_tools.ensure_legacy_tools()
    from quakecore_tools.registry import build_tool_list, build_tool_descriptions

    llm = _build_llm(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url, streaming=streaming)

    tools = build_tool_list()

    safe_skill_context = (skill_context or "").replace("{", "{{").replace("}", "}}").strip()

    if lang == "en":
        system_message = (
            "You are QuakeCore, an intelligent seismic data analysis assistant. "
            "Answer the user's question in English.\n\n"
            "TOOL SELECTION RULES (MUST follow strictly):\n"
            "- User says '初至拾取', '震相拾取', 'phase picking', 'pick arrivals' → MUST call pick_first_arrivals\n"
            "- User says 'file structure', 'read file' → call get_file_structure\n"
            "- User says 'statistics', 'waveform stats' → call get_quick_stats\n"
            "- User says 'continuous monitoring', '地震监测' → call run_continuous_monitoring\n"
            "- User says 'locate', '定位' → call locate_earthquake or locate_uploaded_data_nearseismic\n"
            "- User says 'convert' → call the corresponding convert_* tool\n\n"
            "Other rules:\n"
            "- Do NOT invent parameters the user did not request.\n"
            "- Use get_loaded_context to confirm file is loaded before calling other tools.\n"
            "- If a tool returns success:false, try a different approach or explain the issue.\n"
            "- If a tool returns a Markdown table or image link, copy it EXACTLY into your answer.\n"
        )
    else:
        system_message = (
            "你是 QuakeCore，智能地震数据分析助手。用中文回答。\n\n"
            "工具选择规则（必须严格遵守）：\n"
            "- 用户说'初至拾取'、'震相拾取'、'phase picking'、'pick arrivals' → 直接调用 pick_first_arrivals，不要先调用其他工具\n"
            "- 用户说'文件结构'、'读取文件' → 调用 get_file_structure\n"
            "- 用户说'统计'、'波形统计' → 调用 get_quick_stats\n"
            "- 用户说'连续监测'、'地震监测' → 调用 run_continuous_monitoring\n"
            "- 用户说'定位' → 调用 locate_earthquake 或 locate_uploaded_data_nearseismic\n"
            "- 用户说'转换' → 调用对应的 convert_* 工具\n\n"
            "其他规则：\n"
            "- 不要编造用户未请求的参数。\n"
            "- 工具返回 success:false 时尝试其他方法或向用户说明。\n"
            "- 工具返回的 Markdown 表格或图片链接必须原样复制到回答中。\n"
        )

    if safe_skill_context:
        if lang == "en":
            system_message += f"\nSkill context:\n{safe_skill_context}\n"
        else:
            system_message += f"\n技能上下文：\n{safe_skill_context}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=8,
    )
