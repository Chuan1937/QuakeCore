import os
from typing import Literal, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agent.tools import (
    get_loaded_context,
    get_file_structure,
    read_file_trace,
    get_hdf5_structure,
    read_hdf5_trace,
    convert_hdf5_to_numpy,
    convert_hdf5_to_excel,
    get_hdf5_keys,
    get_segy_binary_header,
    get_segy_structure,
    get_segy_text_header,
    read_trace_sample,
    convert_segy_to_numpy,
    convert_segy_to_excel,
    convert_segy_to_hdf5,
    get_miniseed_structure,
    read_miniseed_trace,
    convert_miniseed_to_numpy,
    convert_miniseed_to_hdf5,
    convert_miniseed_to_sac,
    run_phase_picking,
)

Provider = Literal["deepseek", "ollama"]


def _build_llm(
    provider: Provider,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
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
        )

    # # Default to Ollama LLM
    # # 默认使用 Ollama LLM
    # return ChatOllama(model=model_name, temperature=0)

    # Default to DeepSeek API
    # 默认使用 DeepSeek API
    return ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model=model_name,
        temperature=0,
    )


def get_agent_executor(
    # provider: Provider = "ollama",
    # model_name: str = "qwen2.5:3b",
    provider: Provider = "deepseek",
    model_name: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Create a LangChain ReAct agent with configurable LLM backends."""
    """创建一个具有可配置 LLM 后端的 LangChain ReAct 代理。"""

    llm = _build_llm(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url)

    tools = [
        get_loaded_context,
        get_file_structure,
        read_file_trace,
        get_hdf5_keys,
        get_hdf5_structure,
        read_hdf5_trace,
        convert_hdf5_to_numpy,
        convert_hdf5_to_excel,
        get_segy_structure,
        get_segy_text_header,
        get_segy_binary_header,
        read_trace_sample,
        convert_segy_to_numpy,
        convert_segy_to_excel,
        convert_segy_to_hdf5,
        get_miniseed_structure,
        read_miniseed_trace,
        convert_miniseed_to_numpy,
        convert_miniseed_to_hdf5,
        convert_miniseed_to_sac,
        run_phase_picking,
    ]

    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Important rules:
- Do NOT invent parameters the user did not request.
- For data export/conversion tools: if the user does NOT specify a range, do NOT set `count`.
    (The tools default to exporting ALL traces when `count` is omitted or set to null.)
- If the user asks for a range like "100到200道", clarify whether it is inclusive and whether indexing is 0-based.
    If you must proceed without clarification, assume 0-based trace index and interpret as start_trace=100, count=100 (100..199).
- Always choose an output path under data/convert/ unless the user explicitly requests another folder.
- Output MUST follow the exact format below. Do not add extra text (no markdown, no explanations) outside the fields.
- For generic requests like "读取这个文件" or "给我结构", call get_file_structure.
- Before choosing SEGY/MiniSEED specific tools, use get_loaded_context to determine the loaded file type.
- For "读取第X条轨迹", prefer read_file_trace unless the user explicitly specifies SEGY/MiniSEED.
- If the loaded file is HDF5, prefer get_hdf5_structure / read_hdf5_trace and use convert_hdf5_to_numpy/convert_hdf5_to_excel for conversions.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
