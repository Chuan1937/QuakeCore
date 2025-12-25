import os
from typing import Literal, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agent.tools import (
    get_segy_binary_header,
    get_segy_structure,
    get_segy_text_header,
    read_trace_sample,
)

Provider = Literal["ollama", "deepseek"]


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

    # Default to Ollama (local) provider
    return ChatOllama(model=model_name, temperature=0)


def get_agent_executor(
    provider: Provider = "ollama",
    model_name: str = "qwen2.5:3b",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Create a LangChain ReAct agent with configurable LLM backends."""

    llm = _build_llm(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url)

    tools = [
        get_segy_structure,
        get_segy_text_header,
        get_segy_binary_header,
        read_trace_sample,
    ]

    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

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
