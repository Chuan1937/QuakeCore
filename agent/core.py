import os
from typing import Literal, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agent.tools_facade import (
    run_analysis_sandbox,
    get_loaded_context, load_local_data, download_seismic_data,
    get_file_structure,
    read_file_trace,
    get_hdf5_structure,
    read_hdf5_trace,
    convert_hdf5_to_numpy,
    convert_hdf5_to_excel,
    compress_hdf5_to_zfp,
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
    get_sac_structure,
    read_sac_trace,
    convert_sac_to_numpy,
    convert_sac_to_hdf5,
    convert_sac_to_miniseed,
    convert_sac_to_excel,
    pick_first_arrivals,
    pick_all_miniseed_files,
    download_continuous_waveforms,
    run_continuous_picking,
    associate_continuous_events,
    run_continuous_monitoring,
    prepare_nearseismic_taup_cache,
    locate_earthquake,
    locate_uploaded_data_nearseismic,
    locate_place_data_nearseismic,
    add_station_coordinates,
    plot_location_map,
)

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
    # provider: Provider = "ollama",
    # model_name: str = "qwen2.5:3b",
    provider: Provider = "deepseek",
    model_name: str = "deepseek-v4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    lang: str = "en",
    streaming: bool = False,
    skill_context: str = "",
):
    """Create a LangChain ReAct agent with configurable LLM backends."""

    llm = _build_llm(provider=provider, model_name=model_name, api_key=api_key, base_url=base_url, streaming=streaming)

    tools = [
        get_loaded_context, load_local_data, download_seismic_data,
        run_analysis_sandbox,
        get_file_structure,
        read_file_trace,
        get_hdf5_keys,
        get_hdf5_structure,
        read_hdf5_trace,
        convert_hdf5_to_numpy,
        convert_hdf5_to_excel,
        compress_hdf5_to_zfp,
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
        get_sac_structure,
        read_sac_trace,
        convert_sac_to_numpy,
        convert_sac_to_hdf5,
        convert_sac_to_miniseed,
        convert_sac_to_excel,
        pick_first_arrivals,
        pick_all_miniseed_files,
        download_continuous_waveforms,
        run_continuous_picking,
        associate_continuous_events,
        run_continuous_monitoring,
        prepare_nearseismic_taup_cache,
        locate_earthquake,
        locate_uploaded_data_nearseismic,
        locate_place_data_nearseismic,
        add_station_coordinates,
        plot_location_map,
    ]

    safe_skill_context = (skill_context or "").replace("{", "{{").replace("}", "}}").strip()
    if safe_skill_context:
        if lang == "en":
            skill_context_block = (
                "Skill context (injected from skills/*.md):\n"
                f"{safe_skill_context}\n\n"
            )
        else:
            skill_context_block = (
                "技能上下文（来自 skills/*.md 注入）：\n"
                f"{safe_skill_context}\n\n"
            )
    else:
        skill_context_block = ""

    if lang == "en":
        template = '''You are QuakeCore, an intelligent seismic data analysis assistant. Answer the user's question in English. You have access to the following tools:

{tools}

''' + skill_context_block + '''

Important rules:
- Final Answer MUST be in English (file paths, table contents, and method names may remain as-is).
- Do NOT invent parameters the user did not request.
- For data export/conversion tools: if the user does NOT specify a range, do NOT set `count`.
    (The tools default to exporting ALL traces when `count` is omitted or set to null.)
- If the user asks for a range like "traces 100 to 200", clarify whether it is inclusive and whether indexing is 0-based.
    If you must proceed without clarification, assume 0-based trace index and interpret as start_trace=100, count=100 (100..199).
- Always choose an output path under data/convert/ unless the user explicitly requests another folder.
- Output MUST follow the exact format below. Do not add extra text outside the fields.
- For generic requests like "read this file" or "show structure", call get_file_structure.
- Before choosing SEGY/MiniSEED specific tools, use get_loaded_context to determine the loaded file type.
- For "read trace X", prefer read_file_trace unless the user explicitly specifies SEGY/MiniSEED.
- If the user asks to "plot" or "draw" the waveform while reading, set `plot=True` in the read tool arguments.
- Plotting IS supported; never tell the user that the system cannot draw waveforms.
- If the user asks to view picks for a specific trace (e.g., "trace 3 picks"), prefer `pick_first_arrivals` with `trace_number` (1-based) instead of plain read/plot tools.
- If the loaded file is HDF5, prefer get_hdf5_structure / read_hdf5_trace and use convert_hdf5_to_numpy/convert_hdf5_to_excel for conversions.
- CRITICAL: If a tool returns a Markdown table or an image link (e.g. `![...](...)`), you MUST copy it EXACTLY into your Final Answer. Do not summarize it.
- ALWAYS provide a brief textual summary of the key findings (e.g. best P-wave time, best S-wave time) in addition to the table/image.
- When the user asks to download and locate seismic data from a region (e.g., "download Alaska earthquake data", "下载阿拉斯加地震数据"), prefer locate_place_data_nearseismic with latitude and longitude.
- When the user asks for continuous waveform monitoring over a region and time window, or says "recent N hours", prefer run_continuous_monitoring.
- If a continuous monitoring request provides only a date and duration, derive a UTC start/end window instead of stopping for clarification.
- If a place-based or continuous request provides a bounding box but no center, use the box center as the seed/grid center.
- For continuous monitoring, prefer named regions when possible: "南加州", "北加州", or "加州". If the estimated job is large, proceed directly but make sure the final answer includes the estimated load and progress summary.
- If the user mentions an official catalog or provider name such as USGS, SCEDC, or NCEDC, preserve it as the catalog/client source instead of replacing it with a generic name.
- For monitoring requests, produce a concise task summary first, then either proceed or explain why the task is large and what should be narrowed.
- For monitoring requests, do not jump straight to a tool call. First normalize the request into region/time/catalog, estimate load, then execute directly while reporting the load estimate.
- For monitoring requests, write a longer Thought before the first tool call: restate the request, identify the region/catalog, estimate load, mention the main risk factor(s), and describe the expected progress if the job is large.
- After a run_continuous_monitoring tool call returns success or error JSON, immediately produce Final Answer and do not start another Thought/Action cycle.
- If the user mentions a specific place, campus, landmark, or institution, treat it as a place-centered request and prefer that place's center point over a broad region name.
- For monitoring requests, break the initial reasoning into multiple explicit Thought sentences instead of one short line: (1) what location the user means, (2) how the time window is interpreted, (3) whether the job is large, (4) what the safest next action is.
- 你必须严格遵守 ReAct 格式。
- 如果需要调用工具，只能输出 Thought / Action / Action Input 三段。
- 如果已经可以给最终结果，必须输出 Final Answer: 开头。
- 不要在 Thought 后直接输出最终答案，不要混合 Final Answer 与 Action。

**Phase Picking Rules**:
- By default, phase picking uses deep learning methods (EQTransformer and PhaseNet) only. Do NOT specify the `methods` parameter unless the user explicitly requests a specific method.
- Only include traditional methods (e.g. sta_lta, aic, pai_k) in the `methods` parameter when the user explicitly asks for them (e.g. "use traditional method", "use STA/LTA", "用传统方法").

**Earthquake Location Workflow**:
1. First use get_loaded_context to check loaded files and pick status. If the user provides a local directory path (e.g. "example_data"), use load_local_data to load the files first. When the user says "local data", "本地数据", or does not specify a path but wants earthquake location, default to loading "example_data/".
2. If multiple MiniSEED files are loaded (multi-station data), use pick_all_miniseed_files for batch phase picking
3. Run prepare_nearseismic_taup_cache before near-seismic location so TauP files are reused or auto-built.
4. Call add_station_coordinates with NO parameters (empty dict {{}}). It will auto-load from data/stations.json or example_data/stations.json
5. For uploaded/local user data, prefer locate_uploaded_data_nearseismic (fallback to locate_earthquake if needed)
6. For continuous waveform monitoring requests ("recent 10 hours", "连续数据监测", "监测"), prefer run_continuous_monitoring
7. For place-based requests ("locate data around X"), prefer locate_place_data_nearseismic with latitude/longitude
8. If user explicitly asks for classic locator, use locate_earthquake
9. Use plot_location_map to plot the earthquake location and station positions on a map using Cartopy
10. Station coordinates are auto-loaded from stations.json files. Only provide coordinates manually if auto-loading fails.
11. Test data true locations:
   - Alaska event (data/): 54.65°N, 159.67°W, depth 28 km
   - Luding event (example_data/): 29.67°N, 102.28°E, depth 10 km, M6.8

Language requirement:
- Always respond in English. Do not output Chinese paragraphs.

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
    else:
        template = '''请尽力用中文回答用户问题。你可以使用以下工具：

{tools}

''' + skill_context_block + '''

Important rules:
- Final Answer 必须使用中文（文件路径、表格内容、方法名可保留原样）。
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
- If the user asks to "plot" or "draw" the waveform while reading, set `plot=True` in the read tool arguments.
- Plotting IS supported; never tell the user that the system cannot draw waveforms.
- 当用户要求“查看某个 trace 的拾取结果”（如“trace 3 的拾取”）时，优先调用 `pick_first_arrivals` 并传 `trace_number`（1 基），不要只用 read/plot 工具。
- If the loaded file is HDF5, prefer get_hdf5_structure / read_hdf5_trace and use convert_hdf5_to_numpy/convert_hdf5_to_excel for conversions.
- CRITICAL: If a tool returns a Markdown table or an image link (e.g. `![...](...)`), you MUST copy it EXACTLY into your Final Answer. Do not summarize it.
- ALWAYS provide a brief textual summary of the key findings (e.g. best P-wave time, best S-wave time) in addition to the table/image.
- 当用户要求“连续波形监测”或“最近 N 小时数据”时，优先使用 run_continuous_monitoring。
- 当用户要求“按地点下载并定位”时（如"下载阿拉斯加地震数据"、"下载波形数据并定位"），优先使用 locate_place_data_nearseismic（传入 latitude、longitude）。
- 如果连续监测请求只给了日期和时长，优先按 UTC 推导 start/end，而不是停下来追问。
- 如果只给了范围框而没有中心点，使用范围中心作为 grid_center。
- 连续监测优先使用命名区域：“南加州”、“北加州”或“加州”。如果预估任务很大，也要直接继续执行，但要在最终回答里带上负载估计和进度摘要。
- 如果用户提到官方目录或数据源名称（如 USGS、SCEDC、NCEDC），要保留并传入对应的 catalog/client 参数，不要泛化成“官方目录”。
- 对监测请求，先给出简短任务摘要，再说明为什么任务大、建议怎样缩小范围。
- 如果用户提到具体地点、校园、地标或机构，要优先按地点中心理解，不要退回成大区域名词。
- 对监测请求，不要直接跳到工具调用。先把请求归一化为区域/时间/目录，预估负载，然后直接执行并汇报进度。
- 对监测请求，在第一次工具调用前写更长的 Thought：先复述请求，再识别区域/目录，估算负载，说明主要风险因素，并描述执行过程中的进度信息。
- 当 run_continuous_monitoring 返回 success 或 error 的 JSON 后，立即给出 Final Answer，不要再继续新的 Thought/Action 循环。
- 对监测请求，初始推理不要只写一句话，要拆成多句 Thought：1) 用户说的是哪个地点，2) 时间窗怎么理解，3) 任务是否过大，4) 下一步最稳妥的动作是什么。
- 你必须严格遵守 ReAct 格式。
- 如果需要调用工具，只能输出 Thought / Action / Action Input 三段。
- 如果已经可以给最终结果，必须输出 Final Answer: 开头。
- 不要在 Thought 后直接输出最终答案，不要混合 Final Answer 与 Action。

**震相拾取规则**:
- 默认使用深度学习方法（EQTransformer 和 PhaseNet）进行震相拾取，不需要指定 `methods` 参数。
- 只有当用户明确要求使用传统方法时（例如"使用传统方法"、"用 STA/LTA"），才在 `methods` 参数中包含传统方法（如 sta_lta, aic, pai_k）。

**地震定位工作流程**:
1. 首先使用 get_loaded_context 检查已加载的文件和拾取状态。如果用户提供了本地目录（如 "example_data"），先使用 load_local_data 加载数据。当用户说"本地数据"、"使用本地数据定位"或未指定路径但要求地震定位时，默认加载 "example_data/" 目录。
2. 如果加载了多个 MiniSEED 文件（多个台站数据），使用 pick_all_miniseed_files 批量拾取震相
3. 在近震定位前先执行 prepare_nearseismic_taup_cache，优先复用 TauP 缓存文件，缺失时自动构建。
4. 调用 add_station_coordinates 时传空参数 {{}}，会自动从 data/stations.json 或 example_data/stations.json 加载台站坐标
5. 对用户上传/本地数据，优先使用 locate_uploaded_data_nearseismic（必要时再回退 locate_earthquake）
6. 对“连续波形监测”或“最近 N 小时数据”请求，优先使用 run_continuous_monitoring
7. 对“指定地点下载并定位”请求，优先使用 locate_place_data_nearseismic（传入 latitude/longitude）
8. 如果用户明确要求经典定位器，再使用 locate_earthquake
9. 使用 plot_location_map 将定位结果和台站位置绘制在地图上（Cartopy）
10. 台站坐标自动加载，仅在自动加载失败时才需手动提供
11. 测试数据真实位置：
   - 阿拉斯加事件 (data/)：54.65°N, 159.67°W，深度 28 km
   - 泸定事件 (example_data/)：29.67°N, 102.28°E，深度 10 km，M6.8

语言要求：
- 不要输出英文段落或英文"Summary: ..."。如需总结，请使用中文"摘要：..."。

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
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=6,
        # LangChain Classic compatibility: some versions only support "force".
        early_stopping_method="force",
    )
