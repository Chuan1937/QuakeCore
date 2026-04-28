export type ChatArtifact = {
  type: string;
  name: string;
  path: string;
  url: string;
};

export type WorkflowStep = {
  name: string;
  status: "ok" | "warning" | "error" | "skipped" | string;
  required?: boolean;
  message?: string;
  error?: string | null;
  duration_ms?: number;
};

export type WorkflowResult = {
  status: "success" | "partial_success" | "failed" | string;
  summary?: string;
  message?: string;
  steps?: WorkflowStep[];
  location?: Record<string, unknown>;
  artifacts?: ChatArtifact[];
  error?: string | null;
};

export type ContinuousJobStartResponse = {
  job_id: string;
  session_id: string;
  status: "running" | string;
};

export type ContinuousJobProgressResponse = {
  job_id: string;
  session_id?: string | null;
  status: "running" | "completed" | "failed" | string;
  step?: string;
  percent?: number;
  logs?: Array<{
    stage?: string;
    message?: string;
    downloaded?: number;
    failed?: number;
    total?: number;
    timestamp?: string;
  }>;
  error?: string | null;
};

export type ContinuousJobResultResponse = {
  job_id: string;
  session_id?: string | null;
  status: "running" | "completed" | "failed" | string;
  result?: {
    success?: boolean;
    message?: string;
    artifacts?: ChatArtifact[];
    data?: Record<string, unknown>;
    error?: string | null;
  };
  step?: string;
  percent?: number;
  error?: string | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  error: string | null;
  route: string;
  artifacts: ChatArtifact[];
  workflow?: WorkflowResult | null;
  opencode_admin?: boolean;
  data?: Record<string, unknown>;
};

export type ChatRequest = {
  message: string;
  session_id?: string | null;
  lang?: "en" | "zh";
  attachments?: Array<{
    name: string;
    path: string;
    file_type?: string;
  }>;
};

export type FileUploadResponse = {
  session_id: string;
  filename: string;
  path: string;
  file_type: string;
  bound_to_agent: boolean;
};

export type LlmConfig = {
  provider: "deepseek" | "ollama";
  model_name: string;
  api_key: string | null;
  base_url: string | null;
};

export type ConfigDefaults = {
  providers: Array<"deepseek" | "ollama">;
  default_llm_config: LlmConfig;
  provider_defaults: Record<string, LlmConfig>;
};

export type SkillSummary = {
  name: string;
  path: string;
};

export type SkillDetail = {
  name: string;
  path: string;
  content: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export function toBackendUrl(url: string): string {
  const base = API_BASE_URL;
  if (!url) {
    return url;
  }
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  if (url.startsWith("/")) {
    return `${base}${url}`;
  }
  return `${base}/${url}`;
}

export async function chatWithAgent(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as ChatResponse;
}

export type ProgressEvent = {
  type: string;
  icon?: string;
  status?: string;
  summary?: string;
  detail?: string;
  timestamp?: number;
  tool?: string;
};

export type StreamEvent = {
  type: "status" | "progress" | "final";
  message?: string;
  event?: ProgressEvent;
  response?: {
    session_id: string;
    answer: string;
    error: string | null;
    route: string;
    artifacts: ChatArtifact[];
    workflow: WorkflowResult | null;
    opencode_admin?: boolean;
    data?: Record<string, unknown>;
  };
};

export async function* chatWithAgentStream(
  payload: ChatRequest,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }

  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const parseChunk = (chunk: string): StreamEvent | null => {
    const line = chunk
      .split("\n")
      .find((item) => item.startsWith("data:"));

    if (!line) return null;

    const json = line.slice("data:".length).trim();
    if (!json) return null;

    try {
      return JSON.parse(json) as StreamEvent;
    } catch {
      return null;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      const tailEvent = parseChunk(buffer);
      if (tailEvent) {
        yield tailEvent;
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";

    for (const chunk of chunks) {
      const event = parseChunk(chunk);
      if (event) {
        yield event;
      }
    }
  }
}

export async function uploadFile(
  file: File,
  sessionId?: string | null,
): Promise<FileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const response = await fetch(`${API_BASE_URL}/api/files/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status}`);
  }

  return (await response.json()) as FileUploadResponse;
}

export async function getConfigDefaults(): Promise<ConfigDefaults> {
  const response = await fetch(`${API_BASE_URL}/api/config/defaults`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as ConfigDefaults;
}

export async function getLlmConfig(): Promise<LlmConfig> {
  const response = await fetch(`${API_BASE_URL}/api/config/llm`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as LlmConfig;
}

export async function saveLlmConfig(payload: LlmConfig): Promise<LlmConfig> {
  const response = await fetch(`${API_BASE_URL}/api/config/llm`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as LlmConfig;
}

export async function listSkills(): Promise<{ skills: SkillSummary[] }> {
  const response = await fetch(`${API_BASE_URL}/api/skills`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as { skills: SkillSummary[] };
}

export type OllamaModelsResponse = {
  ok: boolean;
  models: string[];
  message: string;
};

export async function getOllamaModels(baseUrl: string): Promise<OllamaModelsResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/ollama/models?base_url=${encodeURIComponent(baseUrl)}`,
    { cache: "no-store" },
  );

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as OllamaModelsResponse;
}

export async function getSkill(name: string): Promise<SkillDetail> {
  const response = await fetch(`${API_BASE_URL}/api/skills/${encodeURIComponent(name)}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as SkillDetail;
}

export async function startContinuousWorkflow(payload: {
  message: string;
  session_id?: string | null;
  lang?: "en" | "zh";
  params?: Record<string, unknown>;
}): Promise<ContinuousJobStartResponse> {
  const response = await fetch(`${API_BASE_URL}/api/workflows/continuous/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as ContinuousJobStartResponse;
}

export async function getContinuousWorkflowProgress(
  jobId: string,
): Promise<ContinuousJobProgressResponse> {
  const response = await fetch(`${API_BASE_URL}/api/workflows/continuous/${encodeURIComponent(jobId)}/progress`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as ContinuousJobProgressResponse;
}

export async function getContinuousWorkflowResult(
  jobId: string,
): Promise<ContinuousJobResultResponse> {
  const response = await fetch(`${API_BASE_URL}/api/workflows/continuous/${encodeURIComponent(jobId)}/result`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as ContinuousJobResultResponse;
}
