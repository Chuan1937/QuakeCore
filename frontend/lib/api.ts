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

export type ChatResponse = {
  session_id: string;
  answer: string;
  error: string | null;
  route: string;
  artifacts: ChatArtifact[];
  workflow?: WorkflowResult | null;
};

export type ChatRequest = {
  message: string;
  session_id?: string | null;
  lang?: "en" | "zh";
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
