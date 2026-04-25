export type ChatArtifact = {
  type: string;
  url: string;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  error: string | null;
  route: string;
  artifacts: ChatArtifact[];
};

export type ChatRequest = {
  message: string;
  session_id?: string | null;
  lang?: "en" | "zh";
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
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

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
