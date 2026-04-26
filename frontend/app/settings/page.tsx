"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getConfigDefaults,
  getLlmConfig,
  getOllamaModels,
  saveLlmConfig,
  type ConfigDefaults,
  type LlmConfig,
} from "@/lib/api";

const EMPTY_DEEPSEEK = {
  provider: "deepseek" as const,
  model_name: "deepseek-v4-flash",
  api_key: "",
  base_url: "https://api.deepseek.com",
};

export default function SettingsPage() {
  const router = useRouter();
  const [backend, setBackend] = useState<"deepseek" | "ollama">("deepseek");
  const [model, setModel] = useState("deepseek-v4-flash");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("https://api.deepseek.com");
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaMessage, setOllamaMessage] = useState("");
  const [detecting, setDetecting] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        const [defaultsData, configData] = await Promise.all([
          getConfigDefaults(),
          getLlmConfig(),
        ]);
        if (!mounted) return;

        setDefaults(defaultsData);
        const provider = configData.provider;
        setBackend(provider);
        setModel(configData.model_name);

        if (provider === "ollama") {
          const url = configData.base_url || "http://localhost:11434";
          setOllamaUrl(url);
          setBaseUrl(url);
          setApiKey("");
        } else {
          setApiKey(configData.api_key ?? "");
          setBaseUrl(configData.base_url || "https://api.deepseek.com");
          setOllamaUrl("http://localhost:11434");
        }
      } catch (error) {
        if (!mounted) return;
        setMessage(error instanceof Error ? error.message : "加载设置失败。");
      }
    }

    void load();
    return () => {
      mounted = false;
    };
  }, []);

  function handleBackendChange(value: "deepseek" | "ollama") {
    setBackend(value);
    setMessage("");

    if (value === "deepseek") {
      setModel("deepseek-v4-flash");
      setBaseUrl("https://api.deepseek.com");
      setApiKey("");
      setOllamaModels([]);
      setOllamaMessage("");
    } else {
      setModel("");
      setBaseUrl(ollamaUrl);
      setApiKey("");
      setOllamaModels([]);
      setOllamaMessage("");
    }
  }

  async function detectOllamaModels() {
    setDetecting(true);
    setOllamaMessage("正在检测本地 Ollama 模型...");

    try {
      const data = await getOllamaModels(ollamaUrl);
      setOllamaModels(data.models || []);
      setOllamaMessage(data.message || "");

      if (data.models?.length > 0) {
        setModel(data.models[0]);
      }
    } catch {
      setOllamaModels([]);
      setOllamaMessage("检测 Ollama 失败，请确认后端服务正常");
    } finally {
      setDetecting(false);
    }
  }

  function updateOllamaUrl(url: string) {
    setOllamaUrl(url);
    setBaseUrl(url);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      await saveLlmConfig({
        provider: backend,
        model_name: model,
        api_key: backend === "deepseek" ? (apiKey || null) : null,
        base_url: backend === "deepseek" ? baseUrl : ollamaUrl,
      });
      router.push("/");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存失败。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="settings-shell">
      <section className="settings-card">
        <div className="settings-header">
          <button
            type="button"
            className="settings-back"
            onClick={() => router.push("/")}
            aria-label="返回聊天"
          >
            ←
          </button>
          <div>
            <h1>模型设置</h1>
            <p>选择 DeepSeek API 或本地 Ollama。</p>
          </div>
        </div>

        <form className="settings-form" onSubmit={handleSubmit}>
          <label>
            <span>后端</span>
            <select
              value={backend}
              onChange={(event) =>
                handleBackendChange(event.target.value as "deepseek" | "ollama")
              }
            >
              {(defaults?.providers ?? ["deepseek", "ollama"]).map((p) => (
                <option key={p} value={p}>
                  {p === "deepseek" ? "DeepSeek API" : "Ollama 本地"}
                </option>
              ))}
            </select>
          </label>

          {backend === "deepseek" && (
            <>
              <label>
                <span>模型</span>
                <input
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  placeholder="deepseek-v4-flash"
                />
              </label>

              <label>
                <span>API Key</span>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder="可留空，默认读取 DEEPSEEK_API_KEY"
                />
              </label>

              <label>
                <span>Base URL</span>
                <input
                  value={baseUrl}
                  onChange={(event) => setBaseUrl(event.target.value)}
                  placeholder="https://api.deepseek.com"
                />
              </label>
            </>
          )}

          {backend === "ollama" && (
            <>
              <label>
                <span>Ollama 地址</span>
                <input
                  value={ollamaUrl}
                  onChange={(event) => updateOllamaUrl(event.target.value)}
                  placeholder="http://localhost:11434"
                />
              </label>

              <button
                type="button"
                className="settings-detect-btn"
                onClick={detectOllamaModels}
                disabled={detecting}
              >
                {detecting ? "检测中..." : "检测本地模型"}
              </button>

              <label>
                <span>本地模型</span>
                {ollamaModels.length > 0 ? (
                  <select
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                  >
                    {ollamaModels.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <p className="settings-ollama-empty">
                    {ollamaMessage || "未检测到 Ollama 模型"}
                  </p>
                )}
              </label>
            </>
          )}

          {message ? <p className="settings-message">{message}</p> : null}

          <button type="submit" className="settings-submit" disabled={loading}>
            {loading ? "保存中..." : "保存并返回"}
          </button>
        </form>
      </section>
    </main>
  );
}
