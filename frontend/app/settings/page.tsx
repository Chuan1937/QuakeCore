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

type ModelConfig = {
  backend: "deepseek" | "ollama";
  model: string;
  apiKey: string;
  baseUrl: string;
};

const DEEPSEEK_DEFAULTS: ModelConfig = {
  backend: "deepseek",
  model: "deepseek-v4-flash",
  apiKey: "",
  baseUrl: "https://api.deepseek.com",
};

const OLLAMA_DEFAULTS: ModelConfig = {
  backend: "ollama",
  model: "",
  apiKey: "",
  baseUrl: "http://localhost:11434",
};

export default function SettingsPage() {
  const router = useRouter();
  const [config, setConfig] = useState<ModelConfig>(DEEPSEEK_DEFAULTS);
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

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

        if (configData.provider === "ollama") {
          setConfig({
            backend: "ollama",
            model: configData.model_name,
            apiKey: "",
            baseUrl: configData.base_url || "http://localhost:11434",
          });
        } else {
          setConfig({
            backend: "deepseek",
            model: configData.model_name || "deepseek-v4-flash",
            apiKey: configData.api_key ?? "",
            baseUrl: configData.base_url || "https://api.deepseek.com",
          });
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
    setConfig(value === "deepseek" ? { ...DEEPSEEK_DEFAULTS } : { ...OLLAMA_DEFAULTS });
    setOllamaModels([]);
    setOllamaMessage("");
    setMessage("");
  }

  async function detectOllamaModels() {
    setDetecting(true);
    setOllamaMessage("正在检测本地 Ollama 模型...");

    try {
      const data = await getOllamaModels(config.baseUrl);
      setOllamaModels(data.models || []);
      setOllamaMessage(data.message || "");

      if (data.models?.length > 0) {
        setConfig((current) => ({ ...current, model: data.models[0] }));
      }
    } catch {
      setOllamaModels([]);
      setOllamaMessage("检测 Ollama 失败，请确认后端服务正常");
    } finally {
      setDetecting(false);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      await saveLlmConfig({
        provider: config.backend,
        model_name: config.model,
        api_key: config.backend === "deepseek" ? (config.apiKey || null) : null,
        base_url: config.baseUrl,
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
              value={config.backend}
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

          {config.backend === "deepseek" && (
            <>
              <label>
                <span>模型</span>
                <input
                  value={config.model}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, model: event.target.value }))
                  }
                  placeholder="deepseek-v4-flash"
                />
              </label>

              <label>
                <span>API Key</span>
                <input
                  type="password"
                  value={config.apiKey}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, apiKey: event.target.value }))
                  }
                  placeholder="可留空，默认读取 DEEPSEEK_API_KEY"
                />
              </label>

              <label>
                <span>Base URL</span>
                <input
                  value={config.baseUrl}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, baseUrl: event.target.value }))
                  }
                  placeholder="https://api.deepseek.com"
                />
              </label>
            </>
          )}

          {config.backend === "ollama" && (
            <>
              <label>
                <span>Ollama 地址</span>
                <input
                  value={config.baseUrl}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, baseUrl: event.target.value }))
                  }
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
                    value={config.model}
                    onChange={(event) =>
                      setConfig((current) => ({ ...current, model: event.target.value }))
                    }
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
