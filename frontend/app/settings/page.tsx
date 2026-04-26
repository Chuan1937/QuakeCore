"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getConfigDefaults,
  getLlmConfig,
  saveLlmConfig,
  type ConfigDefaults,
  type LlmConfig,
} from "@/lib/api";

const EMPTY_CONFIG: LlmConfig = {
  provider: "deepseek",
  model_name: "deepseek-v4-flash",
  api_key: "",
  base_url: "https://api.deepseek.com",
};

export default function SettingsPage() {
  const router = useRouter();
  const [config, setConfig] = useState<LlmConfig>(EMPTY_CONFIG);
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [message, setMessage] = useState<string>("");
  const [loading, setLoading] = useState(false);

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
        setConfig({
          provider: configData.provider,
          model_name: configData.model_name,
          api_key: configData.api_key ?? "",
          base_url: configData.base_url ?? "",
        });
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

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      await saveLlmConfig({
        provider: config.provider,
        model_name: config.model_name,
        api_key: config.api_key || null,
        base_url: config.base_url || null,
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
              value={config.provider}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  provider: event.target.value as LlmConfig["provider"],
                  model_name:
                    event.target.value === "deepseek"
                      ? "deepseek-v4-flash"
                      : current.model_name || "qwen2.5:3b",
                  base_url:
                    event.target.value === "deepseek"
                      ? "https://api.deepseek.com"
                      : current.base_url,
                }))
              }
            >
              {(defaults?.providers ?? ["deepseek", "ollama"]).map((provider) => (
                <option key={provider} value={provider}>
                  {provider === "deepseek" ? "DeepSeek API" : "Ollama 本地"}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>模型</span>
            <input
              value={config.model_name}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  model_name: event.target.value,
                }))
              }
              placeholder="deepseek-v4-flash"
            />
          </label>

          <label>
            <span>API Key</span>
            <input
              type="password"
              value={config.api_key ?? ""}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  api_key: event.target.value,
                }))
              }
              placeholder="可留空，默认读取 DEEPSEEK_API_KEY"
            />
          </label>

          <label>
            <span>Base URL</span>
            <input
              value={config.base_url ?? ""}
              onChange={(event) =>
                setConfig((current) => ({
                  ...current,
                  base_url: event.target.value,
                }))
              }
              placeholder="https://api.deepseek.com"
            />
          </label>

          {message ? <p className="settings-message">{message}</p> : null}

          <button type="submit" className="settings-submit" disabled={loading}>
            {loading ? "保存中..." : "保存并返回"}
          </button>
        </form>
      </section>
    </main>
  );
}
