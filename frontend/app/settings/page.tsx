"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/nav";
import {
  getConfigDefaults,
  getLlmConfig,
  saveLlmConfig,
  type ConfigDefaults,
  type LlmConfig,
} from "@/lib/api";

const EMPTY_CONFIG: LlmConfig = {
  provider: "deepseek",
  model_name: "",
  api_key: "",
  base_url: "",
};

export default function SettingsPage() {
  const [config, setConfig] = useState<LlmConfig>(EMPTY_CONFIG);
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [message, setMessage] = useState<string>("Loading...");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        const [defaultsData, configData] = await Promise.all([getConfigDefaults(), getLlmConfig()]);
        if (!mounted) {
          return;
        }
        setDefaults(defaultsData);
        setConfig({
          provider: configData.provider,
          model_name: configData.model_name,
          api_key: configData.api_key ?? "",
          base_url: configData.base_url ?? "",
        });
        setMessage("Loaded");
      } catch (error) {
        if (!mounted) {
          return;
        }
        const text = error instanceof Error ? error.message : "Failed to load settings.";
        setMessage(text);
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
    setMessage("Saving...");

    try {
      const saved = await saveLlmConfig({
        provider: config.provider,
        model_name: config.model_name,
        api_key: config.api_key || null,
        base_url: config.base_url || null,
      });
      setConfig({
        provider: saved.provider,
        model_name: saved.model_name,
        api_key: saved.api_key ?? "",
        base_url: saved.base_url ?? "",
      });
      setMessage("Saved to data/config/llm_config.json");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to save settings.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <Nav />
      <section className="hero compact">
        <div className="hero-copy">
          <p className="eyebrow">Settings</p>
          <h1>LLM configuration</h1>
          <p className="lede">
            Configure provider, model, and credentials. The values are saved to
            <code>data/config/llm_config.json</code>.
          </p>
        </div>
        <div className="status-card">
          <span>Status</span>
          <strong>{message}</strong>
          <span>{defaults ? `Providers: ${defaults.providers.join(", ")}` : "Loading defaults..."}</span>
        </div>
      </section>

      <section className="panel">
        <form className="form-grid" onSubmit={handleSubmit}>
          <label>
            <span>Provider</span>
            <select
              value={config.provider}
              onChange={(event) => setConfig((current) => ({ ...current, provider: event.target.value as LlmConfig["provider"] }))}
            >
              {defaults?.providers.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Model name</span>
            <input
              value={config.model_name}
              onChange={(event) => setConfig((current) => ({ ...current, model_name: event.target.value }))}
              placeholder="deepseek-v4-flash"
            />
          </label>

          <label>
            <span>API key</span>
            <input
              type="password"
              value={config.api_key ?? ""}
              onChange={(event) => setConfig((current) => ({ ...current, api_key: event.target.value }))}
              placeholder="Optional"
            />
          </label>

          <label>
            <span>Base URL</span>
            <input
              value={config.base_url ?? ""}
              onChange={(event) => setConfig((current) => ({ ...current, base_url: event.target.value }))}
              placeholder="https://api.deepseek.com"
            />
          </label>

          <div className="form-actions">
            <button type="submit" disabled={loading}>
              {loading ? "Saving..." : "Save"}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
