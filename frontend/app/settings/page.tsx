"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/lib/i18n";
import { LanguageToggle } from "@/components/language-toggle";
import {
  getConfigDefaults,
  getLlmConfig,
  getOllamaModels,
  saveLlmConfig,
  getApiBaseUrl,
  setApiBaseUrl,
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
  const { t } = useLanguage();
  const router = useRouter();
  const [config, setConfig] = useState<ModelConfig>(DEEPSEEK_DEFAULTS);
  const [defaults, setDefaults] = useState<ConfigDefaults | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaMessage, setOllamaMessage] = useState("");
  const [detecting, setDetecting] = useState(false);

  const [backendUrl, setBackendUrl] = useState("http://127.0.0.1:8000");
  const [backendUrlMessage, setBackendUrlMessage] = useState("");

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

        setBackendUrl(getApiBaseUrl());
      } catch (error) {
        if (!mounted) return;
        setMessage(error instanceof Error ? error.message : t("load_settings_failed"));
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
    setOllamaMessage(t("detecting_ollama"));

    try {
      const data = await getOllamaModels(config.baseUrl);
      setOllamaModels(data.models || []);
      setOllamaMessage(data.message || "");

      if (data.models?.length > 0) {
        setConfig((current) => ({ ...current, model: data.models[0] }));
      }
    } catch {
      setOllamaModels([]);
      setOllamaMessage(t("detect_failed"));
    } finally {
      setDetecting(false);
    }
  }

  function handleBackendUrlSave() {
    const trimmed = backendUrl.trim();
    if (!trimmed) {
      setBackendUrlMessage(t("backend_url_empty"));
      return;
    }
    try {
      new URL(trimmed);
    } catch {
      setBackendUrlMessage(t("backend_url_invalid"));
      return;
    }
    setApiBaseUrl(trimmed);
    setBackendUrlMessage(t("backend_url_saved"));
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
      setMessage(error instanceof Error ? error.message : t("save_failed"));
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
            aria-label={t("back_to_chat")}
          >
            ←
          </button>
          <div>
            <h1>{t("model_settings")}</h1>
            <p>{t("settings_subtitle")}</p>
          </div>
          <div style={{ marginLeft: "auto" }}>
            <LanguageToggle />
          </div>
        </div>

        <form className="settings-form" onSubmit={handleSubmit}>
          <label>
            <span>{t("backend_label")}</span>
            <select
              value={config.backend}
              onChange={(event) =>
                handleBackendChange(event.target.value as "deepseek" | "ollama")
              }
            >
              {(defaults?.providers ?? ["deepseek", "ollama"]).map((p) => (
                <option key={p} value={p}>
                  {p === "deepseek" ? t("deepseek_api") : t("ollama_local")}
                </option>
              ))}
            </select>
          </label>

          {config.backend === "deepseek" && (
            <>
              <label>
                <span>{t("model_label")}</span>
                <input
                  value={config.model}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, model: event.target.value }))
                  }
                  placeholder="deepseek-v4-flash"
                />
              </label>

              <label>
                <span>{t("api_key_label")}</span>
                <input
                  type="password"
                  value={config.apiKey}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, apiKey: event.target.value }))
                  }
                  placeholder={t("api_key_placeholder")}
                />
              </label>

              <label>
                <span>{t("base_url_label")}</span>
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
                <span>{t("ollama_url")}</span>
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
                {detecting ? t("detecting") : t("detect_local_models")}
              </button>

              <label>
                <span>{t("local_model")}</span>
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
                    {ollamaMessage || t("no_models")}
                  </p>
                )}
              </label>
            </>
          )}

          {message ? <p className="settings-message">{message}</p> : null}

          <button type="submit" className="settings-submit" disabled={loading}>
            {loading ? t("saving") : t("save_and_return")}
          </button>
        </form>

        <div className="settings-form" style={{ marginTop: "2rem", borderTop: "1px solid var(--border)", paddingTop: "1.5rem" }}>
          <h2 style={{ marginBottom: "1rem", fontSize: "1.1rem" }}>{t("backend_connection")}</h2>
          <label>
            <span>{t("backend_url_label")}</span>
            <input
              value={backendUrl}
              onChange={(event) => {
                setBackendUrl(event.target.value);
                setBackendUrlMessage("");
              }}
              placeholder="http://127.0.0.1:8000"
            />
          </label>
          <button type="button" className="settings-detect-btn" onClick={handleBackendUrlSave}>
            {t("save_backend_url")}
          </button>
          {backendUrlMessage ? <p className="settings-message">{backendUrlMessage}</p> : null}
          <p className="settings-ollama-empty" style={{ marginTop: "0.5rem" }}>
            {t("backend_url_hint")}
          </p>
        </div>
      </section>
    </main>
  );
}
