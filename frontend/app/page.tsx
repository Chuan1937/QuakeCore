"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { MarkdownView } from "@/components/markdown-view";
import { WorkflowSteps } from "@/components/workflow-steps";
import {
  chatWithAgent,
  getLlmConfig,
  saveLlmConfig,
  toBackendUrl,
  uploadFile,
  type ChatArtifact,
  type ChatResponse,
  type LlmConfig,
  type WorkflowResult,
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  route?: string;
  artifacts?: ChatArtifact[];
  error?: string | null;
  workflow?: WorkflowResult | null;
  files?: Array<{ name: string; fileType?: string }>;
  pending?: boolean;
};

type Thread = {
  id: string;
  title: string;
  sessionId: string | null;
  messages: Message[];
};

const UPLOAD_ACCEPT =
  ".mseed,.miniseed,.sac,.sgy,.segy,.h5,.hdf5,.npy,.npz,.csv,.txt";

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function inferPendingText(text: string): string {
  const source = text.toLowerCase();
  if (
    source.includes("初至") ||
    source.includes("拾取") ||
    source.includes("震相") ||
    source.includes("p波") ||
    source.includes("s波")
  ) {
    return "正在分析波形并进行初至/震相拾取…";
  }
  if (source.includes("定位") || source.includes("震中") || source.includes("震源")) {
    return "正在执行地震定位工作流…";
  }
  if (source.includes("结构") || source.includes("采样率") || source.includes("header")) {
    return "正在读取文件结构…";
  }
  if (source.includes("连续") || source.includes("监测")) {
    return "正在准备连续地震监测任务…";
  }
  return "正在思考…";
}

export default function HomePage() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [llmConfig, setLlmConfig] = useState<LlmConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragCounter = useRef(0);

  const canSend = input.trim().length > 0 && !chatLoading && !uploading;

  useEffect(() => {
    async function loadConfig() {
      try {
        const config = await getLlmConfig();
        setLlmConfig(config);
      } catch {
        setLlmConfig({
          provider: "deepseek",
          model_name: "deepseek-v4-flash",
          api_key: null,
          base_url: "https://api.deepseek.com",
        });
      }
    }

    void loadConfig();
  }, []);

  const persistCurrentThread = useCallback(() => {
    if (!messages.length) {
      return;
    }
    const title = messages.find((item) => item.role === "user")?.content.slice(0, 24) || "未命名会话";
    setThreads((current) => {
      const snapshot: Thread = {
        id: sessionId || newId(),
        title,
        sessionId,
        messages: [...messages],
      };
      const next = current.filter((item) => item.id !== snapshot.id);
      return [snapshot, ...next].slice(0, 12);
    });
  }, [messages, sessionId]);

  const handleUploadFiles = useCallback(
    async (inputFiles: File[] | FileList) => {
      const files = Array.from(inputFiles);
      if (files.length === 0) {
        return;
      }

      setUploading(true);
      setError(null);
      let currentSession = sessionId;

      for (const file of files) {
        setMessages((current) => [
          ...current,
          {
            id: newId(),
            role: "user",
            content: `上传了 ${file.name}`,
            files: [{ name: file.name }],
          },
        ]);

        try {
          const uploaded = await uploadFile(file, currentSession);
          currentSession = uploaded.session_id;
          setMessages((current) => [
            ...current,
            {
              id: newId(),
              role: "assistant",
              content: uploaded.bound_to_agent
                ? `已接收 ${uploaded.filename}，识别为 ${uploaded.file_type}，已绑定到当前会话。你可以直接问我：分析文件结构、初至拾取或地震定位。`
                : "已接收文件，但该类型暂未自动绑定为当前地震数据。",
              route: "file_upload",
              files: [{ name: uploaded.filename, fileType: uploaded.file_type }],
            },
          ]);
        } catch (err) {
          const message = err instanceof Error ? err.message : "上传失败。";
          setError(message);
          setMessages((current) => [
            ...current,
            {
              id: newId(),
              role: "assistant",
              content: "文件上传失败。",
              error: message,
            },
          ]);
        }
      }

      setSessionId(currentSession);
      setUploading(false);
    },
    [sessionId],
  );

  async function handleSend(messageText: string) {
    const text = messageText.trim();
    if (!text || chatLoading || uploading) {
      return;
    }

    setChatLoading(true);
    setError(null);
    const pendingId = newId();
    let pendingShown = false;
    setMessages((current) => [...current, { id: newId(), role: "user", content: text }]);
    const timer = setTimeout(() => {
      pendingShown = true;
      setMessages((current) => [
        ...current,
        { id: pendingId, role: "assistant", content: inferPendingText(text), pending: true },
      ]);
    }, 1000);

    try {
      const response: ChatResponse = await chatWithAgent({
        message: text,
        session_id: sessionId,
        lang: "zh",
      });
      clearTimeout(timer);

      setSessionId(response.session_id);
      if (pendingShown) {
        setMessages((current) =>
          current.map((item) =>
            item.id === pendingId
              ? {
                  ...item,
                  pending: false,
                  content: response.answer || response.error || "No response returned.",
                  route: response.route,
                  artifacts: response.artifacts,
                  error: response.error,
                  workflow: response.workflow ?? null,
                }
              : item,
          ),
        );
      } else {
        setMessages((current) => [
          ...current,
          {
            id: newId(),
            role: "assistant",
            content: response.answer || response.error || "No response returned.",
            route: response.route,
            artifacts: response.artifacts,
            error: response.error,
            workflow: response.workflow ?? null,
          },
        ]);
      }
    } catch (err) {
      clearTimeout(timer);
      const message = err instanceof Error ? err.message : "Request failed.";
      setError(message);
      if (pendingShown) {
        setMessages((current) =>
          current.map((item) =>
            item.id === pendingId
              ? { ...item, pending: false, content: "请求失败。", error: message }
              : item,
          ),
        );
      } else {
        setMessages((current) => [
          ...current,
          { id: newId(), role: "assistant", content: "请求失败。", error: message },
        ]);
      }
    } finally {
      setChatLoading(false);
      setInput("");
    }
  }

  function handleNewChat() {
    persistCurrentThread();
    setSessionId(null);
    setMessages([]);
    setInput("");
    setError(null);
  }

  useEffect(() => {
    function onPaste(event: ClipboardEvent) {
      const items = Array.from(event.clipboardData?.items || []);
      const files = items
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile())
        .filter((file): file is File => Boolean(file));
      if (files.length > 0) {
        void handleUploadFiles(files);
      }
    }

    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [handleUploadFiles]);

  async function switchProvider(provider: "deepseek" | "ollama") {
    if (!llmConfig) {
      return;
    }
    const updated: LlmConfig =
      provider === "deepseek"
        ? { provider: "deepseek", model_name: "deepseek-v4-flash", api_key: llmConfig.api_key, base_url: "https://api.deepseek.com" }
        : { provider: "ollama", model_name: "", api_key: null, base_url: "http://localhost:11434" };
    try {
      const saved = await saveLlmConfig(updated);
      setLlmConfig(saved);
      setModelOpen(false);
    } catch {
      setError("切换模型配置失败。");
    }
  }

  return (
    <main
      className={`app-shell ${dragging ? "dragging" : ""}`}
      onDragEnter={(event) => {
        if (!event.dataTransfer.types.includes("Files")) {
          return;
        }
        dragCounter.current += 1;
        setDragging(true);
      }}
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes("Files")) {
          return;
        }
        event.preventDefault();
      }}
      onDragLeave={(event) => {
        if (!event.dataTransfer.types.includes("Files")) {
          return;
        }
        dragCounter.current = Math.max(0, dragCounter.current - 1);
        if (dragCounter.current === 0) {
          setDragging(false);
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        dragCounter.current = 0;
        setDragging(false);
        if (event.dataTransfer.files?.length) {
          void handleUploadFiles(event.dataTransfer.files);
        }
      }}
    >
      <aside className="chat-sidebar">
        <div className="model-picker">
          <button type="button" className="model-button" onClick={() => setModelOpen((v) => !v)}>
            QuakeCore ▾
          </button>
          {modelOpen ? (
            <div className="model-menu">
              <p className="model-menu-title">模型 / 后端</p>
              <button type="button" className="model-option" onClick={() => void switchProvider("deepseek")}>
                {llmConfig?.provider === "deepseek" ? "●" : "○"} DeepSeek API
                <span>model: {llmConfig?.provider === "deepseek" ? llmConfig.model_name : "-"}</span>
              </button>
              <button type="button" className="model-option" onClick={() => void switchProvider("ollama")}>
                {llmConfig?.provider === "ollama" ? "●" : "○"} Ollama 本地
                <span>model: {llmConfig?.provider === "ollama" ? llmConfig.model_name : "-"}</span>
              </button>
              <Link href="/settings" className="model-settings-link">
                打开 Settings
              </Link>
            </div>
          ) : null}
        </div>

        <button type="button" className="new-chat-btn" onClick={handleNewChat}>
          + New Chat
        </button>

        <div className="session-list">
          <p>最近会话</p>
          <button type="button" className="session-item active">
            当前会话
          </button>
          {threads.map((thread) => (
            <button
              key={thread.id}
              type="button"
              className="session-item"
              onClick={() => {
                persistCurrentThread();
                setSessionId(thread.sessionId);
                setMessages(thread.messages);
              }}
            >
              {thread.title}
            </button>
          ))}
        </div>

        <div className="sidebar-bottom">
          <Link href="/skills">Skills</Link>
          <small>
            当前后端：
            {llmConfig?.provider === "ollama" ? "Ollama" : "DeepSeek v4"}
          </small>
        </div>
      </aside>

      <section className="chat-main">
        <div className="chat-main-header">QuakeCore</div>
        <div className="chat-log" aria-live="polite">
          <div className="messages-inner">
            {messages.length === 0 ? (
              <div className="chat-empty-state">
                <h2>今天要分析什么地震数据？</h2>
                <p>上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。</p>
              </div>
            ) : (
              messages.map((message) => (
                <article key={message.id} className={`message-row ${message.role}`}>
                  {message.role === "user" ? (
                    <div className="user-bubble">
                      <p>{message.content}</p>
                      {message.files?.length ? (
                        <div className="file-chip-row">
                          {message.files.map((file) => (
                            <span key={`${file.name}-${file.fileType || ""}`} className="file-chip">
                              {file.name}
                              {file.fileType ? ` · ${file.fileType}` : ""}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : message.pending ? (
                    <div className="assistant-message">
                      <span className="pending-text">
                        {message.content}
                        <span className="dot-wave">
                          <i />
                          <i />
                          <i />
                        </span>
                      </span>
                    </div>
                  ) : (
                    <div className="assistant-message">
                      <MarkdownView content={message.content} />
                      {message.files?.length ? (
                        <div className="file-chip-row">
                          {message.files.map((file) => (
                            <span key={`${file.name}-${file.fileType || ""}`} className="file-chip">
                              {file.name}
                              {file.fileType ? ` · ${file.fileType}` : ""}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      {message.workflow ? <WorkflowSteps workflow={message.workflow} /> : null}
                      {message.error ? <div className="error-pill">{message.error}</div> : null}
                      {message.route ? <div className="route-meta">{message.route}</div> : null}
                      {message.artifacts?.length ? (
                        <div className="artifacts">
                          {message.artifacts.map((artifact) => (
                            <div key={artifact.url} className="artifact-card">
                              {artifact.type === "image" ? (
                                <a
                                  href={toBackendUrl(artifact.url)}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="artifact-preview"
                                >
                                  <img src={toBackendUrl(artifact.url)} alt={artifact.name} />
                                </a>
                              ) : (
                                <a href={toBackendUrl(artifact.url)} target="_blank" rel="noreferrer">
                                  {artifact.name}
                                </a>
                              )}
                              <div className="artifact-meta">
                                <strong>{artifact.name}</strong>
                                <span>{artifact.path}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  )}
                </article>
              ))
            )}
          </div>
        </div>

        <div className="composer-shell">
          <form
            className="chat-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSend(input);
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={UPLOAD_ACCEPT}
              className="hidden-file-input"
              onChange={(event) => {
                if (event.target.files?.length) {
                  void handleUploadFiles(event.target.files);
                }
                event.target.value = "";
              }}
            />
            <button
              type="button"
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              aria-label="上传附件"
              title="上传附件"
              disabled={uploading}
            >
              +
            </button>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Message QuakeCore"
              rows={2}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend(input);
                }
              }}
            />
            <button type="submit" disabled={!canSend} className="send-btn">
              {chatLoading ? "…" : "↑"}
            </button>
          </form>
          {error ? <div className="composer-error">{error}</div> : null}
        </div>
      </section>
      {dragging ? <div className="drop-overlay">释放以上传地震数据文件</div> : null}
    </main>
  );
}
