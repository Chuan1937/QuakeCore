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

type ChatAttachment = {
  id: string;
  name: string;
  path?: string;
  fileKind?: string;
  status: "uploading" | "uploaded" | "failed";
  error?: string;
};

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
  attachments?: ChatAttachment[];
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

function basename(path: string): string {
  const cleaned = String(path || "").replace(/\\/g, "/");
  const parts = cleaned.split("/");
  return parts[parts.length - 1] || cleaned;
}

function stripUuidPrefix(name: string): string {
  return name.replace(/^[a-f0-9]{16,}_/i, "");
}

function getArtifactDisplayName(artifact: ChatArtifact): string {
  const candidate = basename(artifact.name || artifact.path || artifact.url);
  return stripUuidPrefix(candidate);
}

function ArtifactMessageCard({ artifact }: { artifact: ChatArtifact }) {
  const url = toBackendUrl(artifact.url);
  const displayName = getArtifactDisplayName(artifact);
  const rawPath = artifact.path || artifact.name || "";
  const [copied, setCopied] = useState(false);

  async function handleDownload() {
    const response = await fetch(url);
    if (!response.ok) {
      alert("文件不存在或下载失败");
      return;
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = displayName;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(objectUrl);
  }

  async function handleCopy() {
    try {
      if (artifact.type === "image") {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error("image fetch failed");
        }

        const blob = await response.blob();

        if (
          typeof ClipboardItem !== "undefined" &&
          navigator.clipboard &&
          blob.type.startsWith("image/")
        ) {
          await navigator.clipboard.write([
            new ClipboardItem({
              [blob.type]: blob,
            }),
          ]);
        } else {
          await navigator.clipboard.writeText(url);
        }
      } else {
        await navigator.clipboard.writeText(url);
      }

      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      alert("复制失败");
    }
  }

  const isImage = artifact.type === "image";

  return (
    <div className="artifact-message-card">
      {isImage ? (
        <button
          type="button"
          className="artifact-image-link"
          onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
          title="查看大图"
        >
          <img src={url} alt={displayName} />
        </button>
      ) : (
        <div className="artifact-file-box">
          <span className="artifact-file-icon">📄</span>
          <span>{displayName}</span>
        </div>
      )}

      <div className="artifact-card-footer">
        <div className="artifact-card-title">{displayName}</div>

        {rawPath ? (
          <div className="artifact-card-path">
            {stripUuidPrefix(basename(rawPath))}
          </div>
        ) : null}

        <div className="artifact-card-actions">
          <button type="button" onClick={() => void handleDownload()}>
            下载
          </button>

          {isImage ? (
            <>
              <button
                type="button"
                onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
              >
                查看
              </button>

              <button type="button" onClick={() => void handleCopy()}>
                复制
              </button>
            </>
          ) : null}
        </div>

        {copied ? <div className="copy-toast">复制成功</div> : null}
      </div>
    </div>
  );
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
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragCounter = useRef(0);

  const canSend = (input.trim().length > 0 || pendingAttachments.length > 0) && !chatLoading && !uploading;

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
      if (files.length === 0) return;

      setUploading(true);
      setError(null);
      let currentSession = sessionId;

      const attachments: ChatAttachment[] = files.map((f) => ({
        id: newId(),
        name: f.name,
        status: "uploading" as const,
      }));

      setPendingAttachments((prev) => [...prev, ...attachments]);

      for (const [i, file] of files.entries()) {
        try {
          const result = await uploadFile(file, currentSession);
          currentSession = result.session_id;
          setPendingAttachments((prev) =>
            prev.map((a) =>
              a.id === attachments[i].id
                ? { ...a, status: "uploaded", fileKind: result.file_type, path: result.path }
                : a,
            ),
          );
        } catch (err) {
          setPendingAttachments((prev) =>
            prev.map((a) =>
              a.id === attachments[i].id
                ? { ...a, status: "failed", error: err instanceof Error ? err.message : "上传失败" }
                : a,
            ),
          );
        }
      }

      setSessionId(currentSession);
      setUploading(false);
    },
    [sessionId],
  );

  async function handleSend(messageText: string) {
    const text = messageText.trim();
    const hasAttachments = pendingAttachments.filter((a) => a.status === "uploaded").length > 0;
    if ((!text && !hasAttachments) || chatLoading || uploading) {
      return;
    }

    const userAttachments: ChatAttachment[] = pendingAttachments.filter(
      (a) => a.status === "uploaded",
    );

    setChatLoading(true);
    setError(null);
    const pendingId = newId();
    let pendingShown = false;
    setMessages((current) => [
      ...current,
      { id: newId(), role: "user", content: text, attachments: userAttachments },
    ]);
    setPendingAttachments([]);
    setInput("");

    if (!text) {
      setChatLoading(false);
      return;
    }

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
        attachments: userAttachments
          .filter((a): a is ChatAttachment & { path: string } => Boolean(a.path))
          .map((a) => ({
            name: a.name,
            path: a.path,
            file_type: a.fileKind,
          })),
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
                      {message.attachments?.length ? (
                        <div className="msg-attach-row">
                          {message.attachments.map((file) => (
                            <div key={file.id} className="msg-attach-card">
                              <span className="msg-attach-name">{file.name}</span>
                              {file.fileKind ? <span className="msg-attach-kind">{file.fileKind}</span> : null}
                            </div>
                          ))}
                        </div>
                      ) : null}
                      {message.content ? <p>{message.content}</p> : null}
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
                    <div className="assistant-stack">
                      {message.content || message.files?.length || message.error || message.route ? (
                        <div className="assistant-message">
                          {message.content ? <MarkdownView content={message.content} /> : null}
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
                          {message.error ? <div className="error-pill">{message.error}</div> : null}
                          {message.route ? <div className="route-meta">{message.route}</div> : null}
                        </div>
                      ) : null}

                      {message.workflow ? (
                        <div className="assistant-message">
                          <WorkflowSteps workflow={message.workflow} />
                        </div>
                      ) : null}

                      {message.artifacts?.map((artifact) => (
                        <ArtifactMessageCard key={artifact.url} artifact={artifact} />
                      ))}
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

            {pendingAttachments.length > 0 ? (
              <div className="attachment-preview-row">
                {pendingAttachments.map((file) => (
                  <div key={file.id} className="attachment-chip">
                    <span className="attachment-chip-name">{file.name}</span>
                    {file.fileKind ? <span className="attachment-chip-kind">{file.fileKind}</span> : null}
                    {file.status === "uploading" ? <span className="attachment-chip-status">上传中…</span> : null}
                    {file.status === "failed" ? <span className="attachment-chip-status failed">失败</span> : null}
                    <button
                      type="button"
                      className="attachment-chip-remove"
                      onClick={() =>
                        setPendingAttachments((prev) => prev.filter((x) => x.id !== file.id))
                      }
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="composer-input-row">
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
                rows={1}
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
            </div>
          </form>
          {error ? <div className="composer-error">{error}</div> : null}
        </div>
      </section>
      {dragging ? <div className="drop-overlay">释放以上传地震数据文件</div> : null}
    </main>
  );
}
