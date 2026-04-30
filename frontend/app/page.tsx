"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { MarkdownView } from "@/components/markdown-view";
import { WorkflowSteps } from "@/components/workflow-steps";
import { useLanguage, inferPendingKey } from "@/lib/i18n";
import { LanguageToggle } from "@/components/language-toggle";
import {
  chatWithAgentStream,
  getContinuousWorkflowProgress,
  getContinuousWorkflowResult,
  getLlmConfig,
  saveLlmConfig,
  startContinuousWorkflow,
  toBackendUrl,
  uploadFile,
  type ChatArtifact,
  type ContinuousJobProgressResponse,
  type LlmConfig,
  type StreamEvent,
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
  internalRuntime?: string;
  isOpencodeRunning?: boolean;
  liveSteps?: Array<{
    id: string;
    summary: string;
    detail?: string;
    status?: string;
    tool?: string;
    timestamp?: number;
  }>;
  progress?: {
    percent: number;
    step: string;
    status: string;
  };
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
  return inferPendingKey(text);
}

function isContinuousMonitoringRequest(text: string): boolean {
  const source = text.toLowerCase();
  return (
    source.includes("连续") ||
    source.includes("监测") ||
    source.includes("continuous monitoring") ||
    source.includes("continuous")
  );
}

function isOpencodeResponse(event: StreamEvent["response"] | undefined): boolean {
  if (!event) {
    return false;
  }
  return Boolean(event.internal_runtime === "quakecore" || event.route === "result_analysis");
}

const ARTIFACT_PATH_PATTERN =
  /(?:\/api\/artifacts\/[^\s`'"<>()，。；！？：:,]+|(?:\.?\/)?data\/[^\s`'"<>()，。；！？：:,]+)/gi;

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

function normalizeArtifactPath(raw: string): string {
  let value = String(raw || "").trim().replace(/\\/g, "/");
  value = value.replace(/^['"`]+|['"`]+$/g, "");
  value = value.replace(/[，。；！？：:,]+$/g, "");
  if (value.startsWith("/api/artifacts/")) {
    value = value.slice("/api/artifacts/".length);
  }
  if (value.startsWith("./")) {
    value = value.slice(2);
  }
  if (value.startsWith("/")) {
    value = value.slice(1);
  }
  if (value.startsWith("data/")) {
    value = value.slice(5);
  } else if (value.includes("/data/")) {
    value = value.split("/data/").pop() || value;
  }
  return value.replace(/^\/+/, "");
}

function inferArtifactTypeFromPath(path: string): "image" | "file" {
  const lower = path.toLowerCase();
  if (/\.(png|jpg|jpeg|gif|webp|svg)$/.test(lower)) {
    return "image";
  }
  return "file";
}

function extractArtifactsFromText(content: string): ChatArtifact[] {
  const text = String(content || "");
  const matches = text.match(ARTIFACT_PATH_PATTERN) || [];
  const artifacts: ChatArtifact[] = [];
  const seen = new Set<string>();

  for (const token of matches) {
    const path = normalizeArtifactPath(token);
    if (!path || seen.has(path)) {
      continue;
    }
    seen.add(path);
    artifacts.push({
      type: inferArtifactTypeFromPath(path),
      name: basename(path),
      path,
      url: `/api/artifacts/${path}`,
    });
  }
  return artifacts;
}

function mergeArtifacts(primary: ChatArtifact[] | undefined, content: string): ChatArtifact[] {
  const merged: ChatArtifact[] = [];
  const seen = new Set<string>();
  const fromPayload = Array.isArray(primary) ? primary : [];
  const fromText = extractArtifactsFromText(content);

  for (const item of [...fromPayload, ...fromText]) {
    const path = normalizeArtifactPath(item.path || item.url || "");
    if (!path) {
      continue;
    }
    const url = `/api/artifacts/${path}`;
    if (seen.has(url)) {
      continue;
    }
    seen.add(url);
    merged.push({
      type: item.type || inferArtifactTypeFromPath(path),
      name: item.name || basename(path),
      path,
      url,
    });
  }
  return merged;
}

function ArtifactMessageCard({ artifact }: { artifact: ChatArtifact }) {
  const { t } = useLanguage();
  const url = toBackendUrl(artifact.url);
  const displayName = getArtifactDisplayName(artifact);
  const rawPath = artifact.path || artifact.name || "";
  const [copied, setCopied] = useState(false);

  async function handleDownload() {
    const response = await fetch(url);
    if (!response.ok) {
      alert(t("download_failed"));
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
      alert(t("copy_failed"));
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
          title={t("view_large_image")}
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
              {t("download")}
            </button>

            {isImage ? (
              <>
                <button
                  type="button"
                  onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
                >
                  {t("view")}
                </button>

                <button type="button" onClick={() => void handleCopy()}>
                  {t("copy")}
                </button>
              </>
            ) : null}
          </div>

          {copied ? <div className="copy-toast">{t("copy_success")}</div> : null}
      </div>
    </div>
  );
}

export default function HomePage() {
  const { t, lang } = useLanguage();
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
    const title = messages.find((item) => item.role === "user")?.content.slice(0, 24) || t("unnamed_session");
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
                  ? { ...a, status: "failed", error: err instanceof Error ? err.message : t("upload_failed") }
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

  function applyContinuousProgress(
    pendingId: string,
    progress: ContinuousJobProgressResponse,
  ) {
    setMessages((current) =>
      current.map((item) =>
        item.id === pendingId
          ? {
              ...item,
              pending: progress.status === "running",
              route: "continuous_monitoring",
              content: t("monitoring_in_progress"),
              progress: {
                percent: Number(progress.percent ?? 0),
                step: String(progress.step || t("processing")),
                status: String(progress.status || "running"),
              },
            }
          : item,
      ),
    );
  }

  async function runContinuousWorkflow(
    text: string,
    pendingId: string,
    sessionHint: string | null,
  ) {
    const started = await startContinuousWorkflow({
      message: text,
      session_id: sessionHint,
      lang: lang,
    });
    setSessionId(started.session_id);
    const jobId = started.job_id;

    let progress = await getContinuousWorkflowProgress(jobId);
    applyContinuousProgress(pendingId, progress);

    while (progress.status === "running") {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      progress = await getContinuousWorkflowProgress(jobId);
      applyContinuousProgress(pendingId, progress);
    }

    const result = await getContinuousWorkflowResult(jobId);
    if (result.status === "running") {
      return;
    }

    const payload = result.result || {};
    const answer = String(payload.message || result.error || t("monitoring_complete"));
    const artifacts = mergeArtifacts(
      Array.isArray(payload.artifacts) ? payload.artifacts : [],
      answer,
    );
    const errorText = String(payload.error || result.error || "");

    setMessages((current) =>
      current.map((item) =>
        item.id === pendingId
          ? {
              ...item,
              pending: false,
              route: "continuous_monitoring",
              content: answer,
              artifacts,
              error: errorText || null,
              progress: undefined,
            }
          : item,
      ),
    );
  }

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

    if (isContinuousMonitoringRequest(text)) {
      const pendingId = newId();
      setMessages((current) => [
        ...current,
        {
          id: pendingId,
          role: "assistant",
          content: t("monitoring_in_progress"),
          route: "continuous_monitoring",
          pending: true,
          progress: { percent: 2, step: t("task_preparing"), status: "running" },
        },
      ]);
      try {
        await runContinuousWorkflow(text, pendingId, sessionId);
      } catch (err) {
        const message = err instanceof Error ? err.message : t("monitoring_failed");
        setError(message);
        setMessages((current) =>
          current.map((item) =>
            item.id === pendingId
              ? {
                  ...item,
                  pending: false,
                  content: t("monitoring_failed"),
                  error: message,
                }
              : item,
          ),
        );
      } finally {
        setChatLoading(false);
      }
      return;
    }

    const msgId = newId();
    setMessages((current) => [
      ...current,
      {
        id: msgId,
        role: "assistant",
        content: t(inferPendingText(text)),
        pending: true,
        isOpencodeRunning: true,
        liveSteps: [],
      } as Message,
    ]);

    try {
      const stream = chatWithAgentStream({
        message: text,
        session_id: sessionId,
        lang: lang,
        attachments: userAttachments
          .filter((a): a is ChatAttachment & { path: string } => Boolean(a.path))
          .map((a) => ({
            name: a.name,
            path: a.path,
            file_type: a.fileKind,
          })),
      });

      let receivedFinal = false;

      for await (const event of stream) {
        if (event.type === "progress" && event.event) {
          const ev = event.event;
          const liveStep = {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            summary: ev.summary || ev.tool || t("analyzing_badge"),
            detail: ev.detail || "",
            status: ev.status || "running",
            tool: ev.tool,
            timestamp: ev.timestamp || Date.now(),
          };

          setMessages((current) =>
            current.map((item) =>
              item.id === msgId
                ? {
                    ...item,
                    internalRuntime: "quakecore",
                    isOpencodeRunning: true,
                    liveSteps: [liveStep, ...(item.liveSteps || [])].slice(0, 4),
                  }
                : item,
            ),
          );
        }

        if (event.type === "final" && event.response) {
          receivedFinal = true;
          const r = event.response;
          setSessionId(r.session_id);
          const responseText = r.answer || r.error || t("no_response");
          const resolvedArtifacts = mergeArtifacts(r.artifacts, responseText);

          setMessages((current) =>
            current.map((item) =>
              item.id === msgId
                ? {
                    ...item,
                    pending: false,
                    content: responseText,
                    route: r.route,
                    artifacts: resolvedArtifacts,
                    error: r.error,
                    internalRuntime: isOpencodeResponse(r) ? "quakecore" : undefined,
                    isOpencodeRunning: false,
                    workflow: isOpencodeResponse(r) ? null : (r.workflow ?? null),
                  }
                : item,
            ),
          );
          if (isOpencodeResponse(r)) {
            window.setTimeout(() => {
              setMessages((current) =>
                current.map((item) =>
                  item.id === msgId ? { ...item, liveSteps: [] } : item,
                ),
              );
            }, 1200);
          }
          break;
        }
      }

      if (!receivedFinal) {
        setMessages((current) =>
          current.map((item) =>
            item.id === msgId
              ? {
                  ...item,
                  pending: false,
                  content: t("stream_ended"),
                  error: t("stream_ended"),
                  isOpencodeRunning: false,
                }
              : item,
          ),
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : t("request_failed");
      setError(message);
      setMessages((current) =>
        current.map((item) =>
            item.id === msgId
            ? { ...item, pending: false, content: t("request_failed"), error: message, isOpencodeRunning: false }
            : item,
        ),
      );
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
      setError(t("switch_config_failed"));
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
            {t("quakecore_dropdown")}
          </button>
          {modelOpen ? (
            <div className="model-menu">
              <p className="model-menu-title">{t("model_backend")}</p>
              <button type="button" className="model-option" onClick={() => void switchProvider("deepseek")}>
                {llmConfig?.provider === "deepseek" ? "●" : "○"} {t("deepseek_api")}
                <span>model: {llmConfig?.provider === "deepseek" ? llmConfig.model_name : "-"}</span>
              </button>
              <button type="button" className="model-option" onClick={() => void switchProvider("ollama")}>
                {llmConfig?.provider === "ollama" ? "●" : "○"} {t("ollama_local")}
                <span>model: {llmConfig?.provider === "ollama" ? llmConfig.model_name : "-"}</span>
              </button>
              <Link href="/settings" className="model-settings-link">
                {t("open_settings")}
              </Link>
            </div>
          ) : null}
        </div>

        <button type="button" className="new-chat-btn" onClick={handleNewChat}>
          {t("new_chat")}
        </button>

        <div className="session-list">
          <p>{t("recent_sessions")}</p>
          <button type="button" className="session-item active">
            {t("current_session")}
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
          <Link href="/skills">{t("nav_skills")}</Link>
          <LanguageToggle />
          <small>
            {t("current_backend")}
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
                <h2>{t("empty_title")}</h2>
                <p>{t("empty_subtitle")}</p>
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
                    message.route === "continuous_monitoring" && message.progress ? (
                      <div className="assistant-message progress-card">
                        <div>{t("monitoring_in_progress")}</div>
                        <div className="progress-bar">
                          <div style={{ width: `${Math.max(0, Math.min(100, message.progress.percent))}%` }} />
                        </div>
                        <div>{message.progress.step}</div>
                      </div>
                    ) : (
                      <div className="assistant-message assistant-message-live">
                        <div className="qc-live-header">
                          <span className="pending-text">
                            {message.content}
                            <span className="dot-wave">
                              <i />
                              <i />
                              <i />
                            </span>
                          </span>
                          {message.isOpencodeRunning ? <span className="qc-live-badge">{t("analyzing_badge")}</span> : null}
                        </div>
                        {message.liveSteps?.length ? (
                          <div className="qc-live-stack" aria-live="polite">
                            {message.liveSteps.map((step, index) => (
                              <div key={step.id} className="qc-live-card" data-index={index} aria-hidden={index > 0}>
                                {index === 0 ? (
                                  <>
                                    <div className="qc-live-card-title">{step.summary}</div>
                                    {step.detail ? <div className="qc-live-card-detail">{step.detail}</div> : null}
                                    <div className="qc-live-card-footer">{step.status === "running" ? t("processing") : t("completed")}</div>
                                  </>
                                ) : null}
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    )
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

                      {message.workflow && message.internalRuntime !== "quakecore" ? (
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
                    {file.status === "uploading" ? <span className="attachment-chip-status">{t("uploading_status")}</span> : null}
                    {file.status === "failed" ? <span className="attachment-chip-status failed">{t("failed_status")}</span> : null}
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
                aria-label={t("upload_attachments")}
                title={t("upload_attachments")}
                disabled={uploading}
              >
                +
              </button>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={t("message_placeholder")}
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
      {dragging ? <div className="drop-overlay">{t("drop_overlay")}</div> : null}
    </main>
  );
}
