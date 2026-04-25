"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { WorkflowSteps } from "@/components/workflow-steps";
import {
  chatWithAgent,
  toBackendUrl,
  uploadFile,
  type ChatArtifact,
  type ChatResponse,
  type WorkflowResult,
} from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  route?: string;
  artifacts?: ChatArtifact[];
  error?: string | null;
  workflow?: WorkflowResult | null;
  files?: Array<{ name: string; fileType?: string }>;
};

const UPLOAD_ACCEPT =
  ".mseed,.miniseed,.sac,.sgy,.segy,.h5,.hdf5,.npy,.npz,.csv,.txt";

export default function HomePage() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragCounter = useRef(0);

  const canSend = input.trim().length > 0 && !chatLoading && !uploading;

  const handleUploadFiles = useCallback(async (inputFiles: File[] | FileList) => {
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
            role: "assistant",
            content: "文件上传失败。",
            error: message,
          },
        ]);
      }
    }

    setSessionId(currentSession);
    setUploading(false);
  }, [sessionId]);

  async function handleSend(messageText: string) {
    const text = messageText.trim();
    if (!text || chatLoading || uploading) {
      return;
    }

    setChatLoading(true);
    setError(null);
    setMessages((current) => [...current, { role: "user", content: text }]);

    try {
      const response: ChatResponse = await chatWithAgent({
        message: text,
        session_id: sessionId,
        lang: "zh",
      });

      setSessionId(response.session_id);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer || response.error || "No response returned.",
          route: response.route,
          artifacts: response.artifacts,
          error: response.error,
          workflow: response.workflow ?? null,
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed.";
      setError(message);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: message,
          error: message,
        },
      ]);
    } finally {
      setChatLoading(false);
      setInput("");
    }
  }

  function handleNewChat() {
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

  return (
    <main
      className={`shell chat-shell minimal-chat gpt-like ${dragging ? "dragging" : ""}`}
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
      <nav className="top-nav gpt-top-nav" aria-label="Primary">
        <div className="brand-mark">QuakeCore</div>
        <div className="nav-links gpt-nav-links">
          <Link href="/settings">Settings</Link>
          <Link href="/skills">Skills</Link>
          <button type="button" className="nav-action" onClick={handleNewChat}>
            New chat
          </button>
        </div>
      </nav>
      <section className="panel chat-panel gpt-chat-panel">
        <div className="chat-log gpt-chat-log" aria-live="polite">
          {messages.length === 0 ? (
            <div className="gpt-empty" />
          ) : (
            messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`bubble gpt-bubble ${message.role}`}>
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
                {message.workflow ? <WorkflowSteps workflow={message.workflow} /> : null}
                {message.error ? <div className="error-pill">{message.error}</div> : null}
                {message.route && message.role === "assistant" ? (
                  <div className="gpt-route">route: {message.route}</div>
                ) : null}
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
              </article>
            ))
          )}
        </div>

        <div className="gpt-composer-wrap">
          <form
            className="composer chat-composer gpt-composer"
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
              className="upload-btn gpt-upload-btn"
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
            <button type="submit" disabled={!canSend} className="send-btn gpt-send-btn">
              {chatLoading ? "..." : uploading ? "..." : "Send"}
            </button>
          </form>
          {error ? <div className="composer-error">{error}</div> : null}
        </div>
      </section>
      {dragging ? <div className="drop-overlay">释放以上传地震数据文件</div> : null}
    </main>
  );
}
