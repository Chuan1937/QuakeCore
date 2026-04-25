"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { chatWithAgent, type ChatArtifact, type ChatResponse } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  route?: string;
  artifacts?: ChatArtifact[];
  error?: string | null;
};

const EXAMPLES = [
  "Analyze this file structure.",
  "Perform phase picking on the waveform.",
  "Help me locate the earthquake from the picked phases.",
];

export default function HomePage() {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSend = input.trim().length > 0 && !loading;

  const conversationCount = useMemo(
    () => messages.filter((message) => message.role === "user").length,
    [messages],
  );

  async function handleSend(messageText: string) {
    const text = messageText.trim();
    if (!text || loading) {
      return;
    }

    setLoading(true);
    setError(null);
    setMessages((current) => [...current, { role: "user", content: text }]);

    try {
      const response: ChatResponse = await chatWithAgent({
        message: text,
        session_id: sessionId,
        lang: "en",
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
      setLoading(false);
      setInput("");
    }
  }

  return (
    <main className="shell">
      <nav className="top-nav" aria-label="Primary">
        <div className="brand-mark">QuakeCore</div>
        <div className="nav-links">
          <Link href="/">Chat</Link>
          <Link href="/settings">Settings</Link>
          <Link href="/skills">Skills</Link>
        </div>
      </nav>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">QuakeCore</p>
          <h1>Seismic analysis, routed through chat.</h1>
          <p className="lede">
            Ask about file structure, phase picking, or earthquake location.
            The backend returns a route and artifacts directly in the response.
          </p>
        </div>
        <div className="status-card">
          <span>Session</span>
          <strong>{sessionId ?? "new"}</strong>
          <span>{conversationCount} user message(s)</span>
        </div>
      </section>

      <section className="panel">
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => void handleSend(example)}>
              {example}
            </button>
          ))}
        </div>

        <div className="chat-log" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a conversation</h2>
              <p>Type a question below or use one of the example prompts.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`bubble ${message.role}`}>
                <header>
                  <strong>{message.role}</strong>
                  {message.route ? <span>route: {message.route}</span> : null}
                </header>
                <p>{message.content}</p>
                {message.error ? <div className="error-pill">{message.error}</div> : null}
                {message.artifacts?.length ? (
                  <div className="artifacts">
                    {message.artifacts.map((artifact) => (
                      <a key={artifact.url} href={artifact.url} target="_blank" rel="noreferrer">
                        {artifact.type}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            ))
          )}
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void handleSend(input);
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask QuakeCore about your seismic data..."
            rows={3}
          />
          <div className="composer-actions">
            <span>{loading ? "Sending..." : error ?? "Ready"}</span>
            <button type="submit" disabled={!canSend}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
