import { createElement } from "react";
import type { ReactElement } from "react";

type MarkdownViewProps = {
  content: string;
};

function renderInline(text: string, keyPrefix: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);

  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyPrefix}-strong-${index}`}>{part.slice(2, -2)}</strong>;
    }

    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${keyPrefix}-code-${index}`}>{part.slice(1, -1)}</code>;
    }

    return <span key={`${keyPrefix}-text-${index}`}>{part}</span>;
  });
}

export function MarkdownView({ content }: MarkdownViewProps) {
  const lines = content.split(/\r?\n/);
  const blocks: ReactElement[] = [];
  let paragraph: string[] = [];
  let listItems: string[] = [];

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    const text = paragraph.join(" ").trim();
    if (text) {
      blocks.push(
        <p key={`p-${blocks.length}`}>{renderInline(text, `p-${blocks.length}`)}</p>,
      );
    }
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) {
      return;
    }
    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {listItems.map((item, index) => (
          <li key={`li-${blocks.length}-${index}`}>{renderInline(item, `li-${blocks.length}-${index}`)}</li>
        ))}
      </ul>,
    );
    listItems = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^#{1,3}\s+/.test(trimmed)) {
      flushParagraph();
      flushList();
      const level = trimmed.match(/^#{1,3}/)?.[0].length ?? 1;
      const text = trimmed.replace(/^#{1,3}\s+/, "");
      const headingKey = `h-${blocks.length}`;
      blocks.push(
        createElement(
          `h${level}`,
          { key: headingKey },
          renderInline(text, headingKey),
        ),
      );
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      flushParagraph();
      listItems.push(trimmed.replace(/^[-*]\s+/, ""));
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  if (!blocks.length) {
    return <p className="markdown-empty">No content.</p>;
  }

  return <div className="markdown-body">{blocks}</div>;
}
