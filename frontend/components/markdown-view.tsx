"use client";

import type { ReactNode } from "react";
import { toBackendUrl } from "@/lib/api";
import { useLanguage } from "@/lib/i18n";

type MarkdownViewProps = {
  content: string;
};

function parseInline(text: string, keyPrefix: string): ReactNode[] {
  const tokens = text
    .split(/(`[^`]+`|\*\*[^*]+\*\*|!\[[^\]]*\]\([^\)]+\)|\[[^\]]+\]\([^\)]+\))/g)
    .filter(Boolean);

  return tokens.map((token, index) => {
    if (token.startsWith("**") && token.endsWith("**")) {
      return <strong key={`${keyPrefix}-strong-${index}`}>{token.slice(2, -2)}</strong>;
    }
    if (token.startsWith("`") && token.endsWith("`")) {
      return <code key={`${keyPrefix}-code-${index}`}>{token.slice(1, -1)}</code>;
    }

    const imageMatch = token.match(/^!\[([^\]]*)\]\(([^\)]+)\)$/);
    if (imageMatch) {
      const alt = imageMatch[1] || "image";
      const src = imageMatch[2] || "";
      return (
        <a key={`${keyPrefix}-img-link-${index}`} href={toBackendUrl(src)} target="_blank" rel="noreferrer">
          <img className="markdown-image" src={toBackendUrl(src)} alt={alt} />
        </a>
      );
    }

    const linkMatch = token.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
    if (linkMatch) {
      return (
        <a key={`${keyPrefix}-link-${index}`} href={toBackendUrl(linkMatch[2])} target="_blank" rel="noreferrer">
          {linkMatch[1]}
        </a>
      );
    }

    return <span key={`${keyPrefix}-text-${index}`}>{token}</span>;
  });
}

function parseTable(lines: string[], start: number): { end: number; node: ReactNode } | null {
  if (!lines[start]?.includes("|") || !lines[start + 1]?.includes("|")) {
    return null;
  }
  const separator = lines[start + 1].trim();
  if (!/^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(separator)) {
    return null;
  }

  const header = lines[start]
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());

  const rows: string[][] = [];
  let cursor = start + 2;
  while (cursor < lines.length) {
    const line = lines[cursor].trim();
    if (!line || !line.includes("|")) {
      break;
    }
    rows.push(
      line
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim()),
    );
    cursor += 1;
  }

  return {
    end: cursor - 1,
    node: (
      <div className="markdown-table-wrap">
        <table>
          <thead>
            <tr>
              {header.map((cell, idx) => (
                <th key={`th-${idx}`}>{parseInline(cell, `th-${idx}`)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr key={`tr-${rowIdx}`}>
                {row.map((cell, cellIdx) => (
                  <td key={`td-${rowIdx}-${cellIdx}`}>{parseInline(cell, `td-${rowIdx}-${cellIdx}`)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ),
  };
}

export function MarkdownView({ content }: MarkdownViewProps) {
  const { t } = useLanguage();
  const lines = content.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    const table = parseTable(lines, i);
    if (table) {
      blocks.push(<div key={`table-${i}`}>{table.node}</div>);
      i = table.end + 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      blocks.push(
        <pre key={`code-${i}`}>
          <code className={lang ? `language-${lang}` : undefined}>{codeLines.join("\n")}</code>
        </pre>,
      );
      i += 1;
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(4, heading[1].length);
      const text = heading[2];
      if (level === 1) {
        blocks.push(<h1 key={`h1-${i}`}>{parseInline(text, `h1-${i}`)}</h1>);
      } else if (level === 2) {
        blocks.push(<h2 key={`h2-${i}`}>{parseInline(text, `h2-${i}`)}</h2>);
      } else if (level === 3) {
        blocks.push(<h3 key={`h3-${i}`}>{parseInline(text, `h3-${i}`)}</h3>);
      } else {
        blocks.push(<h4 key={`h4-${i}`}>{parseInline(text, `h4-${i}`)}</h4>);
      }
      i += 1;
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`ul-${i}`}>
          {items.map((item, idx) => (
            <li key={`li-${i}-${idx}`}>{parseInline(item, `li-${i}-${idx}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (i < lines.length && lines[i].trim() && !lines[i].trim().startsWith("#") && !/^[-*]\s+/.test(lines[i].trim())) {
      if (lines[i].trim().startsWith("```")) {
        break;
      }
      if (parseTable(lines, i)) {
        break;
      }
      paragraph.push(lines[i].trim());
      i += 1;
    }
    blocks.push(<p key={`p-${i}`}>{parseInline(paragraph.join(" "), `p-${i}`)}</p>);
  }

  if (!blocks.length) {
    return <p className="markdown-empty">{t("no_content")}</p>;
  }

  return <div className="markdown-body">{blocks}</div>;
}
