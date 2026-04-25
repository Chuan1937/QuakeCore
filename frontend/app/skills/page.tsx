"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/nav";
import { MarkdownView } from "@/components/markdown-view";
import { getSkill, listSkills, type SkillDetail, type SkillSummary } from "@/lib/api";

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [activeSkill, setActiveSkill] = useState<string>("");
  const [detail, setDetail] = useState<SkillDetail | null>(null);
  const [message, setMessage] = useState<string>("Loading...");

  useEffect(() => {
    let mounted = true;

    async function loadSkills() {
      try {
        const result = await listSkills();
        if (!mounted) {
          return;
        }
        setSkills(result.skills);
        if (result.skills.length > 0) {
          const first = result.skills[0].name;
          setActiveSkill(first);
          const firstDetail = await getSkill(first);
          if (mounted) {
            setDetail(firstDetail);
            setMessage("Loaded");
          }
        } else {
          setMessage("No skills found");
        }
      } catch (error) {
        if (!mounted) {
          return;
        }
        setMessage(error instanceof Error ? error.message : "Failed to load skills.");
      }
    }

    void loadSkills();

    return () => {
      mounted = false;
    };
  }, []);

  async function selectSkill(name: string) {
    setActiveSkill(name);
    setMessage("Loading skill...");
    try {
      const next = await getSkill(name);
      setDetail(next);
      setMessage("Loaded");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Failed to load skill.");
    }
  }

  return (
    <main className="shell">
      <Nav />
      <section className="hero compact">
        <div className="hero-copy">
          <p className="eyebrow">Skills</p>
          <h1>Markdown skill library</h1>
          <p className="lede">
            Browse the markdown skill files served by the backend and inspect
            the content used by the assistant.
          </p>
        </div>
        <div className="status-card">
          <span>Status</span>
          <strong>{message}</strong>
          <span>{skills.length} skill(s) available</span>
        </div>
      </section>

      <section className="skills-layout">
        <aside className="panel skills-list-panel">
          <h2>Available skills</h2>
          <div className="skills-list">
            {skills.map((skill) => (
              <button
                key={skill.name}
                type="button"
                className={skill.name === activeSkill ? "active" : ""}
                onClick={() => void selectSkill(skill.name)}
              >
                <span>{skill.name}</span>
                <small>{skill.path}</small>
              </button>
            ))}
          </div>
        </aside>

        <section className="panel skill-detail-panel">
          {detail ? (
            <>
              <div className="skill-detail-header">
                <div>
                  <p className="eyebrow">Markdown</p>
                  <h2>{detail.name}</h2>
                </div>
                <code>{detail.path}</code>
              </div>
              <MarkdownView content={detail.content} />
            </>
          ) : (
            <div className="empty-state">
              <h2>No skill selected</h2>
              <p>Select a skill on the left to view its markdown content.</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

