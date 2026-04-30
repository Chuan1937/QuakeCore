"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/nav";
import { MarkdownView } from "@/components/markdown-view";
import { getSkill, listSkills, type SkillDetail, type SkillSummary } from "@/lib/api";
import { useLanguage } from "@/lib/i18n";

export default function SkillsPage() {
  const { t } = useLanguage();
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [activeSkill, setActiveSkill] = useState<string>("");
  const [detail, setDetail] = useState<SkillDetail | null>(null);
  const [message, setMessage] = useState<string>(t("loading"));

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
          setMessage(t("no_skills"));
        }
      } catch (error) {
        if (!mounted) {
          return;
        }
        setMessage(error instanceof Error ? error.message : t("load_skills_failed"));
      }
    }

    void loadSkills();

    return () => {
      mounted = false;
    };
  }, [t]);

  async function selectSkill(name: string) {
    setActiveSkill(name);
    setMessage(t("loading_skill"));
    try {
      const next = await getSkill(name);
      setDetail(next);
      setMessage("Loaded");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : t("load_skill_failed"));
    }
  }

  return (
    <main className="shell">
      <Nav />
      <section className="hero compact">
        <div className="hero-copy">
          <p className="eyebrow">{t("skills_title")}</p>
          <h1>{t("skills_subtitle")}</h1>
          <p className="lede">
            {t("skills_description")}
          </p>
        </div>
        <div className="status-card">
          <span>{t("status_label")}</span>
          <strong>{message}</strong>
          <span>{skills.length} {t("skills_available")}</span>
        </div>
      </section>

      <section className="skills-layout">
        <aside className="panel skills-list-panel">
          <h2>{t("available_skills")}</h2>
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
                  <p className="eyebrow">{t("markdown_label")}</p>
                  <h2>{detail.name}</h2>
                </div>
                <code>{detail.path}</code>
              </div>
              <MarkdownView content={detail.content} />
            </>
          ) : (
            <div className="empty-state">
              <h2>{t("no_skill_selected")}</h2>
              <p>{t("select_skill_hint")}</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
