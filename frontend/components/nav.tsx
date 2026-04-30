"use client";

import Link from "next/link";
import { useLanguage } from "@/lib/i18n";
import { LanguageToggle } from "@/components/language-toggle";

export function Nav() {
  const { t } = useLanguage();
  return (
    <nav className="top-nav" aria-label="Primary">
      <div className="brand-mark">{t("brand_name")}</div>
      <div className="nav-links">
        <Link href="/">{t("nav_chat")}</Link>
        <Link href="/settings">{t("nav_settings")}</Link>
        <Link href="/skills">{t("nav_skills")}</Link>
        <LanguageToggle />
      </div>
    </nav>
  );
}
