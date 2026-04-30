"use client";

import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";

export type Lang = "zh" | "en";

const STORAGE_KEY = "quakecore-lang";

const dict: Record<string, Record<Lang, string>> = {
  nav_chat: { zh: "聊天", en: "Chat" },
  nav_settings: { zh: "设置", en: "Settings" },
  nav_skills: { zh: "技能", en: "Skills" },
  brand_name: { zh: "QuakeCore", en: "QuakeCore" },

  quakecore_dropdown: { zh: "QuakeCore ▾", en: "QuakeCore ▾" },
  model_backend: { zh: "模型 / 后端", en: "Model / Backend" },
  deepseek_api: { zh: "DeepSeek API", en: "DeepSeek API" },
  ollama_local: { zh: "Ollama 本地", en: "Ollama Local" },
  open_settings: { zh: "打开设置", en: "Open Settings" },
  new_chat: { zh: "+ 新会话", en: "+ New Chat" },
  recent_sessions: { zh: "最近会话", en: "Recent Sessions" },
  current_session: { zh: "当前会话", en: "Current Session" },
  current_backend: { zh: "当前后端：", en: "Backend: " },

  empty_title: { zh: "今天要分析什么地震数据？", en: "What seismic data to analyze today?" },
  empty_subtitle: { zh: "上传 MiniSEED、SAC、SEGY、HDF5，或直接提问。", en: "Upload MiniSEED, SAC, SEGY, HDF5, or ask directly." },
  message_placeholder: { zh: "向 QuakeCore 提问", en: "Message QuakeCore" },
  drop_overlay: { zh: "释放以上传地震数据文件", en: "Drop to upload seismic data files" },
  upload_attachments: { zh: "上传附件", en: "Upload attachments" },

  uploading_status: { zh: "上传中…", en: "Uploading…" },
  failed_status: { zh: "失败", en: "Failed" },

  pending_analysis: { zh: "正在分析波形并进行初至/震相拾取…", en: "Analyzing waveform and picking arrivals/phases…" },
  pending_location: { zh: "正在执行地震定位工作流…", en: "Running earthquake location workflow…" },
  pending_reading: { zh: "正在读取文件结构…", en: "Reading file structure…" },
  pending_monitoring: { zh: "正在准备连续地震监测任务…", en: "Preparing continuous monitoring task…" },
  pending_thinking: { zh: "正在思考…", en: "Thinking…" },
  monitoring_in_progress: { zh: "连续地震监测进行中", en: "Continuous monitoring in progress" },
  monitoring_complete: { zh: "连续监测完成。", en: "Monitoring complete." },
  monitoring_failed: { zh: "连续监测任务失败。", en: "Monitoring task failed." },
  task_preparing: { zh: "任务准备中", en: "Preparing task" },

  processing: { zh: "正在处理", en: "Processing" },
  completed: { zh: "已完成", en: "Completed" },
  analyzing_badge: { zh: "QuakeCore 正在分析...", en: "QuakeCore analyzing..." },

  download_failed: { zh: "文件不存在或下载失败", en: "File not found or download failed" },
  copy_failed: { zh: "复制失败", en: "Copy failed" },
  view_large_image: { zh: "查看大图", en: "View full image" },
  download: { zh: "下载", en: "Download" },
  view: { zh: "查看", en: "View" },
  copy: { zh: "复制", en: "Copy" },
  copy_success: { zh: "复制成功", en: "Copied" },

  unnamed_session: { zh: "未命名会话", en: "Unnamed session" },
  upload_failed: { zh: "上传失败", en: "Upload failed" },
  request_failed: { zh: "请求失败。", en: "Request failed." },
  switch_config_failed: { zh: "切换模型配置失败。", en: "Failed to switch model config." },
  stream_ended: { zh: "流已结束，未收到响应。", en: "Stream ended without response." },
  no_response: { zh: "未收到回复。", en: "No response returned." },

  model_settings: { zh: "模型设置", en: "Model Settings" },
  settings_subtitle: { zh: "选择 DeepSeek API 或本地 Ollama。", en: "Choose DeepSeek API or local Ollama." },
  backend_label: { zh: "后端", en: "Backend" },
  model_label: { zh: "模型", en: "Model" },
  api_key_label: { zh: "API Key", en: "API Key" },
  base_url_label: { zh: "Base URL", en: "Base URL" },
  ollama_url: { zh: "Ollama 地址", en: "Ollama URL" },
  local_model: { zh: "本地模型", en: "Local Model" },
  api_key_placeholder: { zh: "可留空，默认读取 DEEPSEEK_API_KEY", en: "Leave empty to use DEEPSEEK_API_KEY" },
  save_and_return: { zh: "保存并返回", en: "Save & Return" },
  saving: { zh: "保存中...", en: "Saving..." },
  detecting: { zh: "检测中...", en: "Detecting..." },
  detect_local_models: { zh: "检测本地模型", en: "Detect Local Models" },
  detecting_ollama: { zh: "正在检测本地 Ollama 模型...", en: "Detecting local Ollama models..." },
  detect_failed: { zh: "检测 Ollama 失败，请确认后端服务正常", en: "Failed to detect Ollama. Check service status." },
  no_models: { zh: "未检测到 Ollama 模型", en: "No Ollama models found" },
  save_failed: { zh: "保存失败。", en: "Save failed." },
  load_settings_failed: { zh: "加载设置失败。", en: "Failed to load settings." },
  back_to_chat: { zh: "返回聊天", en: "Back to chat" },

  loading: { zh: "加载中...", en: "Loading..." },
  loading_skill: { zh: "正在加载技能...", en: "Loading skill..." },
  no_skills: { zh: "未找到技能", en: "No skills found" },
  load_skills_failed: { zh: "加载技能失败。", en: "Failed to load skills." },
  load_skill_failed: { zh: "加载技能失败。", en: "Failed to load skill." },
  skills_title: { zh: "技能", en: "Skills" },
  skills_subtitle: { zh: "Markdown 技能库", en: "Markdown skill library" },
  skills_description: { zh: "浏览后端提供的 Markdown 技能文件，查看助手使用的内容。", en: "Browse the markdown skill files served by the backend and inspect the content used by the assistant." },
  status_label: { zh: "状态", en: "Status" },
  skills_available: { zh: "个技能可用", en: "skill(s) available" },
  available_skills: { zh: "可用技能", en: "Available skills" },
  markdown_label: { zh: "Markdown", en: "Markdown" },
  no_skill_selected: { zh: "未选择技能", en: "No skill selected" },
  select_skill_hint: { zh: "请从左侧选择一个技能以查看其 Markdown 内容。", en: "Select a skill on the left to view its markdown content." },

  status_success: { zh: "成功", en: "Success" },
  status_partial_success: { zh: "部分成功", en: "Partial Success" },
  status_failed: { zh: "失败", en: "Failed" },
  status_ok: { zh: "正常", en: "OK" },
  status_warning: { zh: "警告", en: "Warning" },
  status_error: { zh: "错误", en: "Error" },
  status_skipped: { zh: "已跳过", en: "Skipped" },
  workflow: { zh: "工作流", en: "Workflow" },
  optional: { zh: "可选", en: "optional" },
  required: { zh: "必需", en: "required" },

  no_content: { zh: "无内容。", en: "No content." },
};

type LanguageContextType = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string) => string;
};

const LanguageContext = createContext<LanguageContextType | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("zh");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "en" || stored === "zh") {
      setLangState(stored);
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = useCallback((newLang: Lang) => {
    setLangState(newLang);
    localStorage.setItem(STORAGE_KEY, newLang);
  }, []);

  const t = useCallback((key: string): string => {
    return dict[key]?.[lang] ?? key;
  }, [lang]);

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useLanguage must be used within LanguageProvider");
  return ctx;
}

export function inferPendingKey(text: string): string {
  const source = text.toLowerCase();
  if (
    source.includes("初至") ||
    source.includes("拾取") ||
    source.includes("震相") ||
    source.includes("p波") ||
    source.includes("s波") ||
    source.includes("arrival") ||
    source.includes("phase") ||
    source.includes("picking") ||
    source.includes("pick")
  ) {
    return "pending_analysis";
  }
  if (source.includes("定位") || source.includes("震中") || source.includes("震源") || source.includes("location") || source.includes("epicenter") || source.includes("hypocenter")) {
    return "pending_location";
  }
  if (source.includes("结构") || source.includes("采样率") || source.includes("header") || source.includes("structure") || source.includes("sample rate") || source.includes("sampling")) {
    return "pending_reading";
  }
  if (source.includes("连续") || source.includes("监测") || source.includes("continuous") || source.includes("monitoring")) {
    return "pending_monitoring";
  }
  return "pending_thinking";
}
