"use client";

import type { WorkflowResult } from "@/lib/api";
import { useLanguage } from "@/lib/i18n";

type WorkflowStepsProps = {
  workflow: WorkflowResult;
};

const STATUS_LABEL_KEYS: Record<string, string> = {
  success: "status_success",
  partial_success: "status_partial_success",
  failed: "status_failed",
  ok: "status_ok",
  warning: "status_warning",
  error: "status_error",
  skipped: "status_skipped",
};

export function WorkflowSteps({ workflow }: WorkflowStepsProps) {
  const { t } = useLanguage();
  const summary = workflow.summary || workflow.message || "";
  const steps = workflow.steps || [];

  return (
    <section className="workflow-card" aria-label="Workflow result">
      <header className="workflow-header">
        <strong>{t("workflow")}</strong>
        <span className={`workflow-status workflow-status-${workflow.status}`}>
          {t(STATUS_LABEL_KEYS[workflow.status] || workflow.status)}
        </span>
      </header>

      {summary ? <p className="workflow-summary">{summary}</p> : null}

      {steps.length ? (
        <ul className="workflow-steps">
          {steps.map((step, index) => (
            <li key={`${step.name}-${index}`} className={`workflow-step workflow-step-${step.status}`}>
              <div className="workflow-step-top">
                <code>{step.name}</code>
                <span>{t(STATUS_LABEL_KEYS[step.status] || step.status)}</span>
              </div>
              <div className="workflow-step-meta">
                {typeof step.duration_ms === "number" ? <span>{step.duration_ms} ms</span> : null}
                {step.required === false ? <span>{t("optional")}</span> : <span>{t("required")}</span>}
              </div>
              {step.message ? <p>{step.message}</p> : null}
              {step.error ? <p className="workflow-step-error">{step.error}</p> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
