import type { WorkflowResult } from "@/lib/api";

type WorkflowStepsProps = {
  workflow: WorkflowResult;
};

const STATUS_LABELS: Record<string, string> = {
  success: "Success",
  partial_success: "Partial Success",
  failed: "Failed",
  ok: "OK",
  warning: "Warning",
  error: "Error",
  skipped: "Skipped",
};

export function WorkflowSteps({ workflow }: WorkflowStepsProps) {
  const summary = workflow.summary || workflow.message || "";
  const steps = workflow.steps || [];

  return (
    <section className="workflow-card" aria-label="Workflow result">
      <header className="workflow-header">
        <strong>Workflow</strong>
        <span className={`workflow-status workflow-status-${workflow.status}`}>
          {STATUS_LABELS[workflow.status] || workflow.status}
        </span>
      </header>

      {summary ? <p className="workflow-summary">{summary}</p> : null}

      {steps.length ? (
        <ul className="workflow-steps">
          {steps.map((step, index) => (
            <li key={`${step.name}-${index}`} className={`workflow-step workflow-step-${step.status}`}>
              <div className="workflow-step-top">
                <code>{step.name}</code>
                <span>{STATUS_LABELS[step.status] || step.status}</span>
              </div>
              <div className="workflow-step-meta">
                {typeof step.duration_ms === "number" ? <span>{step.duration_ms} ms</span> : null}
                {step.required === false ? <span>optional</span> : <span>required</span>}
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
