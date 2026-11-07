import {
  Activity,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle2,
  ClipboardList,
  Clock,
  FileText,
  FolderArchive,
  MessageSquareText,
  ShieldCheck,
  Stethoscope,
  TriangleAlert,
} from "lucide-react";

/* ── Static mock date/time ─────────────────── */
function formatTime(d) {
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
function formatDate(d) {
  return d.toLocaleDateString("en-GB", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

/* ── Module card definitions ────────────────── */
const MODULES = [
  {
    key: "brain",
    hdrClass: "hdr-teal",
    cardClass: "teal",
    eyebrow: "Imaging",
    title: "Brain Imaging Analysis",
    desc: "Submit a structural MRI scan for an automated imaging assessment. Receive a decision-support profile with a confidence score and a visual explanation of key regions.",
    icon: Brain,
    actionLabel: "Open Analysis",
    delay: "anim-1",
  },
  {
    key: "speech",
    hdrClass: "hdr-blue",
    cardClass: "blue",
    eyebrow: "Language",
    title: "Speech & Language Assessment",
    desc: "Upload a clinical speech transcript and receive a language assessment profile based on automatically extracted speech markers and patient demographics.",
    icon: MessageSquareText,
    actionLabel: "Open Assessment",
    delay: "anim-2",
  },
  {
    key: "combined",
    hdrClass: "hdr-emerald",
    cardClass: "emerald",
    eyebrow: "Summary",
    title: "Integrated Patient Summary",
    desc: "Consolidate brain imaging and language assessment outputs into a single structured clinical decision-support summary, ready for clinical review.",
    icon: BarChart3,
    actionLabel: "Open Summary",
    delay: "anim-3",
  },
  {
    key: "reports",
    hdrClass: "hdr-indigo",
    cardClass: "indigo",
    eyebrow: "Records",
    title: "Patient Reports",
    desc: "Access all generated clinical reports for this patient case. Download in HTML, PDF or Markdown. Recover previous patient cases using the case ID lookup.",
    icon: FileText,
    actionLabel: "Open Reports",
    delay: "anim-4",
  },
];

/* ── Workflow steps ─────────────────────────── */
function workflowSteps(brainResult, speechResult, reports) {
  const hasCombined = reports.some((r) => r.type === "Combined Multimodal");
  const hasArchived = reports.length >= 3;
  return [
    {
      label: "Patient Case Created",
      sub: "Case ID generated",
      icon: ClipboardList,
      state: "done",
    },
    {
      label: "Brain Imaging Analysis",
      sub: brainResult ? brainResult.prediction : "Awaiting upload",
      icon: Brain,
      state: brainResult ? "done" : "active",
    },
    {
      label: "Speech-Language Assessment",
      sub: speechResult ? speechResult.prediction : "Awaiting transcript",
      icon: MessageSquareText,
      state: speechResult ? "done" : brainResult ? "active" : "pending",
    },
    {
      label: "Integrated Summary",
      sub: hasCombined ? "Report generated" : "Awaiting assessments",
      icon: BarChart3,
      state: hasCombined ? "done" : speechResult ? "active" : "pending",
    },
    {
      label: "Reports Archived",
      sub: hasArchived ? `${reports.length} reports saved` : "Pending completion",
      icon: FolderArchive,
      state: hasArchived ? "done" : "pending",
    },
  ];
}

/* ── Status badge helper ────────────────────── */
function StatusBadge({ done, label }) {
  if (done) {
    return (
      <span className="dash-module-status dash-status-complete">
        <span className="dash-status-dot" />
        {label || "Complete"}
      </span>
    );
  }
  return (
    <span className="dash-module-status dash-status-pending">
      <span className="dash-status-dot" />
      Pending
    </span>
  );
}

/* ═══════════════════════════════════════════════
   DASHBOARD
   ═══════════════════════════════════════════════ */
export default function Home({ patientCaseId, brainResult, speechResult, reports, navigate }) {
  const now = new Date();
  const steps = workflowSteps(brainResult, speechResult, reports);

  const moduleStatus = {
    brain:    !!brainResult,
    speech:   !!speechResult,
    combined: reports.some((r) => r.type === "Combined Multimodal"),
    reports:  reports.length > 0,
  };

  return (
    <div className="page-content">

      {/* ── 1. Premium header ──────────────────── */}
      <header className="dash-header">
        <div className="dash-header-left">
          <div className="dash-header-eyebrow">
            <Activity size={11} />
            Alzheimer Clinical Decision-Support Platform
          </div>

          <h1 className="dash-header-title">
            Clinical Decision-Support<br />Dashboard
          </h1>

          <p className="dash-header-subtitle">
            Integrated brain imaging and speech-language assessment for clinical review.
            All outputs are for decision-support only and must be reviewed by a qualified
            healthcare professional.
          </p>

          <div className="dash-case-chip">
            <div>
              <div className="dash-case-lbl">Active Patient Case</div>
              <div className="dash-case-val">{patientCaseId}</div>
            </div>
            <div className="dash-case-sep" />
            <div>
              <div className="dash-case-lbl">Session Status</div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
                <span style={{
                  width: 7, height: 7, borderRadius: "50%",
                  background: "#6ee7b7", display: "inline-block",
                  boxShadow: "0 0 0 2px rgba(110,231,183,0.30)",
                }} />
                <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "#fff" }}>Active</span>
              </div>
            </div>
          </div>
        </div>

        <div className="dash-header-right">
          {/* Date-time widget */}
          <div className="dash-datetime">
            <div className="dash-datetime-time">{formatTime(now)}</div>
            <div className="dash-datetime-date">{formatDate(now)}</div>
          </div>

          {/* Safety badge */}
          <div className="dash-safety-badge">
            <ShieldCheck size={13} />
            Decision Support Only
          </div>

          {/* Quick stats inside header */}
          <div className="dash-stat-chips">
            <div className="dash-stat-chip">
              <span className="dash-stat-chip-val">{reports.length}</span>
              <span className="dash-stat-chip-lbl">Reports</span>
            </div>
            <div className="dash-stat-chip">
              <span className="dash-stat-chip-val">{brainResult ? "✓" : "–"}</span>
              <span className="dash-stat-chip-lbl">Imaging</span>
            </div>
            <div className="dash-stat-chip">
              <span className="dash-stat-chip-val">{speechResult ? "✓" : "–"}</span>
              <span className="dash-stat-chip-lbl">Language</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Session results (conditional) ─────── */}
      {(brainResult || speechResult) && (
        <div className="dash-results-row">
          {brainResult && (
            <div className="dash-result-chip">
              <div className="dash-result-chip-icon">
                <Brain size={18} />
              </div>
              <div>
                <div className="dash-result-chip-lbl">Brain Imaging Profile</div>
                <div className="dash-result-chip-val">{brainResult.prediction}</div>
                <div className="dash-result-chip-conf">Confidence: {brainResult.confidence}</div>
              </div>
            </div>
          )}
          {speechResult && (
            <div className="dash-result-chip">
              <div className="dash-result-chip-icon">
                <MessageSquareText size={18} />
              </div>
              <div>
                <div className="dash-result-chip-lbl">Language Assessment Profile</div>
                <div className="dash-result-chip-val">{speechResult.prediction}</div>
                <div className="dash-result-chip-conf">Confidence: {speechResult.confidence}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── 2. Module cards ───────────────────── */}
      <div className="dash-section-label">
        <Stethoscope size={12} />
        Clinical Modules
      </div>

      <div className="dash-modules-grid">
        {MODULES.map((mod) => {
          const Icon = mod.icon;
          const isDone = moduleStatus[mod.key];
          return (
            <button
              key={mod.key}
              type="button"
              className={`dash-module-card ${mod.cardClass} ${mod.delay}`}
              onClick={() => navigate(mod.key)}
              aria-label={`Open ${mod.title}`}
            >
              {/* Colored icon header */}
              <div className={`dash-module-icon-header ${mod.hdrClass}`}>
                <div className="dash-module-icon">
                  <Icon size={27} />
                </div>
              </div>

              {/* Card content */}
              <div className="dash-module-body">
                <div className="dash-module-eyebrow">{mod.eyebrow}</div>
                <div className="dash-module-title">{mod.title}</div>
                <div className="dash-module-desc">{mod.desc}</div>
              </div>

              {/* Footer: status + CTA */}
              <div className="dash-module-footer">
                <StatusBadge
                  done={isDone}
                  label={mod.key === "reports" ? `${reports.length} Reports` : "Complete"}
                />
                <button
                  type="button"
                  className="dash-module-btn"
                  tabIndex={-1}
                  aria-hidden="true"
                >
                  {mod.actionLabel}
                  <ArrowRight size={12} />
                </button>
              </div>
            </button>
          );
        })}
      </div>

      {/* ── 3. Workflow timeline ──────────────── */}
      <div className="dash-section-label">
        <Clock size={12} />
        Patient Workflow
      </div>

      <div className="dash-timeline-card">
        <div style={{ marginBottom: 20 }}>
          <div style={{
            fontFamily: "var(--font-display)", fontWeight: 700,
            fontSize: "0.95rem", color: "var(--text-primary)", marginBottom: 4,
          }}>
            Clinical Assessment Pathway
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Track the progress of this patient case through each stage of the assessment workflow.
          </div>
        </div>

        <div className="dash-timeline">
          {steps.map((step, i) => {
            const Icon = step.icon;
            const isDone    = step.state === "done";
            const isActive  = step.state === "active";
            return (
              <div
                key={step.label}
                className={`dash-timeline-step${isDone ? " done" : ""}`}
              >
                <div className={`dash-timeline-node ${step.state}`}>
                  {isDone
                    ? <CheckCircle2 size={17} />
                    : isActive
                      ? <Icon size={17} />
                      : <Icon size={17} />
                  }
                </div>
                <div className="dash-timeline-info">
                  <div className="dash-timeline-name">{step.label}</div>
                  <div className={`dash-timeline-sub ${isDone ? "done-txt" : isActive ? "active-txt" : ""}`}>
                    {isDone
                      ? step.sub
                      : isActive
                        ? "In progress"
                        : "Pending"}
                  </div>
                  {isDone && (
                    <div style={{ marginTop: 2, fontSize: "0.62rem", color: "var(--text-faint)" }}>
                      {step.sub}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── 4. Clinical safety note ───────────── */}
      <div className="dash-safety-note">
        <div className="dash-safety-icon">
          <TriangleAlert size={20} />
        </div>
        <div className="dash-safety-content">
          <div className="dash-safety-title">
            Clinical Safety Notice
          </div>
          <p className="dash-safety-text">
            <strong>This platform provides decision-support information only. It does not provide a medical diagnosis.</strong>{" "}
            All results generated by this platform — including brain imaging assessments,
            speech-language assessment profiles, and integrated patient summaries —
            must be reviewed and interpreted by a qualified healthcare professional
            before any clinical action is taken. Do not use these outputs as the sole
            basis for clinical decision-making.
          </p>
        </div>
      </div>

    </div>
  );
}
