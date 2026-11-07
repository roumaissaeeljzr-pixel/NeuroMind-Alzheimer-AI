import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock,
  Download,
  FileText,
  MessageSquareText,
  Search,
} from "lucide-react";
import { useState } from "react";
import ReportCard from "../components/ReportCard.jsx";
import StatCard from "../components/StatCard.jsx";

/* ── Mock data ───────────────────────────────── */
const RECENT_CASES = [
  { id: "PAT-20260625-024256-A69E", label: "Jun 25, 2026 · 02:42" },
  { id: "PAT-20260625-004447-7B2C", label: "Jun 25, 2026 · 00:44" },
  { id: "PAT-20260624-221930-D4A1", label: "Jun 24, 2026 · 22:19" },
];

const MOCK_TIMELINE = [
  {
    title:  "Patient Case Created",
    time:   "2026-06-25  18:35",
    result: "Case ID: PAT-20260625-183502-CC1F",
    done:   true,
    icon:   Activity,
  },
  {
    title:  "Brain Imaging Analysis",
    time:   "2026-06-25  18:40",
    result: "Cognitively Normal Profile · 97.36% confidence",
    done:   true,
    icon:   Brain,
  },
  {
    title:  "Speech & Language Assessment",
    time:   "2026-06-25  18:48",
    result: "Within Normal Range · 90.40% confidence",
    done:   true,
    icon:   MessageSquareText,
  },
  {
    title:  "Integrated Patient Summary",
    time:   "2026-06-25  18:55",
    result: "Summary generated and archived",
    done:   true,
    icon:   BarChart3,
  },
  {
    title:  "Reports Archived",
    time:   "2026-06-25  18:55",
    result: "3 reports available for download",
    done:   true,
    icon:   FileText,
  },
];

/* ── Report section component ────────────────── */
function ReportSection({ title, reports, icon: Icon, color, emptyLabel }) {
  const colorMap = {
    teal:   { bg: "var(--teal-50)",  border: "var(--teal-100)",  iconColor: "var(--teal-500)",  countBg: "var(--teal-50)",  countColor: "var(--teal-600)" },
    blue:   { bg: "var(--blue-50)",  border: "var(--blue-100)",  iconColor: "var(--blue-500)",  countBg: "var(--blue-50)",  countColor: "var(--blue-600)" },
    indigo: { bg: "#eef2ff",         border: "#c7d2fe",          iconColor: "#6366f1",          countBg: "#eef2ff",         countColor: "#4f46e5" },
  };
  const c = colorMap[color] || colorMap.teal;

  return (
    <div className="report-section-card">
      <div className="report-section-header">
        <div
          className="report-section-icon"
          style={{ background: c.bg, border: `1px solid ${c.border}` }}
        >
          <Icon size={17} style={{ color: c.iconColor }} />
        </div>
        <span className="report-section-title">{title}</span>
        <span
          className="report-section-count"
          style={{ background: c.countBg, color: c.countColor, border: `1px solid ${c.border}` }}
        >
          {reports.length} {reports.length === 1 ? "report" : "reports"}
        </span>
        {reports.length > 0 && (
          <button type="button" className="btn btn-secondary btn-sm">
            <Download size={13} /> Download All
          </button>
        )}
      </div>

      <div className="report-section-body">
        {reports.length > 0 ? (
          <div className="stack-xs">
            {reports.map((r) => <ReportCard key={r.id} report={r} />)}
          </div>
        ) : (
          <div className="empty-state" style={{ padding: "28px 0" }}>
            <div className="empty-icon">
              <Icon size={20} style={{ color: c.iconColor }} />
            </div>
            <div className="empty-title">{emptyLabel}</div>
            <div className="empty-desc">
              Reports will appear here once an assessment has been completed for this patient case.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   PATIENT REPORTS PAGE
   ───────────────────────────────────────────── */
export default function MyReports({ patientCaseId, setPatientCaseId, reports }) {
  const [searchVal, setSearchVal]   = useState("");
  const [searchInput, setSearchInput] = useState("");

  const brainReports    = reports.filter((r) => r.type === "Brain Scan");
  const speechReports   = reports.filter((r) => r.type === "Speech and Language");
  const combinedReports = reports.filter((r) => r.type === "Combined Multimodal");

  const handleSearch = () => {
    const trimmed = searchInput.trim();
    if (trimmed) {
      setPatientCaseId(trimmed);
      setSearchVal(trimmed);
    }
  };

  return (
    <div className="page-content">

      {/* ── Page header ── */}
      <div className="page-header">
        <div className="page-eyebrow">
          <div className="page-eyebrow-dot" />
          <span className="page-eyebrow-text">Patient Reports</span>
        </div>
        <h1 className="page-title">Patient Reports</h1>
        <p className="page-subtitle">
          Access, review, and download clinical assessment reports. Retrieve previous
          patient cases using the case ID lookup.
        </p>
      </div>

      <div className="stack">

        {/* Patient Case ID card */}
        <div className="case-id-banner">
          <div className="case-id-banner-left">
            <div className="case-id-icon">
              <Activity size={18} />
            </div>
            <div>
              <div className="case-id-label">Active Patient Case</div>
              <div className="case-id-value">{patientCaseId}</div>
            </div>
          </div>
          <div className="case-id-banner-right">
            <span className="badge badge-teal">
              {reports.length} report{reports.length !== 1 ? "s" : ""} on record
            </span>
          </div>
        </div>

        {/* ① Search patient case ID */}
        <div>
          <div className="step-header">
            <div className="step-num">1</div>
            <span className="step-title">Search Patient Case ID</span>
            <div className="step-sep" />
          </div>

          <div className="search-panel">
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
              Retrieve Previous Patient Case
            </div>
            <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", marginBottom: 14 }}>
              Enter a patient case ID to load all associated assessment reports, or select a
              recent case from the list below.
            </p>

            <div className="search-input-row">
              <div className="search-input-wrap">
                <Search />
                <input
                  type="text"
                  placeholder="Paste Patient Case ID (PAT-YYYYMMDD-HHMMSS-XXXX)"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSearch();
                  }}
                />
              </div>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleSearch}
                disabled={!searchInput.trim()}
              >
                <Search size={14} /> Retrieve Case
              </button>
            </div>

            {searchVal && (
              <div style={{
                marginTop: 12, padding: "10px 14px",
                background: "var(--success-bg)", border: "1px solid var(--success-100)",
                borderRadius: "var(--r-md)",
                fontSize: "0.78rem", color: "var(--success-600)", fontWeight: 600,
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <CheckCircle2 size={14} />
                Patient case loaded: {searchVal}
              </div>
            )}
          </div>
        </div>

        {/* ② Recent patient cases */}
        <div>
          <div className="step-header">
            <div className="step-num">2</div>
            <span className="step-title">Recent Patient Cases</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <p style={{ fontSize: "0.79rem", color: "var(--text-muted)", marginBottom: 14 }}>
                Select a recent patient case to switch the active case and view its reports.
              </p>
              <div className="recent-cases-grid">
                {RECENT_CASES.map(({ id, label }) => (
                  <button
                    key={id}
                    type="button"
                    className="recent-case-chip"
                    onClick={() => setPatientCaseId(id)}
                    title={`Switch to case ${id}`}
                  >
                    <div>
                      <div className="recent-case-chip-lbl">Case</div>
                      <div className="recent-case-chip-id">{id}</div>
                    </div>
                    <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: 2 }}>{label}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid-4">
          <StatCard
            label="Total Reports"
            value={reports.length}
            helper="All formats"
            tone="teal"
            icon={FileText}
          />
          <StatCard
            label="Imaging Reports"
            value={brainReports.length}
            helper="Brain imaging"
            tone="teal"
            icon={Brain}
          />
          <StatCard
            label="Language Reports"
            value={speechReports.length}
            helper="Speech assessments"
            tone="violet"
            icon={MessageSquareText}
          />
          <StatCard
            label="Summaries"
            value={combinedReports.length}
            helper="Integrated"
            tone="amber"
            icon={BarChart3}
          />
        </div>

        {/* ③ Analysis timeline */}
        <div>
          <div className="step-header">
            <div className="step-num">3</div>
            <span className="step-title">Analysis Timeline</span>
            <div className="step-sep" />
          </div>

          <div className="analysis-timeline">
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
              Clinical Assessment History
            </div>
            <p style={{ fontSize: "0.79rem", color: "var(--text-muted)", marginBottom: 20 }}>
              Chronological record of all assessments performed for this patient case.
            </p>

            {MOCK_TIMELINE.map((item, i) => {
              const Icon = item.icon;
              return (
                <div key={item.title} className="timeline-entry">
                  <div className="timeline-track">
                    <div className={`timeline-node ${item.done ? "complete" : "pending"}`}>
                      {item.done ? <CheckCircle2 size={15} /> : <Clock size={15} />}
                    </div>
                    {i < MOCK_TIMELINE.length - 1 && <div className="timeline-connector" />}
                  </div>
                  <div className="timeline-entry-body">
                    <div className="timeline-entry-title">{item.title}</div>
                    <div className="timeline-entry-meta">
                      <span className="timeline-entry-time">
                        <Clock size={11} style={{ display: "inline", marginRight: 3 }} />
                        {item.time}
                      </span>
                      {item.done && (
                        <span className="timeline-entry-result">{item.result}</span>
                      )}
                    </div>
                  </div>
                  <div style={{ flexShrink: 0 }}>
                    <span className={`status-pill ${item.done ? "status-ready" : "status-pending"}`}>
                      {item.done ? "Complete" : "Pending"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ④ Brain imaging reports */}
        <div>
          <div className="step-header">
            <div className="step-num">4</div>
            <span className="step-title">Brain Imaging Analysis Reports</span>
            <div className="step-sep" />
          </div>
          <ReportSection
            title="Brain Imaging Analysis Reports"
            reports={brainReports}
            icon={Brain}
            color="teal"
            emptyLabel="No brain imaging reports on record"
          />
        </div>

        {/* ⑤ Speech-language reports */}
        <div>
          <div className="step-header">
            <div className="step-num">5</div>
            <span className="step-title">Speech & Language Assessment Reports</span>
            <div className="step-sep" />
          </div>
          <ReportSection
            title="Speech & Language Assessment Reports"
            reports={speechReports}
            icon={MessageSquareText}
            color="blue"
            emptyLabel="No language assessment reports on record"
          />
        </div>

        {/* ⑥ Integrated summaries */}
        <div>
          <div className="step-header">
            <div className="step-num">6</div>
            <span className="step-title">Integrated Patient Summary Reports</span>
            <div className="step-sep" />
          </div>
          <ReportSection
            title="Integrated Patient Summary Reports"
            reports={combinedReports}
            icon={BarChart3}
            color="indigo"
            emptyLabel="No integrated summaries on record"
          />
        </div>

      </div>
    </div>
  );
}
