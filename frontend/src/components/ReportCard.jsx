import { Download, FileText } from "lucide-react";

export default function ReportCard({ report }) {
  return (
    <article className="report-card">
      <div className="report-card-icon">
        <FileText size={16} />
      </div>

      <div className="report-card-info">
        <div className="report-type">{report.type}</div>
        <div className="report-title">{report.title}</div>
        <div className="report-date">{report.date}</div>
      </div>

      <span
        className={`status-pill ${report.status === "Ready" ? "status-ready" : "status-pending"}`}
        style={{ flexShrink: 0 }}
      >
        {report.status}
      </span>

      <button
        type="button"
        className="btn-icon"
        aria-label={`Download ${report.title}`}
        title="Download report"
        style={{ flexShrink: 0 }}
      >
        <Download size={14} />
      </button>
    </article>
  );
}
