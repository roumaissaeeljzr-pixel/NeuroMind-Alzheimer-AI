import { Plus, ShieldCheck, User } from "lucide-react";

export default function TopBar({ patientCaseId, onNewCase }) {
  return (
    <header className="topbar">
      {/* Patient identity block */}
      <div className="topbar-patient">
        <div className="topbar-avatar" aria-hidden="true">
          <User size={16} />
        </div>
        <div>
          <div className="topbar-label">Active Patient Case</div>
          <div className="topbar-id">{patientCaseId}</div>
        </div>
      </div>

      <div className="topbar-sep" aria-hidden="true" />

      {/* Decision-support status badge */}
      <div className="topbar-status" role="status">
        <div className="topbar-dot" aria-hidden="true" />
        <ShieldCheck size={12} aria-hidden="true" />
        <span>Decision Support Only</span>
      </div>

      <div style={{ flex: 1 }} />

      {/* Actions */}
      <div className="topbar-actions">
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={onNewCase}
        >
          <Plus size={13} />
          New Case
        </button>
      </div>
    </header>
  );
}
