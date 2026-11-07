import { ShieldAlert } from "lucide-react";

export default function MedicalNote({ children, title = "Clinical Safety Notice" }) {
  return (
    <aside className="medical-note" role="note" aria-label={title}>
      <div className="medical-note-icon">
        <ShieldAlert size={16} />
      </div>
      <div style={{ flex: 1 }}>
        <div className="medical-note-title">
          {title}
        </div>
        <div className="medical-note-body">{children}</div>
      </div>
    </aside>
  );
}
