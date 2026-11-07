import { CheckCircle2, UploadCloud } from "lucide-react";

export default function UploadCard({ title, accept, fileName, onChange, description, formats = [] }) {
  return (
    <div className="upload-zone">
      <input
        type="file"
        accept={accept}
        onChange={onChange}
        aria-label={`Upload ${title}`}
      />

      <div className="upload-icon-wrap">
        <UploadCloud size={22} />
      </div>

      <div className="upload-title">{title}</div>

      {description && (
        <div className="upload-hint">{description}</div>
      )}

      {formats.length > 0 && (
        <div className="upload-formats">
          {formats.map((fmt) => (
            <span key={fmt} className="format-pill">{fmt}</span>
          ))}
        </div>
      )}

      {fileName ? (
        <div className="file-pill-selected">
          <CheckCircle2 size={13} />
          {fileName}
        </div>
      ) : (
        <div style={{
          fontSize: "0.72rem",
          color: "var(--text-faint)",
          marginTop: 2,
        }}>
          or drag &amp; drop here
        </div>
      )}
    </div>
  );
}
