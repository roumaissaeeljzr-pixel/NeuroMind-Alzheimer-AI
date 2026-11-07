import { CheckCircle2, TriangleAlert } from "lucide-react";

export default function ResultCard({ title, result, confidence, description, tone = "good" }) {
  const Icon = tone === "alert" ? TriangleAlert : CheckCircle2;
  const pct  = Math.min(100, Math.max(0, parseFloat(confidence) || 0));

  return (
    <section className={`result-card ${tone}`}>
      <div className="result-card-top">
        <div className={`result-icon-wrap ${tone}`}>
          <Icon size={20} />
        </div>
        <div style={{ flex: 1 }}>
          <div className="result-eyebrow">{title}</div>
          <div className={`result-value ${tone === "good" ? "teal" : "amber"}`}>{result}</div>
        </div>
      </div>

      {confidence && (
        <div className="confidence-wrap">
          <div className="confidence-hd">
            <span className="confidence-lbl">Confidence Score</span>
            <span className="confidence-pct">{confidence}</span>
          </div>
          <div className="confidence-track">
            <div
              className={`confidence-fill${tone === "alert" ? " amber" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {description && (
        <p className="result-desc">{description}</p>
      )}
    </section>
  );
}
