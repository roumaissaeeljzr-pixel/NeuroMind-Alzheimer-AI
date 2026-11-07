export default function StatCard({ label, value, helper, tone = "teal", icon: Icon }) {
  return (
    <article className={`stat-card ${tone}`}>
      <div className="stat-card-row">
        <div className="stat-label">{label}</div>
        {Icon && (
          <div
            className="card-icon-wrap icon-mint"
            style={{ width: 32, height: 32, borderRadius: 8 }}
          >
            <Icon size={14} />
          </div>
        )}
      </div>
      <div className="stat-value">{value}</div>
      {helper && (
        <div className="stat-helper">
          <span style={{
            width: 5, height: 5, borderRadius: "50%",
            background: "currentColor", opacity: 0.4, flexShrink: 0,
            display: "inline-block",
          }} />
          {helper}
        </div>
      )}
    </article>
  );
}
