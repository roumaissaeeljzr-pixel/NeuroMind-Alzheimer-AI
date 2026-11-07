export default function LoadingOverlay({
  message = "Processing...",
  sub     = "Please wait while we prepare the results.",
}) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite" aria-label={message}>
      <div className="loading-box">
        {/* Dual-ring spinner */}
        <div className="loading-spinner" />

        <div>
          <div className="loading-text">{message}</div>
          <div className="loading-sub">{sub}</div>
        </div>

        {/* Animated progress dots */}
        <div style={{ display: "flex", gap: 6 }}>
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              style={{
                width: 6, height: 6,
                borderRadius: "50%",
                background: "var(--teal-300)",
                opacity: 0.7,
                animation: `pulse-dot 1.4s ease-in-out ${i * 0.22}s infinite`,
                display: "inline-block",
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
