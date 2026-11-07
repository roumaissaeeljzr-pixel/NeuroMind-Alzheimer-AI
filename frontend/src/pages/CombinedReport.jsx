import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Circle,
  Download,
  FileText,
  MessageSquareText,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useMemo, useState } from "react";
import MedicalNote from "../components/MedicalNote.jsx";

/* ─────────────────────────────────────────────
   INTEGRATED PATIENT SUMMARY PAGE
   ───────────────────────────────────────────── */
export default function CombinedReport({ patientCaseId, brainResult, speechResult, setReports, setIsBusy }) {
  const [generated, setGenerated] = useState(false);

  const hasAny  = brainResult || speechResult;
  const hasBoth = brainResult && speechResult;

  const summaryText = useMemo(() => {
    if (hasBoth) {
      return `Both the brain imaging assessment and the speech and language assessment have been completed for this patient case. The brain imaging profile indicates "${brainResult.prediction}" with ${brainResult.confidence} confidence. The language assessment profile indicates "${speechResult.prediction}" with ${speechResult.confidence} confidence. Both assessments were conducted independently and their results should not be merged into a single clinical conclusion without qualified professional review.`;
    }
    if (brainResult) {
      return `Only the brain imaging assessment has been completed for this patient case (${brainResult.prediction}, ${brainResult.confidence} confidence). A speech and language assessment has not yet been performed. A partial integrated summary can support clinical review pending the language assessment.`;
    }
    if (speechResult) {
      return `Only the speech and language assessment has been completed (${speechResult.prediction}, ${speechResult.confidence} confidence). A brain imaging analysis has not yet been performed. A partial integrated summary can support clinical review pending the imaging assessment.`;
    }
    return "No assessments have been completed for this patient case. Please complete at least one assessment before generating an integrated summary.";
  }, [brainResult, speechResult, hasBoth]);

  const handleGenerate = () => {
    setIsBusy(true);
    window.setTimeout(() => {
      setGenerated(true);
      setReports((prev) => [
        {
          id: `combined-${Date.now()}`,
          type: "Combined Multimodal",
          title: "Integrated Patient Summary Report",
          date: new Date().toLocaleString(),
          status: "Ready",
        },
        ...prev,
      ]);
      setIsBusy(false);
    }, 900);
  };

  return (
    <div className="page-content">

      {/* ── Page header ── */}
      <div className="page-header">
        <div className="page-eyebrow">
          <div className="page-eyebrow-dot" />
          <span className="page-eyebrow-text">Integrated Patient Summary</span>
        </div>
        <h1 className="page-title">Integrated Patient Summary</h1>
        <p className="page-subtitle">
          Consolidate brain imaging and language assessment outputs into a structured
          clinical decision-support summary for review or referral.
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
            <span className={`status-pill ${generated ? "status-ready" : "status-pending"}`}>
              {generated ? "Summary Generated" : "Summary Pending"}
            </span>
            {generated && (
              <span className="badge badge-green">
                <CheckCircle2 size={10} /> Ready to Download
              </span>
            )}
          </div>
        </div>

        {/* ① Brain imaging assessment status */}
        {/* ② Speech & language assessment status */}
        <div>
          <div className="step-header">
            <div className="step-num">1</div>
            <span className="step-title">Assessment Status Overview</span>
            <div className="step-sep" />
          </div>

          <div className="grid-2">
            {/* Brain imaging status */}
            <div className="assessment-status-card">
              <div className={`assessment-status-icon ${brainResult ? "available" : "missing"}`}>
                {brainResult ? <CheckCircle2 /> : <Circle />}
              </div>
              <div className="assessment-status-body">
                <div className="assessment-status-label">Brain Imaging Analysis</div>
                <div className="assessment-status-title">
                  {brainResult ? brainResult.prediction : "Not Yet Completed"}
                </div>
                <div className="assessment-status-detail">
                  {brainResult
                    ? `Confidence: ${brainResult.confidence} · Assessment complete`
                    : "Run the Brain Imaging Analysis to include imaging results in this summary."}
                </div>
              </div>
              <div>
                <span className={`assessment-status-badge ${brainResult ? "available" : "missing"}`}>
                  {brainResult ? "Available" : "Missing"}
                </span>
              </div>
            </div>

            {/* Speech assessment status */}
            <div className="assessment-status-card">
              <div className={`assessment-status-icon ${speechResult ? "available" : "missing"}`}>
                {speechResult ? <CheckCircle2 /> : <Circle />}
              </div>
              <div className="assessment-status-body">
                <div className="assessment-status-label">Speech &amp; Language Assessment</div>
                <div className="assessment-status-title">
                  {speechResult ? speechResult.prediction : "Not Yet Completed"}
                </div>
                <div className="assessment-status-detail">
                  {speechResult
                    ? `Confidence: ${speechResult.confidence} · Assessment complete`
                    : "Run the Speech & Language Assessment to include language results in this summary."}
                </div>
              </div>
              <div>
                <span className={`assessment-status-badge ${speechResult ? "available" : "missing"}`}>
                  {speechResult ? "Available" : "Missing"}
                </span>
              </div>
            </div>
          </div>

          {/* Readiness indicator */}
          {!hasAny && (
            <div className="info-box" style={{ marginTop: 14 }}>
              <div className="info-box-title">No Assessments Available</div>
              <div className="info-box-body">
                Complete at least one assessment (Brain Imaging Analysis or Speech &amp; Language Assessment)
                before generating the integrated summary.
              </div>
            </div>
          )}

          {hasAny && !hasBoth && (
            <div style={{
              marginTop: 14,
              padding: "12px 16px",
              background: "#fffbeb",
              border: "1px solid rgba(245,158,11,0.22)",
              borderRadius: "var(--r-lg)",
              fontSize: "0.79rem",
              color: "#92400e",
            }}>
              <strong>Partial summary only.</strong> Only one assessment is available.
              Complete {brainResult ? "the Speech & Language Assessment" : "the Brain Imaging Analysis"} for a full integrated summary.
            </div>
          )}
        </div>

        {/* ③ Generate button */}
        <div>
          <div className="step-header">
            <div className="step-num">2</div>
            <span className="step-title">Generate Integrated Patient Summary</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sm" style={{ marginBottom: 14 }}>
                <div className="card-icon-wrap icon-teal">
                  <BarChart3 size={18} />
                </div>
                <div>
                  <div className="card-label">Summary Preparation</div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)" }}>
                    Integrated Assessment Interpretation
                  </div>
                </div>
              </div>

              {/* Interpretation text */}
              <div style={{
                background: "rgba(15,143,117,0.04)",
                border: "1px solid var(--border-line)",
                borderRadius: "var(--r-lg)",
                padding: "16px 18px",
                marginBottom: 20,
                fontSize: "0.83rem",
                color: "var(--text-secondary)",
                lineHeight: 1.65,
                whiteSpace: "pre-line",
              }}>
                {summaryText}
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="btn btn-primary btn-lg"
                  onClick={handleGenerate}
                  disabled={!hasAny}
                >
                  {generated ? <RefreshCw size={15} /> : <Sparkles size={15} />}
                  {generated ? "Regenerate Integrated Summary" : "Generate Integrated Patient Summary"}
                </button>
                {!hasAny && (
                  <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                    Complete at least one assessment to enable this action.
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ④ Clinical Decision-Support Explanation (shown after generation) */}
        {generated && (
          <div>
            <div className="step-header">
              <div className="step-num">3</div>
              <span className="step-title">Clinical Decision-Support Explanation</span>
              <div className="step-sep" />
            </div>

            <div className="summary-output">
              <div className="summary-output-header">
                <div className="row-sm">
                  <ShieldCheck size={16} style={{ color: "var(--teal-600)" }} />
                  <span className="summary-output-title">Integrated Patient Summary — {patientCaseId}</span>
                </div>
                <span className="badge badge-green">
                  <CheckCircle2 size={10} /> Generated
                </span>
              </div>

              <div className="summary-output-body">
                <div className="summary-finding">
                  <div className="summary-finding-icon">
                    <FileText />
                  </div>
                  <div>
                    <div className="summary-finding-label">Summary Scope</div>
                    <div className="summary-finding-text">
                      This integrated summary consolidates independently generated assessment outputs for
                      patient case <strong>{patientCaseId}</strong>. The two assessments were performed
                      independently and their results must not be merged into a single clinical score.
                    </div>
                  </div>
                </div>

                {brainResult && (
                  <div className="summary-finding">
                    <div className="summary-finding-icon">
                      <Brain />
                    </div>
                    <div>
                      <div className="summary-finding-label">Brain Imaging Analysis</div>
                      <div className="summary-finding-text">
                        Profile: <strong>{brainResult.prediction}</strong> · Confidence: <strong>{brainResult.confidence}</strong>
                        <br />
                        The structural imaging assessment did not identify a pattern strongly associated
                        with neurodegenerative change. The brain imaging profile is consistent with the
                        cognitively normal reference group.
                      </div>
                    </div>
                  </div>
                )}

                {speechResult && (
                  <div className="summary-finding">
                    <div className="summary-finding-icon">
                      <MessageSquareText />
                    </div>
                    <div>
                      <div className="summary-finding-label">Speech &amp; Language Assessment</div>
                      <div className="summary-finding-text">
                        Profile: <strong>{speechResult.prediction}</strong> · Confidence: <strong>{speechResult.confidence}</strong>
                        <br />
                        The language markers extracted from the speech transcript were within the
                        expected range for the patient's demographic profile. No significant atypical
                        language pattern was identified.
                      </div>
                    </div>
                  </div>
                )}

                <div className="summary-finding">
                  <div className="summary-finding-icon">
                    <ShieldCheck />
                  </div>
                  <div>
                    <div className="summary-finding-label">Clinical Interpretation Guidance</div>
                    <div className="summary-finding-text">
                      Both assessments yielded outputs consistent with a low-concern profile.
                      However, <strong>these outputs do not constitute a medical diagnosis</strong> and
                      must be reviewed in the context of the patient's full clinical history by a
                      qualified neurologist, geriatrician, or specialist clinician.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ⑤ Medical safety notes */}
        <MedicalNote title="Clinical Safety Notice — Assessment Independence">
          Brain imaging and speech-language assessments are independent clinical decision-support
          tools. They were trained on separate datasets that are not paired at the individual
          patient level. This integrated summary presents both outputs side-by-side for clinical
          convenience. It does not produce a combined score or merged diagnosis.
        </MedicalNote>

        <MedicalNote title="Clinical Safety Notice — Not a Diagnosis">
          This integrated patient summary is not a medical diagnosis. All assessment outputs
          must be reviewed and interpreted by a qualified healthcare professional before any
          clinical action is taken. Do not use these outputs as the sole basis for clinical
          decision-making.
        </MedicalNote>

        {/* ⑥ Download integrated reports */}
        <div>
          <div className="step-header">
            <div className="step-num">4</div>
            <span className="step-title">Download Integrated Reports</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body" style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
                  Integrated Patient Summary Report
                </div>
                <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", lineHeight: 1.5, maxWidth: 480 }}>
                  {generated
                    ? "The integrated summary is ready. Download in your preferred format for clinical records or referral documentation."
                    : "Generate the integrated patient summary above to unlock report downloads."}
                </p>
              </div>
              <div className="download-group" style={{ flexShrink: 0 }}>
                <button type="button" className="btn btn-secondary" disabled={!generated}>
                  <Download size={14} /> HTML
                </button>
                <button type="button" className="btn btn-secondary" disabled={!generated}>
                  <Download size={14} /> PDF
                </button>
                <button type="button" className="btn btn-secondary" disabled={!generated}>
                  <Download size={14} /> Markdown
                </button>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
