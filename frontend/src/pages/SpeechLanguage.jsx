import {
  Activity,
  CheckCircle2,
  Download,
  FileText,
  Mic,
  Play,
  User,
} from "lucide-react";
import { useState } from "react";
import MedicalNote from "../components/MedicalNote.jsx";
import ResultCard from "../components/ResultCard.jsx";
import StatCard from "../components/StatCard.jsx";
import UploadCard from "../components/UploadCard.jsx";

/* ── Mock transcript ─────────────────────────── */
const MOCK_TRANSCRIPT = `*PAR: well the woman is washing the dishes and there's a little boy um [/] um trying to get a cookie from the cookie jar .
*PAR: and the [//] and the stool [//] the stool that he's standing on is about to fall over .
*PAR: and [/] and the um [/] the woman seems to be [//] to not notice that the um [//] the kitchen is flooding .
*PAR: there's water overflowing from the sink and she's just [//] she just keeps wiping the dishes .
*PAR: and in the background you can see um some trees through the window .`;

/* ── Speech markers data ─────────────────────── */
const speechMarkers = [
  { label: "Hesitation words",      value: 2,  max: 10, flag: "noted"  },
  { label: "Interrupted words",     value: 0,  max: 10, flag: "normal" },
  { label: "Non-verbal markers",    value: 0,  max: 10, flag: "normal" },
  { label: "Self-corrections",      value: 1,  max: 10, flag: "noted"  },
  { label: "Unclear words",         value: 0,  max: 10, flag: "normal" },
  { label: "Pauses (filled)",       value: 3,  max: 10, flag: "noted"  },
];

const demographics = [
  { label: "Patient age",        value: "62" },
  { label: "Sex",                value: "Female" },
  { label: "Years of education", value: "12" },
];

/* ─────────────────────────────────────────────
   SPEECH & LANGUAGE ASSESSMENT PAGE
   ───────────────────────────────────────────── */
export default function SpeechLanguage({ patientCaseId, speechResult, setSpeechResult, setReports, setIsBusy }) {
  const [fileName, setFileName]         = useState("");
  const [showTranscript, setShowTranscript] = useState(false);

  const handleAnalyze = () => {
    setIsBusy(true);
    window.setTimeout(() => {
      setSpeechResult({ prediction: "Within Normal Range", confidence: "90.40%" });
      setReports((prev) => [
        {
          id: `speech-${Date.now()}`,
          type: "Speech and Language",
          title: "Speech & Language Assessment Report",
          date: new Date().toLocaleString(),
          status: "Ready",
        },
        ...prev,
      ]);
      setIsBusy(false);
    }, 900);
  };

  const result = speechResult || { prediction: "Within Normal Range", confidence: "90.40%" };
  const confPct = parseFloat(result.confidence) || 0;

  return (
    <div className="page-content">

      {/* ── Page header ── */}
      <div className="page-header">
        <div className="page-eyebrow">
          <div className="page-eyebrow-dot" />
          <span className="page-eyebrow-text">Speech & Language Assessment</span>
        </div>
        <h1 className="page-title">Speech & Language Assessment</h1>
        <p className="page-subtitle">
          Upload a clinical speech transcript to receive a language assessment profile
          with confidence score and clinical decision-support explanation.
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
            <span className={`status-pill ${speechResult ? "status-ready" : "status-pending"}`}>
              {speechResult ? "Assessment Complete" : "Awaiting Transcript"}
            </span>
            {speechResult && (
              <span className="badge badge-teal">
                <CheckCircle2 size={10} /> {result.confidence} Confidence
              </span>
            )}
          </div>
        </div>

        {/* ① Patient information form */}
        <div>
          <div className="step-header">
            <div className="step-num">1</div>
            <span className="step-title">Patient Information</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sm" style={{ marginBottom: 18 }}>
                <div className="card-icon-wrap icon-teal">
                  <User size={17} />
                </div>
                <div>
                  <div className="card-label">Session Details</div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)" }}>
                    Patient &amp; Clinician Details
                  </div>
                </div>
              </div>

              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label" htmlFor="sp-name">Patient Name</label>
                  <input id="sp-name" type="text" className="form-input" placeholder="Full name (anonymized)" />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="sp-date">Assessment Date</label>
                  <input id="sp-date" type="date" className="form-input" />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="sp-clinician">Responsible Clinician</label>
                  <input id="sp-clinician" type="text" className="form-input" placeholder="Clinician name" />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="sp-referral">Referral Source</label>
                  <input id="sp-referral" type="text" className="form-input" placeholder="e.g., Neurology, GP" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ② Clinical notes */}
        <div>
          <div className="step-header">
            <div className="step-num">2</div>
            <span className="step-title">Clinical Notes</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="form-group">
                <label className="form-label" htmlFor="sp-notes">Clinical Notes</label>
                <textarea
                  id="sp-notes"
                  className="form-textarea"
                  style={{ minHeight: 100 }}
                  placeholder="Optional clinical context for this assessment session. For clinician reference only — this text is not used in the language assessment."
                />
              </div>
              <p style={{ fontSize: "0.73rem", color: "var(--text-muted)", marginTop: 8, fontStyle: "italic" }}>
                Clinical notes are stored for the clinician's reference only and are not processed by the assessment.
              </p>
            </div>
          </div>
        </div>

        {/* ③ Speech transcript upload */}
        <div>
          <div className="step-header">
            <div className="step-num">3</div>
            <span className="step-title">Speech Transcript Upload</span>
            <div className="step-sep" />
          </div>

          <div className="grid-2" style={{ alignItems: "start" }}>
            <UploadCard
              title="Upload Speech Transcript"
              description="Upload a picture-description or conversational speech transcript"
              accept=".cha,.txt"
              fileName={fileName}
              formats={[".cha", ".txt"]}
              onChange={(e) => {
                setFileName(e.target.files?.[0]?.name || "");
                setShowTranscript(true);
              }}
            />
            <div className="card">
              <div className="card-body">
                <div className="card-label" style={{ marginBottom: 8 }}>Transcript Requirements</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    "CHAT format (.cha) preferred for best results",
                    "Plain text (.txt) accepted as alternative",
                    "Transcript should be a picture-description task",
                    "Speaker turns should be clearly labelled",
                    "Ensure all speaker identifiers are anonymized",
                  ].map((note) => (
                    <div key={note} className="row-sm" style={{ alignItems: "flex-start" }}>
                      <CheckCircle2 size={13} style={{ color: "var(--teal-500)", flexShrink: 0, marginTop: 2 }} />
                      <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.45 }}>{note}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ④ Transcript preview */}
        <div>
          <div className="step-header">
            <div className="step-num">4</div>
            <span className="step-title">Transcript Preview</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sb" style={{ marginBottom: 12 }}>
                <div className="row-sm">
                  <div className="card-icon-wrap icon-mint" style={{ width: 32, height: 32 }}>
                    <FileText size={15} />
                  </div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.92rem", color: "var(--text-primary)" }}>
                    Transcript Content
                  </div>
                </div>
                {showTranscript && (
                  <span className="badge badge-green">
                    <CheckCircle2 size={10} /> Loaded
                  </span>
                )}
              </div>

              {showTranscript ? (
                <div className="transcript-viewer">{MOCK_TRANSCRIPT}</div>
              ) : (
                <div className="transcript-viewer-empty">
                  Upload a transcript file above to preview its content here.
                </div>
              )}

              {showTranscript && (
                <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 10, fontStyle: "italic" }}>
                  Transcript shown for review only. The assessment processes the full file content automatically.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* ⑤ Automatically extracted speech information */}
        <div>
          <div className="step-header">
            <div className="step-num">5</div>
            <span className="step-title">Automatically Extracted Speech Information</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sm" style={{ marginBottom: 16 }}>
                <div className="card-icon-wrap icon-teal" style={{ width: 34, height: 34 }}>
                  <Mic size={16} />
                </div>
                <div>
                  <div className="card-label">Extracted Language Markers</div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.92rem", color: "var(--text-primary)" }}>
                    Speech Marker Analysis
                  </div>
                </div>
              </div>

              {/* Speech markers with visual bars */}
              <div style={{ marginBottom: 20 }}>
                <div className="section-head" style={{ marginBottom: 10 }}>
                  <span className="section-head-text">Speech Markers</span>
                  <div className="section-head-line" />
                </div>
                {speechMarkers.map(({ label, value, max, flag }) => (
                  <div key={label} className="marker-row">
                    <div className="marker-name">{label}</div>
                    <div className="marker-bar-wrap">
                      <div
                        className={`marker-bar-fill${value === 0 ? " zero" : ""}`}
                        style={{ width: `${(value / max) * 100}%` }}
                      />
                    </div>
                    <div className="marker-val">{value}</div>
                    <span className={`marker-flag ${flag}`}>
                      {flag === "normal" ? "Normal" : "Noted"}
                    </span>
                  </div>
                ))}
              </div>

              {/* Demographics table */}
              <div>
                <div className="section-head" style={{ marginBottom: 10 }}>
                  <span className="section-head-text">Patient Demographics</span>
                  <div className="section-head-line" />
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Value</th>
                      <th>Reference Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Patient age</td>
                      <td className="val">62</td>
                      <td style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>55–75 typical assessment range</td>
                    </tr>
                    <tr>
                      <td>Sex</td>
                      <td className="val">Female</td>
                      <td style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Used for normative adjustment</td>
                    </tr>
                    <tr>
                      <td>Years of education</td>
                      <td className="val">12</td>
                      <td style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Applied to assessment calibration</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Run assessment button */}
        <div className="card">
          <div className="card-body" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
                Run Language Assessment
              </div>
              <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                {fileName
                  ? `Transcript ready: ${fileName}. Click to generate the language assessment profile and confidence score.`
                  : "Upload a speech transcript (Step 3) to enable the assessment."}
              </p>
            </div>
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={handleAnalyze}
              disabled={!fileName}
              style={{ flexShrink: 0 }}
            >
              <Play size={16} />
              Run Language Assessment
            </button>
          </div>
        </div>

        {/* ⑥ Language assessment profile + ⑦ Confidence score */}
        <div>
          <div className="step-header">
            <div className="step-num">6</div>
            <span className="step-title">Language Assessment Profile &amp; Confidence Score</span>
            <div className="step-sep" />
          </div>

          <div className="grid-2" style={{ alignItems: "start" }}>
            <ResultCard
              title="Language Assessment Profile"
              result={result.prediction}
              confidence={result.confidence}
              description="The language markers extracted from this transcript are within the expected range for the patient's demographic profile. No significant atypical language pattern was identified in this assessment."
              tone="good"
            />

            <div className="stack-sm">
              <StatCard
                label="Assessment Confidence Score"
                value={result.confidence}
                helper="Based on extracted speech markers and patient demographics"
                tone="teal"
              />

              {/* Mini bar for confidence */}
              <div className="card-flat" style={{ padding: "14px 18px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <span style={{ fontSize: "0.72rem", fontWeight: 600, color: "var(--text-muted)" }}>Confidence Score</span>
                  <span style={{ fontSize: "0.84rem", fontWeight: 700, color: "var(--teal-600)" }}>{result.confidence}</span>
                </div>
                <div className="confidence-track">
                  <div className="confidence-fill" style={{ width: `${confPct}%` }} />
                </div>
                <p style={{ fontSize: "0.70rem", color: "var(--text-muted)", marginTop: 8, lineHeight: 1.45 }}>
                  A higher confidence score reflects stronger agreement between the extracted
                  speech markers and the reference dataset. It does not indicate a diagnosis.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ⑧ Simple clinical explanation */}
        <div>
          <div className="step-header">
            <div className="step-num">7</div>
            <span className="step-title">Clinical Decision-Support Explanation</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sm" style={{ marginBottom: 14 }}>
                <div className="card-icon-wrap icon-mint" style={{ width: 34, height: 34 }}>
                  <FileText size={15} />
                </div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.92rem", color: "var(--text-primary)" }}>
                  Simple Clinical Explanation
                </div>
              </div>

              <div className="prose">
                <p>
                  This language assessment was based on a picture-description transcript
                  (Cookie Theft picture task). The assessment identified{" "}
                  <strong>2 hesitation words</strong>, <strong>1 self-correction</strong>,
                  and <strong>3 filled pauses</strong> in the speech sample.
                </p>
                <p>
                  These markers are <strong>within the expected range</strong> for a 62-year-old
                  female patient with 12 years of education. When adjusted for the patient's
                  demographic profile, no significant deviation from the reference group was detected.
                </p>
                <p>
                  The language assessment profile is classified as <strong>"Within Normal Range"</strong>{" "}
                  with a confidence score of <strong>90.40%</strong>. This means the extracted
                  speech pattern shows a high degree of similarity with transcripts from clinically
                  unimpaired speakers in the reference dataset.
                </p>
                <p>
                  This output is for clinical decision-support only. A qualified speech-language
                  pathologist or neurologist must interpret these findings in the context of
                  the patient's full clinical history and a formal cognitive evaluation.
                </p>
              </div>

              {/* Key findings summary */}
              <div style={{ marginTop: 18, padding: "14px 16px", background: "var(--teal-50)", border: "1px solid var(--teal-100)", borderRadius: "var(--r-lg)" }}>
                <div style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--teal-600)", textTransform: "uppercase", letterSpacing: "0.09em", marginBottom: 8 }}>
                  Key Findings Summary
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {[
                    { label: "Speech fluency", value: "Within expected range" },
                    { label: "Lexical access",   value: "No significant difficulty" },
                    { label: "Narrative structure", value: "Coherent and organised" },
                    { label: "Information content", value: "Adequate for task" },
                  ].map(({ label, value }) => (
                    <div key={label} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                      <CheckCircle2 size={13} style={{ color: "var(--teal-500)", flexShrink: 0, marginTop: 2 }} />
                      <div>
                        <div style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--teal-700)" }}>{label}</div>
                        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{value}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ⑨ Report download */}
        <div>
          <div className="step-header">
            <div className="step-num">8</div>
            <span className="step-title">Download Report</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body" style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
                  Speech &amp; Language Assessment Report
                </div>
                <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", lineHeight: 1.5, maxWidth: 480 }}>
                  Includes the extracted speech markers, demographic adjustments, language assessment
                  profile, confidence score, and clinical decision-support explanation.
                </p>
              </div>
              <div className="download-group" style={{ flexShrink: 0 }}>
                <button type="button" className="btn btn-secondary">
                  <Download size={14} /> HTML
                </button>
                <button type="button" className="btn btn-secondary">
                  <Download size={14} /> PDF
                </button>
                <button type="button" className="btn btn-secondary">
                  <Download size={14} /> Markdown
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Safety Note */}
        <MedicalNote title="Clinical Safety Notice — Speech & Language Assessment">
          This language assessment is a clinical decision-support output only. It does not
          replace a formal neuropsychological evaluation or a diagnosis by a licensed
          healthcare professional. The transcript analysis must be interpreted by a
          qualified speech-language pathologist or neurologist before any clinical action is taken.
        </MedicalNote>

      </div>
    </div>
  );
}
