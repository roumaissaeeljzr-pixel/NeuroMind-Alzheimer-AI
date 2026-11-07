import {
  Activity,
  Brain,
  CheckCircle2,
  Download,
  FileImage,
  HelpCircle,
  Layers,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Tag,
  UploadCloud,
} from "lucide-react";
import { useState } from "react";
import MedicalNote from "../components/MedicalNote.jsx";
import ResultCard from "../components/ResultCard.jsx";
import UploadCard from "../components/UploadCard.jsx";

/* ── Mock Q&A ────────────────────────────────── */
const clinicalQA = [
  {
    q: "What does a cognitively normal profile mean for this patient?",
    a: "A cognitively normal profile indicates that the structural characteristics of this MRI scan are consistent with those typically observed in individuals without neurodegenerative changes. This is a decision-support output, not a clinical conclusion.",
  },
  {
    q: "Should this result be used to rule out Alzheimer's disease?",
    a: "No. This output cannot be used to rule out or confirm any diagnosis. A qualified neurologist or radiologist must review the original scan and interpret this result in the context of the patient's full clinical history.",
  },
  {
    q: "What is the significance of the Alzheimer's disease-compatible score (2.64%)?",
    a: "The 2.64% score reflects a low imaging pattern overlap with atypical neurodegenerative profiles in the reference dataset. Even a low score requires clinical review. It does not mean the patient is free of risk.",
  },
  {
    q: "What brain regions were highlighted in the visual explanation?",
    a: "The visual explanation highlights regions that contributed most to the assessment output, such as the hippocampal area and surrounding cortical structures. These highlights are analytical indicators, not confirmed pathological findings.",
  },
];

/* ─────────────────────────────────────────────
   BRAIN IMAGING ANALYSIS PAGE
   ───────────────────────────────────────────── */
export default function BrainScan({ patientCaseId, brainResult, setBrainResult, setReports, setIsBusy }) {
  const [fileName, setFileName] = useState("");
  const [qaOpen, setQaOpen] = useState(null);

  const handleAnalyze = () => {
    setIsBusy(true);
    window.setTimeout(() => {
      setBrainResult({
        prediction: "Cognitively Normal Profile",
        confidence: "97.36%",
        probCn: "97.36",
        probAd: "2.64",
      });
      setReports((prev) => [
        {
          id: `brain-${Date.now()}`,
          type: "Brain Scan",
          title: "Brain Imaging Analysis Report",
          date: new Date().toLocaleString(),
          status: "Ready",
        },
        ...prev,
      ]);
      setIsBusy(false);
    }, 900);
  };

  const result = brainResult || {
    prediction: "Cognitively Normal Profile",
    confidence: "97.36%",
    probCn: "97.36",
    probAd: "2.64",
  };

  const cnPct  = parseFloat(result.probCn);
  const adPct  = parseFloat(result.probAd);

  return (
    <div className="page-content">

      {/* ── Page header ── */}
      <div className="page-header">
        <div className="page-eyebrow">
          <div className="page-eyebrow-dot" />
          <span className="page-eyebrow-text">Brain Imaging Analysis</span>
        </div>
        <h1 className="page-title">Brain Imaging Analysis</h1>
        <p className="page-subtitle">
          Structural MRI assessment for clinical decision-support.
          Review imaging profile, confidence score, and visual explanation.
        </p>
      </div>

      <div className="stack">

        {/* ① Patient Case ID card */}
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
            <span className={`status-pill ${brainResult ? "status-ready" : "status-pending"}`}>
              {brainResult ? "Analysis Complete" : "Awaiting Upload"}
            </span>
            {brainResult && (
              <span className="badge badge-teal">
                <CheckCircle2 size={10} /> {result.confidence} Confidence
              </span>
            )}
          </div>
        </div>

        {/* ② MRI upload panel */}
        <div>
          <div className="step-header">
            <div className="step-num">1</div>
            <span className="step-title">Upload MRI Scan</span>
            <div className="step-sep" />
          </div>

          <div className="grid-2" style={{ alignItems: "start" }}>
            <UploadCard
              title="Upload Brain Imaging File"
              description="Select an anonymized structural MRI scan file"
              accept=".nii,.nii.gz"
              fileName={fileName}
              formats={[".nii", ".nii.gz"]}
              onChange={(e) => setFileName(e.target.files?.[0]?.name || "")}
            />

            <div className="card">
              <div className="card-body">
                <div className="info-box" style={{ marginBottom: 14 }}>
                  <div className="info-box-title">Accepted File Format</div>
                  <div className="info-box-body">
                    NIfTI format (.nii or .nii.gz). The file must be anonymized — all personal
                    identifiers must be removed before upload. Do not upload raw DICOM files.
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    "File must be anonymized before upload",
                    "Supported: T1-weighted structural MRI",
                    "Maximum recommended size: 500 MB",
                    "Single-scan files only (no 4D sequences)",
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

        {/* ③ Analyze button */}
        <div>
          <div className="step-header">
            <div className="step-num">2</div>
            <span className="step-title">Run Brain Imaging Analysis</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
              <div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
                  Analyze Brain Imaging
                </div>
                <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", lineHeight: 1.5, maxWidth: 520 }}>
                  {fileName
                    ? `Ready to analyze: ${fileName}. The assessment will produce a brain imaging profile with a confidence score and visual explanation.`
                    : "Upload an MRI scan above before running the analysis. The output will include a clinical decision-support profile and confidence score."}
                </p>
              </div>
              <button
                type="button"
                className="btn btn-primary btn-lg"
                onClick={handleAnalyze}
                disabled={!fileName}
                style={{ flexShrink: 0 }}
              >
                <ScanLine size={16} />
                Analyze Brain Imaging
              </button>
            </div>
          </div>
        </div>

        {/* ④ Brain imaging profile card */}
        <div>
          <div className="step-header">
            <div className="step-num">3</div>
            <span className="step-title">Brain Imaging Profile</span>
            <div className="step-sep" />
          </div>

          <ResultCard
            title="Brain Imaging Profile"
            result={result.prediction}
            confidence={result.confidence}
            description="The structural imaging assessment did not identify a pattern strongly associated with neurodegenerative change. The brain imaging profile is consistent with a cognitively normal reference group. This output is for clinical decision-support only."
            tone="good"
          />
        </div>

        {/* ⑤ Confidence score  ⑥ CN score  ⑦ AD score */}
        <div>
          <div className="step-header">
            <div className="step-num">4</div>
            <span className="step-title">Confidence Score & Imaging Profile Scores</span>
            <div className="step-sep" />
          </div>

          <div className="score-grid">
            {/* CN card */}
            <div className="score-card cn">
              <div className="score-card-icon">
                <CheckCircle2 />
              </div>
              <div className="score-card-label">
                Cognitively Normal-Compatible Profile Score
              </div>
              <div className="score-card-value">{result.probCn}%</div>
              <div className="score-card-bar-track">
                <div className="score-card-bar-fill" style={{ width: `${cnPct}%` }} />
              </div>
              <div className="score-card-sub">
                High overlap with imaging patterns from the cognitively normal reference group.
                This score does not confirm the absence of disease.
              </div>
            </div>

            {/* AD card */}
            <div className="score-card ad">
              <div className="score-card-icon">
                <ShieldAlert />
              </div>
              <div className="score-card-label">
                Alzheimer's Disease-Compatible Imaging Profile Score
              </div>
              <div className="score-card-value">{result.probAd}%</div>
              <div className="score-card-bar-track">
                <div className="score-card-bar-fill" style={{ width: `${adPct}%` }} />
              </div>
              <div className="score-card-sub">
                Low overlap with atypical neurodegenerative imaging patterns.
                Clinical review is still required regardless of this score.
              </div>
            </div>
          </div>

          {/* Overall confidence note */}
          <div className="card-flat" style={{ padding: "14px 18px", marginTop: 12, display: "flex", alignItems: "center", gap: 10 }}>
            <Sparkles size={15} style={{ color: "var(--teal-500)", flexShrink: 0 }} />
            <span style={{ fontSize: "0.80rem", color: "var(--text-secondary)" }}>
              <strong>Overall Confidence Score: {result.confidence}</strong> — The assessment confidence
              reflects the system's certainty in the pattern classification, not a probability of disease.
            </span>
          </div>
        </div>

        {/* ⑧ Visual Explanation */}
        <div>
          <div className="step-header">
            <div className="step-num">5</div>
            <span className="step-title">Visual Explanation</span>
            <div className="step-sep" />
          </div>

          <div className="grid-2" style={{ alignItems: "start" }}>
            <div className="visual-panel" style={{ minHeight: 240 }}>
              <div className="visual-panel-icon">
                <FileImage size={42} />
              </div>
              <div className="visual-panel-title">Brain Imaging Visual Explanation</div>
              <div className="visual-panel-sub" style={{ maxWidth: 280 }}>
                A color-coded activation map will appear here, highlighting the brain regions
                that most influenced the assessment output. Warmer colors indicate higher influence.
              </div>
            </div>

            <div className="card">
              <div className="card-body">
                <div className="row-sm" style={{ marginBottom: 14 }}>
                  <div className="card-icon-wrap icon-mint">
                    <Layers size={17} />
                  </div>
                  <div>
                    <div className="card-label">Clinical Decision-Support Explanation</div>
                    <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.92rem", color: "var(--text-primary)" }}>
                      How to Read the Visual Explanation
                    </div>
                  </div>
                </div>

                <div className="prose">
                  <p>
                    The visual explanation uses a color-coded overlay applied to the original MRI scan.
                    Brain regions that contributed most to the assessment result are highlighted in
                    warmer tones (orange → red), while less influential regions appear in cooler tones.
                  </p>
                  <p style={{ marginTop: 10 }}>
                    <strong>Regions typically assessed:</strong> hippocampal structures, entorhinal
                    cortex, parietal regions, and prefrontal areas associated with memory and
                    executive function.
                  </p>
                  <p style={{ marginTop: 10 }}>
                    <strong>Important:</strong> The highlighted regions represent analytical decision
                    patterns only. They are not a confirmed pathological finding and must be reviewed
                    by a qualified radiologist or neurologist.
                  </p>
                </div>

                <div style={{
                  marginTop: 16, display: "flex", gap: 12, flexWrap: "wrap",
                }}>
                  {[
                    { color: "#ef4444", label: "High influence" },
                    { color: "#f97316", label: "Moderate influence" },
                    { color: "#3b82f6", label: "Low influence" },
                    { color: "#94a3b8", label: "Not assessed" },
                  ].map(({ color, label }) => (
                    <div key={label} className="row-sm" style={{ gap: 6 }}>
                      <div style={{ width: 12, height: 12, borderRadius: 3, background: color, flexShrink: 0 }} />
                      <span style={{ fontSize: "0.70rem", color: "var(--text-muted)" }}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ⑨ Clinical Q&A */}
        <div>
          <div className="step-header">
            <div className="step-num">6</div>
            <span className="step-title">Questions About This Result</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div className="row-sm" style={{ marginBottom: 16 }}>
                <div className="card-icon-wrap icon-mint">
                  <HelpCircle size={17} />
                </div>
                <div>
                  <div className="card-label">Frequently Asked Clinical Questions</div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.92rem", color: "var(--text-primary)" }}>
                    Clinical Decision-Support Q&A
                  </div>
                </div>
              </div>

              <div className="qa-list">
                {clinicalQA.map((item, i) => (
                  <div key={i} className="qa-item">
                    <button
                      type="button"
                      className="qa-question"
                      style={{ background: "none", border: "none", cursor: "pointer", width: "100%", textAlign: "left", padding: 0 }}
                      onClick={() => setQaOpen(qaOpen === i ? null : i)}
                    >
                      <span className="qa-q-icon">Q</span>
                      <span className="qa-q-text">{item.q}</span>
                      <Tag size={13} style={{ color: "var(--text-faint)", flexShrink: 0, marginLeft: "auto", transform: qaOpen === i ? "rotate(90deg)" : "none", transition: "transform 200ms" }} />
                    </button>
                    {(qaOpen === i || true) && (
                      <div className="qa-answer" style={{ marginTop: 8 }}>
                        {item.a}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ⑩ Report download */}
        <div>
          <div className="step-header">
            <div className="step-num">7</div>
            <span className="step-title">Download Report</span>
            <div className="step-sep" />
          </div>

          <div className="card">
            <div className="card-body">
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
                <div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.94rem", color: "var(--text-primary)", marginBottom: 4 }}>
                    Brain Imaging Analysis Report
                  </div>
                  <p style={{ fontSize: "0.80rem", color: "var(--text-muted)", lineHeight: 1.5, maxWidth: 480 }}>
                    The report includes the brain imaging profile, confidence score, visual explanation
                    summary, and all clinical safety notices. Suitable for referral documentation.
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
        </div>

        {/* Clinical Safety Note */}
        <MedicalNote title="Clinical Safety Notice — Brain Imaging Analysis">
          This brain imaging assessment is a clinical decision-support tool only. It does not
          constitute a medical diagnosis. All outputs must be reviewed and interpreted by a
          qualified radiologist or neurologist in the context of the patient's full clinical
          history. Do not use this output as the sole basis for any clinical decision.
        </MedicalNote>

      </div>
    </div>
  );
}
