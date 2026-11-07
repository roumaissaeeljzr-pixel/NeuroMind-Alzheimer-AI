import { Activity, Brain, Database, MessageSquareText, Server, ShieldCheck, Users } from "lucide-react";
import MedicalNote from "../components/MedicalNote.jsx";

const platform = [
  {
    icon: Activity,
    label: "Clinical Dashboard",
    desc: "A professional dashboard for clinicians to review assessment outputs, reports, and patient case history.",
    tone: "teal",
  },
  {
    icon: Brain,
    label: "Brain Imaging Analysis",
    desc: "Structural brain scan pattern assessment providing an imaging profile with a confidence score.",
    tone: "teal",
  },
  {
    icon: MessageSquareText,
    label: "Speech & Language Assessment",
    desc: "Picture-description transcript analysis extracting language markers and generating an assessment profile.",
    tone: "teal",
  },
  {
    icon: Server,
    label: "Clinical Integration Layer",
    desc: "A secure service layer that orchestrates assessment requests and manages report generation.",
    tone: "blue",
  },
  {
    icon: Database,
    label: "Patient Record Store",
    desc: "Structured storage for patient case identifiers, assessment history, and report file references.",
    tone: "blue",
  },
  {
    icon: Users,
    label: "Clinician-Centred Design",
    desc: "All outputs are presented in clinician-friendly language. No technical terminology is exposed in the interface.",
    tone: "blue",
  },
];

const safeguards = [
  "Brain imaging and language assessments are conducted independently — results are never automatically merged.",
  "All outputs are labeled as decision-support only. No automated diagnostic conclusion is generated.",
  "Every report page includes a clinical safety notice reinforcing the need for professional review.",
  "Patient case identifiers are generated locally and do not contain personal health information.",
  "Assessment confidence scores are always displayed to support transparent clinical interpretation.",
];

export default function About() {
  return (
    <div className="page-content">
      <div className="page-header">
        <div className="page-eyebrow">
          <div className="page-eyebrow-dot" />
          <span className="page-eyebrow-text">Platform Overview</span>
        </div>
        <h1 className="page-title">About NeuroMind Clinical Platform</h1>
        <p className="page-subtitle">
          A professional clinical decision-support platform for independent brain imaging
          and speech-language assessments.
        </p>
      </div>

      <div className="stack">
        {/* Overview */}
        <div className="card">
          <div className="card-body-lg">
            <div className="row-sm" style={{ marginBottom: 14 }}>
              <div className="card-icon-wrap icon-teal">
                <Activity size={19} />
              </div>
              <div>
                <div style={{ fontSize: "0.70rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.09em" }}>
                  Platform Mission
                </div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.98rem", color: "var(--text-primary)" }}>
                  Clinical Decision-Support for Cognitive Assessment
                </div>
              </div>
            </div>

            <div className="prose">
              <p>
                NeuroMind is a professional clinical decision-support platform designed to assist
                healthcare professionals in reviewing brain imaging and speech-language assessment
                outputs for patients being evaluated for cognitive decline.
              </p>
              <p>
                The platform organises independently generated assessment results, provides
                confidence scores, and produces structured reports suitable for clinical review
                and referral documentation. It is designed for use by neurologists, geriatricians,
                speech-language pathologists, and primary care physicians.
              </p>
              <p>
                <strong>This platform does not provide medical diagnoses.</strong> All outputs
                are clearly labeled as clinical decision-support and must be interpreted by a
                qualified healthcare professional.
              </p>
            </div>
          </div>
        </div>

        {/* Platform components */}
        <div>
          <div className="section-head">
            <span className="section-head-text">Platform Components</span>
            <div className="section-head-line" />
          </div>

          <div className="arch-grid">
            {platform.map(({ icon: Icon, label, desc, tone }) => (
              <div key={label} className="arch-item">
                <div className="row-sm" style={{ marginBottom: 8 }}>
                  <div className={`card-icon-wrap ${tone === "teal" ? "icon-mint" : "icon-blue"}`} style={{ width: 32, height: 32 }}>
                    <Icon size={15} />
                  </div>
                  <div className="arch-item-label">{label}</div>
                </div>
                <div className="arch-item-desc">{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Clinical safeguards */}
        <div className="card">
          <div className="card-body">
            <div className="row-sm" style={{ marginBottom: 14 }}>
              <div className="card-icon-wrap icon-mint">
                <ShieldCheck size={17} />
              </div>
              <span style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "0.95rem", color: "var(--text-primary)" }}>
                Clinical Safety Safeguards
              </span>
            </div>

            <div className="stack-xs">
              {safeguards.map((item, i) => (
                <div key={i} style={{
                  display: "flex", gap: 12, alignItems: "flex-start",
                  padding: "10px 0",
                  borderBottom: i < safeguards.length - 1 ? "1px solid var(--border-light)" : "none",
                }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%",
                    background: "var(--success-bg)", border: "1px solid var(--success-100)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexShrink: 0, marginTop: 1,
                  }}>
                    <ShieldCheck size={11} style={{ color: "var(--success-600)" }} />
                  </div>
                  <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                    {item}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Limitations */}
        <div className="info-box">
          <div className="info-box-title">Important Clinical Limitation</div>
          <div className="info-box-body">
            The brain imaging and language assessments are trained on independent datasets
            that are not matched at the individual patient level. Consequently, results from
            both assessments cannot be automatically combined into a single integrated score.
            The Integrated Patient Summary presents both outputs side-by-side to support
            — not replace — clinical interpretation.
          </div>
        </div>

        <MedicalNote title="Clinical Safety Notice — Platform Scope">
          NeuroMind is a clinical decision-support platform only. All assessment outputs
          require review and interpretation by a qualified healthcare professional before
          any clinical action is taken. This platform does not generate medical diagnoses
          and should not be used as the sole basis for clinical decision-making.
        </MedicalNote>
      </div>
    </div>
  );
}
