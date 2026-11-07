import { useMemo, useState } from "react";
import LoadingOverlay from "./components/LoadingOverlay.jsx";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import About from "./pages/About.jsx";
import BrainScan from "./pages/BrainScan.jsx";
import CombinedReport from "./pages/CombinedReport.jsx";
import Home from "./pages/Home.jsx";
import MyReports from "./pages/MyReports.jsx";
import SpeechLanguage from "./pages/SpeechLanguage.jsx";

function makePatientCaseId() {
  const now = new Date();
  const stamp = now.toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const suffix = Math.random().toString(16).slice(2, 6).toUpperCase();
  return `PAT-${stamp}-${suffix}`;
}

const initialReports = [
  {
    id: "brain-1",
    type: "Brain Scan",
    title: "Brain Imaging Analysis Report",
    date: "2026-06-25 18:40",
    status: "Ready",
  },
  {
    id: "speech-1",
    type: "Speech and Language",
    title: "Speech & Language Assessment Report",
    date: "2026-06-25 18:48",
    status: "Ready",
  },
  {
    id: "combined-1",
    type: "Combined Multimodal",
    title: "Integrated Patient Summary Report",
    date: "2026-06-25 18:55",
    status: "Ready",
  },
];

export default function App() {
  const [activePage, setActivePage] = useState("home");
  const [patientCaseId, setPatientCaseId] = useState(makePatientCaseId);
  const [brainResult, setBrainResult] = useState(null);
  const [speechResult, setSpeechResult] = useState(null);
  const [reports, setReports] = useState(initialReports);
  const [isBusy, setIsBusy] = useState(false);

  const pageProps = useMemo(
    () => ({
      patientCaseId,
      setPatientCaseId,
      brainResult,
      setBrainResult,
      speechResult,
      setSpeechResult,
      reports,
      setReports,
      setIsBusy,
    }),
    [patientCaseId, brainResult, speechResult, reports]
  );

  const handleNewCase = () => {
    setPatientCaseId(makePatientCaseId());
    setBrainResult(null);
    setSpeechResult(null);
  };

  const pages = {
    home:     <Home {...pageProps} navigate={setActivePage} />,
    brain:    <BrainScan {...pageProps} />,
    speech:   <SpeechLanguage {...pageProps} />,
    combined: <CombinedReport {...pageProps} />,
    reports:  <MyReports {...pageProps} />,
    about:    <About />,
  };

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="main-panel">
        <TopBar patientCaseId={patientCaseId} onNewCase={handleNewCase} />
        {pages[activePage] ?? pages.home}
      </main>
      {isBusy && (
        <LoadingOverlay
          message="Running Clinical Assessment..."
          sub="Analyzing data — this usually takes a few seconds."
        />
      )}
    </div>
  );
}
