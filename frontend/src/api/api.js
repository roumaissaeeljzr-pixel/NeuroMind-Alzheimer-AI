const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.blob();
}

export function createPatientCase() {
  return request("/patient-cases", { method: "POST" });
}

export function getPatientCase(patientCaseId) {
  return request(`/patient-cases/${encodeURIComponent(patientCaseId)}`);
}

export function analyzeBrainScan(formData) {
  return request("/brain/analyze", { method: "POST", body: formData });
}

export function generateBrainVisualExplanation(patientCaseId) {
  return request(`/brain/${encodeURIComponent(patientCaseId)}/visual-explanation`, {
    method: "POST",
  });
}

export function analyzeSpeech(formData) {
  return request("/speech/analyze", { method: "POST", body: formData });
}

export function getReports(patientCaseId) {
  return request(`/reports?patient_case_id=${encodeURIComponent(patientCaseId)}`);
}

export function generateCombinedReport(patientCaseId) {
  return request(`/reports/combined/${encodeURIComponent(patientCaseId)}`, {
    method: "POST",
  });
}

export function downloadReport(path) {
  return request(`/reports/download?path=${encodeURIComponent(path)}`);
}
