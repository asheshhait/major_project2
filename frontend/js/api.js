const API_BASE = "/api";

async function readResponse(response, operation) {
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `${operation} failed (${response.status}).`);
  }
  return payload;
}

export async function getHealth() {
  const response = await fetch(`${API_BASE.replace("/api", "")}/health`);
  return readResponse(response, "Health check");
}

export async function uploadPenFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });
  return readResponse(response, "Upload");
}

export async function saveSample(sample) {
  const response = await fetch(`${API_BASE}/samples`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sample),
  });
  return readResponse(response, "Saving sample");
}

export async function saveBadSample(sample) {
  const response = await fetch(`${API_BASE}/samples/bad`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sample),
  });
  return readResponse(response, "Recording bad sample");
}

export async function trainModel() {
  const response = await fetch(`${API_BASE}/train`, { method: "POST" });
  return readResponse(response, "Training");
}

export async function getTrainStatus() {
  const response = await fetch(`${API_BASE}/train/status`);
  return readResponse(response, "Training status");
}

export async function predictWord(wordId, points) {
  const response = await fetch(`${API_BASE}/predict/${wordId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  return readResponse(response, "Prediction");
}

export async function saveWordCorrections(sessionId, words) {
  const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/words`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ words }),
  });
  return readResponse(response, "Saving box corrections");
}

export async function redetectWordBoxes(sessionId, strokes) {
  const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strokes }),
  });
  return readResponse(response, "Re-detecting word boxes");
}
