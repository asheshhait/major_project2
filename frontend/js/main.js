import {
  getHealth,
  getTrainStatus,
  uploadPenFile,
  trainModel,
  saveSample,
  saveBadSample,
  predictWord,
  saveWordCorrections,
  redetectWordBoxes,
} from "./api.js";
import { PlotCanvas } from "./plot.js";
import { splitWordIntoChars } from "./words.js";

const canvas = document.querySelector("#drawing-canvas");
const fileInput = document.querySelector("#file-input");
const uploadButton = document.querySelector("#upload-button");
const trainButton = document.querySelector("#train-button");
const showPredictionsButton = document.querySelector("#show-predictions-button");
const fullScriptView = document.querySelector("#full-script-view");
const fullScriptStatus = document.querySelector("#full-script-status");
const fullScriptPlotMount = document.querySelector("#full-script-plot-mount");
const fullScriptWords = document.querySelector("#full-script-words");
const backToViewerButton = document.querySelector("#back-to-viewer-button");
const plotCanvasHome = canvas.parentElement;
const themeToggle = document.querySelector("#theme-toggle");
const correctWordInput = document.querySelector("#correct-word");
const charChips = document.querySelector("#char-chips");
const addCharInput = document.querySelector("#add-char-input");
const addCharButton = document.querySelector("#add-char-button");
const uploadStatus = document.querySelector("#upload-status");
const previousWordButton = document.querySelector("#previous-word");
const nextWordButton = document.querySelector("#next-word");
const wordCounter = document.querySelector("#word-counter");
const saveNextButton = document.querySelector("#save-next-button");
const skipButton = document.querySelector("#skip-button");
const badSampleButton = document.querySelector("#bad-sample-button");
const actionStatus = document.querySelector("#action-status");
const predictionText = document.querySelector("#prediction-text");
const predictionEngine = document.querySelector("#prediction-engine");
const confidenceBar = document.querySelector(".confidence-bar span");
const zoomInButton = document.querySelector("#zoom-in");
const zoomOutButton = document.querySelector("#zoom-out");
const zoomResetButton = document.querySelector("#zoom-reset");
const fullPageViewButton = document.querySelector("#full-page-view");
const selectedWordViewButton = document.querySelector("#selected-word-view");
const boxEditingHelp = document.querySelector("#box-editing-help");
const boxEditingStatus = document.querySelector("#box-editing-status");
const boxSelectModeButton = document.querySelector("#box-select-mode");
const boxRedetectActionButton = document.querySelector("#box-redetect-action");
const boxCreateModeButton = document.querySelector("#box-create-mode");
const boxDeleteActionButton = document.querySelector("#box-delete-action");
const boxCancelActionButton = document.querySelector("#box-cancel-action");

const plot = new PlotCanvas(canvas);
zoomInButton.addEventListener("click", () => plot.zoomBy(1.25));
zoomOutButton.addEventListener("click", () => plot.zoomBy(0.8));
zoomResetButton.addEventListener("click", () => plot.resetView());
fullPageViewButton.addEventListener("click", () => {
  if (uploadedSession) setBoxEditMode("select");
  plot.setViewMode("page");
  fullPageViewButton.classList.add("active");
  selectedWordViewButton.classList.remove("active");
});
selectedWordViewButton.addEventListener("click", () => {
  if (uploadedSession) setBoxEditMode("select");
  plot.setViewMode("word");
  selectedWordViewButton.classList.add("active");
  fullPageViewButton.classList.remove("active");
});
let wordCharacters = [];
let uploadedSession = null;
let selectedWordIndex = 0;
let modelIsTrained = false;
let reviewCompleted = false;
let boxEditMode = "select";
let selectedBoxIndices = [];
let correctionsInProgress = false;
let previousFullScriptPlotMode = null;

function renderCharacters() {
  charChips.replaceChildren();

  if (wordCharacters.length === 0) {
    const emptyMessage = document.createElement("span");
    emptyMessage.className = "segmentation-empty";
    emptyMessage.textContent = "Type a word to see its characters here.";
    charChips.append(emptyMessage);
    return;
  }

  wordCharacters.forEach((character, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";

    const text = document.createElement("span");
    text.textContent = character;
    chip.append(text);

    const removeButton = document.createElement("button");
    removeButton.className = "chip-remove";
    removeButton.type = "button";
    removeButton.textContent = "×";
    removeButton.setAttribute("aria-label", `Remove character ${character}`);
    removeButton.addEventListener("click", () => {
      wordCharacters.splice(index, 1);
      renderCharacters();
    });
    chip.append(removeButton);
    charChips.append(chip);
  });
}

uploadButton.addEventListener("click", () => fileInput.click());
showPredictionsButton.addEventListener("click", predictFullScript);
backToViewerButton.addEventListener("click", () => showFullScriptView(false));

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

correctWordInput.addEventListener("input", () => {
  wordCharacters = splitWordIntoChars(correctWordInput.value);
  renderCharacters();
});

addCharButton.addEventListener("click", () => {
  const characters = splitWordIntoChars(addCharInput.value);
  if (characters.length === 0) return;
  wordCharacters.push(...characters);
  addCharInput.value = "";
  renderCharacters();
  addCharInput.focus();
});

addCharInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addCharButton.click();
  }
});

function selectWord(index) {
  if (!uploadedSession?.words.length) return;
  reviewCompleted = false;
  selectedWordIndex = Math.max(0, Math.min(index, uploadedSession.words.length - 1));
  plot.selectWord(selectedWordIndex);
  selectedBoxIndices = [selectedWordIndex];
  plot.setSelectedWordIndices(selectedBoxIndices);
  updateBoxActionButtons();
  wordCounter.textContent = `Word ${selectedWordIndex + 1} / ${uploadedSession.words.length}`;
  previousWordButton.disabled = selectedWordIndex === 0;
  nextWordButton.disabled = selectedWordIndex === uploadedSession.words.length - 1;
  const canLabel = uploadedSession.words.length > 0;
  saveNextButton.disabled = !canLabel;
  skipButton.disabled = !canLabel;
  badSampleButton.disabled = !canLabel;
  correctWordInput.value = "";
  wordCharacters = [];
  renderCharacters();
  predictionText.textContent = "Unknown";
  predictionEngine.textContent = modelIsTrained ? "Checking template model…" : "Model not trained";
  confidenceBar.style.width = "0%";
  if (modelIsTrained) void predictSelectedWord();
}

function selectedWordPoints() {
  const word = uploadedSession?.words[selectedWordIndex];
  return word ? pointsForWord(word, uploadedSession) : [];
}

function pointsForWord(word, session) {
  const strokeIds = new Set(word.stroke_ids);
  return session.strokes
    .filter((stroke) => strokeIds.has(stroke.id))
    .flatMap((stroke) => stroke.points);
}

function showFullScriptView(show) {
  const layout = document.querySelector(".layout");
  if (show && fullScriptView.hidden) {
    previousFullScriptPlotMode = plot.viewMode;
    plot.setViewMode("page");
    fullScriptPlotMount.append(canvas);
  } else if (!show && !fullScriptView.hidden) {
    plotCanvasHome.insertBefore(canvas, plotCanvasHome.querySelector(".box-editing"));
    if (previousFullScriptPlotMode) plot.setViewMode(previousFullScriptPlotMode);
    previousFullScriptPlotMode = null;
  }
  fullScriptView.hidden = !show;
  layout.hidden = show;
  requestAnimationFrame(() => plot.draw());
}

function renderFullScript(predictions) {
  fullScriptWords.replaceChildren();
  const orderedPredictions = [...predictions].sort(
    (left, right) => left.word.line_index - right.word.line_index
      || left.word.word_index - right.word.word_index,
  );
  orderedPredictions.forEach(({ prediction }) => {
    const item = document.createElement("li");
    const label = document.createElement("strong");
    label.textContent = prediction.text || "Unknown";
    const confidence = document.createElement("span");
    confidence.textContent = `— Confidence: ${Math.round((prediction.confidence || 0) * 100)}%`;
    item.append(label, confidence);
    fullScriptWords.append(item);
  });
}

async function predictFullScript() {
  const session = uploadedSession;
  if (!session?.words.length) return;
  showFullScriptView(true);
  fullScriptWords.replaceChildren();
  fullScriptStatus.textContent = `Predicting all ${session.words.length} detected words…`;
  fullScriptStatus.classList.remove("error");
  showPredictionsButton.disabled = true;
  const predictions = [];
  try {
    for (let index = 0; index < session.words.length; index += 1) {
      const word = session.words[index];
      const prediction = await predictWord(word.id, pointsForWord(word, session));
      predictions.push({ word, prediction });
      if (uploadedSession !== session) return;
      fullScriptStatus.textContent = `Predicted ${index + 1} of ${session.words.length} words…`;
    }
    renderFullScript(predictions);
    fullScriptStatus.textContent = `Finished predicting ${predictions.length} words. Unknown results mean the model could not recognize that word.`;
  } catch (error) {
    fullScriptStatus.textContent = error instanceof Error
      ? error.message
      : "Could not predict the full script.";
    fullScriptStatus.classList.add("error");
  } finally {
    showPredictionsButton.disabled = !uploadedSession?.words.length;
  }
}

async function predictSelectedWord() {
  const word = uploadedSession?.words[selectedWordIndex];
  if (!word) return;
  try {
    const prediction = await predictWord(word.id, selectedWordPoints());
    if (uploadedSession?.words[selectedWordIndex]?.id !== word.id) return;
    predictionText.textContent = prediction.text;
    predictionEngine.textContent = prediction.engine;
    confidenceBar.style.width = `${Math.round(prediction.confidence * 100)}%`;
    wordCharacters = prediction.text && prediction.text !== "Unknown"
      ? Array.isArray(prediction.chars) && prediction.chars.length
        ? prediction.chars
        : splitWordIntoChars(prediction.text)
      : [];
    renderCharacters();
  } catch (error) {
    predictionEngine.textContent = error instanceof Error ? error.message : "Prediction failed";
  }
}

function advanceWord(message) {
  if (selectedWordIndex < uploadedSession.words.length - 1) {
    selectWord(selectedWordIndex + 1);
    actionStatus.textContent = message;
    actionStatus.classList.remove("error");
  } else {
    reviewCompleted = true;
    actionStatus.textContent = `${message} All words in this upload have been reviewed.`;
    actionStatus.classList.remove("error");
    saveNextButton.disabled = true;
    skipButton.disabled = true;
    badSampleButton.disabled = true;
  }
}

function updateBoxActionButtons() {
  const hasSession = Boolean(uploadedSession?.words);
  const canEdit = hasSession && !correctionsInProgress;
  boxSelectModeButton.disabled = !canEdit;
  boxRedetectActionButton.disabled = !canEdit;
  boxCreateModeButton.disabled = !canEdit;
  boxDeleteActionButton.disabled = !canEdit || selectedBoxIndices.length !== 1;
}

function setBoxEditMode(mode) {
  if (!uploadedSession) {
    boxEditingStatus.textContent = "Upload a handwriting file before editing boxes.";
    boxEditingStatus.classList.add("error");
    return;
  }
  boxEditMode = mode;
  plot.setEditMode(mode);
  fullPageViewButton.classList.add("active");
  selectedWordViewButton.classList.remove("active");
  for (const [button, buttonMode] of [
    [boxSelectModeButton, "select"],
    [boxCreateModeButton, "create"],
  ]) {
    button.classList.toggle("active", mode === buttonMode);
  }
  boxEditingHelp.textContent = mode === "create"
    ? "Drag a rectangle around the word's strokes. They will be reassigned from any box they currently belong to."
    : "Click a box to choose that word. Drag empty plot space to pan.";
  boxEditingStatus.textContent = "";
  boxEditingStatus.classList.remove("error");
  updateBoxActionButtons();
}

function renumberWords(words) {
  const lineCounts = new Map();
  return [...words]
    .sort((left, right) => left.line_index - right.line_index || left.bbox.x - right.bbox.x)
    .map((word) => {
      const wordIndex = lineCounts.get(word.line_index) || 0;
      lineCounts.set(word.line_index, wordIndex + 1);
      return { ...word, word_index: wordIndex };
    });
}

function boxFromStrokes(strokeIds, lineIndex, id = null) {
  const selectedIds = new Set(strokeIds);
  const points = uploadedSession.strokes
    .filter((stroke) => selectedIds.has(stroke.id))
    .flatMap((stroke) => stroke.points);
  if (points.length === 0) return null;
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const padding = Math.max(maxX - minX, maxY - minY, 1) * 0.08;
  return {
    id,
    line_index: lineIndex,
    word_index: 0,
    stroke_ids: [...selectedIds].sort((a, b) => a - b),
    bbox: {
      x: minX - padding,
      y: minY - padding,
      width: Math.max(maxX - minX + padding * 2, 1),
      height: Math.max(maxY - minY + padding * 2, 1),
    },
  };
}

async function applyWordCorrections(words, nextSelectedIndex, successMessage) {
  if (correctionsInProgress) return;
  correctionsInProgress = true;
  updateBoxActionButtons();
  boxEditingStatus.textContent = "Saving box corrections…";
  boxEditingStatus.classList.remove("error");
  try {
    const saved = await saveWordCorrections(
      uploadedSession.session_id,
      renumberWords(words),
    );
    uploadedSession.words = saved.words;
    selectedWordIndex = Math.max(0, Math.min(nextSelectedIndex, saved.words.length - 1));
    selectedBoxIndices = saved.words.length ? [selectedWordIndex] : [];
    plot.setSession(uploadedSession);
    plot.setEditMode(boxEditMode);
    if (saved.words.length) {
      selectWord(selectedWordIndex);
      plot.setSelectedWordIndices(selectedBoxIndices);
    } else {
      wordCounter.textContent = "Word 0 / 0";
      saveNextButton.disabled = true;
      skipButton.disabled = true;
      badSampleButton.disabled = true;
    }
    boxEditingStatus.textContent = successMessage;
    boxEditingStatus.classList.remove("error");
  } catch (error) {
    boxEditingStatus.textContent = error instanceof Error
      ? error.message
      : "Could not save word box corrections.";
    boxEditingStatus.classList.add("error");
  } finally {
    correctionsInProgress = false;
    updateBoxActionButtons();
  }
}

function handleWordClick(index) {
  selectedBoxIndices = [index];
  selectWord(index);
  boxEditingStatus.textContent = `Selected word box ${index + 1}.`;
}

async function createMissingBox(selection) {
  const strokeIds = uploadedSession.strokes
    .filter((stroke) => {
      const centerX = stroke.points.reduce((sum, point) => sum + point.x, 0) / stroke.points.length;
      const centerY = stroke.points.reduce((sum, point) => sum + point.y, 0) / stroke.points.length;
      return centerX >= selection.x && centerX <= selection.x + selection.width
        && centerY >= selection.y && centerY <= selection.y + selection.height;
    })
    .map((stroke) => stroke.id);
  if (!strokeIds.length) {
    boxEditingStatus.textContent = "No unassigned pen strokes found inside that rectangle.";
    boxEditingStatus.classList.add("error");
    return;
  }
  const centerY = (selection.y + selection.y + selection.height) / 2;
  const lineCenters = new Map();
  for (const word of uploadedSession.words) {
    const center = word.bbox.y + word.bbox.height / 2;
    const existing = lineCenters.get(word.line_index) || [];
    existing.push(center);
    lineCenters.set(word.line_index, existing);
  }
  const lineIndex = [...lineCenters.entries()]
    .map(([line, centers]) => [line, centers.reduce((sum, value) => sum + value, 0) / centers.length])
    .sort((left, right) => Math.abs(left[1] - centerY) - Math.abs(right[1] - centerY))[0]?.[0] ?? 0;
  const newBox = boxFromStrokes(strokeIds, lineIndex);
  const selectedIds = new Set(strokeIds);
  const words = uploadedSession.words
    .map((word) => {
      const remainingIds = word.stroke_ids.filter((strokeId) => !selectedIds.has(strokeId));
      return remainingIds.length ? boxFromStrokes(remainingIds, word.line_index, word.id) : null;
    })
    .filter(Boolean);
  words.push(newBox);
  await applyWordCorrections(words, words.length - 1, "Missing word box created and saved.");
}

async function deleteSelectedBox() {
  if (selectedBoxIndices.length !== 1) return;
  const index = selectedBoxIndices[0];
  const words = uploadedSession.words.filter((_, wordIndex) => wordIndex !== index);
  await applyWordCorrections(words, Math.max(index - 1, 0), "Word box deleted. Its strokes are now available to draw into a new box.");
}

async function redetectCurrentWordBoxes() {
  if (!uploadedSession || correctionsInProgress) return;
  correctionsInProgress = true;
  updateBoxActionButtons();
  boxEditingStatus.textContent = "Re-detecting word boxes from this upload…";
  boxEditingStatus.classList.remove("error");
  try {
    const strokes = uploadedSession.strokes.map((stroke) => ({
      id: stroke.id,
      x_min: stroke.bbox.x,
      x_max: stroke.bbox.x + stroke.bbox.width,
      y_min: stroke.bbox.y,
      y_max: stroke.bbox.y + stroke.bbox.height,
      width: stroke.width,
      height: stroke.height,
      points: stroke.points,
    }));
    const result = await redetectWordBoxes(uploadedSession.session_id, strokes);
    uploadedSession.words = result.words;
    selectedWordIndex = 0;
    selectedBoxIndices = result.words.length ? [0] : [];
    reviewCompleted = false;
    plot.setSession(uploadedSession);
    plot.setEditMode("select");
    setBoxEditMode("select");
    if (result.words.length) {
      selectWord(0);
      plot.setSelectedWordIndices(selectedBoxIndices);
      boxEditingStatus.textContent = `Re-detected ${result.words.length} word boxes. Check the boundaries before labeling.`;
      boxEditingStatus.classList.remove("error");
    } else {
      boxEditingStatus.textContent = "No word boxes were detected in this upload.";
      boxEditingStatus.classList.add("error");
    }
  } catch (error) {
    boxEditingStatus.textContent = error instanceof Error
      ? error.message
      : "Could not re-detect word boxes.";
    boxEditingStatus.classList.add("error");
  } finally {
    correctionsInProgress = false;
    updateBoxActionButtons();
  }
}

plot.setCallbacks({
  onWordClick: handleWordClick,
  onCreateBox: createMissingBox,
});

boxSelectModeButton.addEventListener("click", () => setBoxEditMode("select"));
boxRedetectActionButton.addEventListener("click", redetectCurrentWordBoxes);
boxCreateModeButton.addEventListener("click", () => setBoxEditMode("create"));
boxDeleteActionButton.addEventListener("click", deleteSelectedBox);
boxCancelActionButton.addEventListener("click", () => {
  selectedBoxIndices = uploadedSession?.words.length ? [selectedWordIndex] : [];
  plot.setSelectedWordIndices(selectedBoxIndices);
  setBoxEditMode("select");
});

previousWordButton.addEventListener("click", () => selectWord(selectedWordIndex - 1));
nextWordButton.addEventListener("click", () => selectWord(selectedWordIndex + 1));
skipButton.addEventListener("click", () => advanceWord("Skipped without saving."));

saveNextButton.addEventListener("click", async () => {
  const word = uploadedSession?.words[selectedWordIndex];
  const text = correctWordInput.value.trim();
  if (!word) return;
  if (!text) {
    actionStatus.textContent = "Enter the correct Bangla word before saving.";
    actionStatus.classList.add("error");
    correctWordInput.focus();
    return;
  }
  if (wordCharacters.length === 0) {
    actionStatus.textContent = "Add at least one character in Segmentation before saving.";
    actionStatus.classList.add("error");
    return;
  }

  saveNextButton.disabled = true;
  actionStatus.textContent = "Saving labeled sample…";
  actionStatus.classList.remove("error");
  try {
    await saveSample({
      session_id: uploadedSession.session_id,
      word_id: word.id,
      text,
      chars: wordCharacters,
      points: selectedWordPoints(),
    });
    advanceWord(`Saved “${text}”.`);
  } catch (error) {
    actionStatus.textContent = error instanceof Error ? error.message : "Could not save this sample.";
    actionStatus.classList.add("error");
  } finally {
    saveNextButton.disabled = reviewCompleted || !uploadedSession?.words.length;
  }
});

badSampleButton.addEventListener("click", async () => {
  const word = uploadedSession?.words[selectedWordIndex];
  if (!word) return;
  badSampleButton.disabled = true;
  actionStatus.textContent = "Recording bad sample…";
  actionStatus.classList.remove("error");
  try {
    await saveBadSample({
      session_id: uploadedSession.session_id,
      word_id: word.id,
      points: selectedWordPoints(),
    });
    advanceWord("Bad sample recorded and excluded from training.");
  } catch (error) {
    actionStatus.textContent = error instanceof Error ? error.message : "Could not record this sample.";
    actionStatus.classList.add("error");
  } finally {
    badSampleButton.disabled = reviewCompleted || !uploadedSession?.words.length;
  }
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  showFullScriptView(false);
  uploadStatus.classList.remove("error");
  uploadStatus.textContent = `Reading ${file.name}…`;
  uploadButton.disabled = true;
  try {
    uploadedSession = await uploadPenFile(file);
    showFullScriptView(false);
    showPredictionsButton.disabled = !uploadedSession.words.length;
    plot.setSession(uploadedSession);
    selectedWordViewButton.classList.add("active");
    fullPageViewButton.classList.remove("active");
    selectWord(0);
    actionStatus.textContent = "";
    const { valid_rows, skipped_rows, stroke_count, word_count } = uploadedSession.stats;
    uploadStatus.textContent = uploadedSession.stats.restored
      ? `Restored your saved boxes for ${file.name}: ${word_count} words, ${stroke_count} strokes.`
      : `Loaded ${file.name}: ${word_count} word boxes, ${stroke_count} strokes, ${valid_rows} points` +
        (skipped_rows ? ` (${skipped_rows} rows skipped).` : ".");
  } catch (error) {
    uploadedSession = null;
    showPredictionsButton.disabled = true;
    plot.setSession(null);
    updateBoxActionButtons();
    wordCounter.textContent = "Word 0 / 0";
    previousWordButton.disabled = true;
    nextWordButton.disabled = true;
    saveNextButton.disabled = true;
    skipButton.disabled = true;
    badSampleButton.disabled = true;
    uploadStatus.classList.add("error");
    uploadStatus.textContent = error instanceof Error ? error.message : "Upload failed unexpectedly.";
  } finally {
    uploadButton.disabled = false;
    fileInput.value = "";
  }
});

trainButton.addEventListener("click", async () => {
  trainButton.disabled = true;
  actionStatus.textContent = "Training from saved labels and preparing the YOLOv11 character detector…";
  actionStatus.classList.remove("error");
  try {
    const status = await trainModel();
    modelIsTrained = status.trained_samples > 0;
    predictionEngine.textContent = status.model
      || (status.yolo_character_detector?.status === "running"
        ? "YOLOv11 character detector training"
        : status.segmentation_model?.trained
          ? "Word-boundary model ready"
          : "No word-recognition samples trained");
    actionStatus.textContent = status.message;
    if (uploadedSession?.words.length) await predictSelectedWord();
    if (status.yolo_character_detector?.status === "running") {
      void monitorYoloTraining();
    }
  } catch (error) {
    actionStatus.textContent = error instanceof Error ? error.message : "Training failed.";
    actionStatus.classList.add("error");
  } finally {
    trainButton.disabled = false;
  }
});

async function monitorYoloTraining() {
  const timer = window.setInterval(async () => {
    try {
      const status = await getTrainStatus();
      const detector = status.yolo_character_detector;
      if (detector?.status === "running") {
        actionStatus.textContent = "YOLOv11 is training character boxes from synthetic word layouts…";
        return;
      }
      window.clearInterval(timer);
      if (detector?.status === "complete") {
        const metrics = detector.metrics || {};
        predictionEngine.textContent =
          "YOLOv11 character detector ready; word boxes are grouped by spacing";
        actionStatus.textContent =
          `YOLOv11 trained. Synthetic validation mAP50: ${Number(metrics.validation_map50 || 0).toFixed(3)}. ` +
          "Upload a handwriting script or use Re-detect boxes to test character-to-word grouping.";
      } else {
        actionStatus.textContent = detector?.error || "YOLOv11 training did not complete.";
        actionStatus.classList.add("error");
      }
    } catch (error) {
      window.clearInterval(timer);
      actionStatus.textContent = error instanceof Error
        ? error.message
        : "Could not read YOLOv11 training status.";
      actionStatus.classList.add("error");
    }
  }, 5000);
}

window.addEventListener("DOMContentLoaded", async () => {
  renderCharacters();
  try {
    const [health, training] = await Promise.all([getHealth(), getTrainStatus()]);
    modelIsTrained = training.trained_samples > 0;
    predictionEngine.textContent = modelIsTrained
      ? `${training.model} ready (${training.trained_samples} saved samples)`
      : training.segmentation_model?.trained
        ? "Word-boundary model ready"
        : "No saved training samples yet";
    if (training.yolo_character_detector?.status === "complete") {
      predictionEngine.textContent = "YOLOv11 character detector ready; word boxes are grouped by spacing";
    } else if (training.yolo_character_detector?.status === "running") {
      predictionEngine.textContent = "YOLOv11 character detector training";
      void monitorYoloTraining();
    } else if (training.yolo_character_detector?.status === "failed") {
      predictionEngine.textContent = `YOLOv11 training failed: ${training.yolo_character_detector.error}`;
    }
    console.log("Backend ready:", health, training);
  } catch (error) {
    predictionEngine.textContent = "Could not load saved model status";
    console.error("Could not initialize app status:", error);
  }
  plot.draw();
});
