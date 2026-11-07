const splitSelect = document.querySelector("#split-select");
const questionSelect = document.querySelector("#question-select");
const questionText = document.querySelector("#question-text");
const contextText = document.querySelector("#question-context");
const choicesBox = document.querySelector("#choices");
const radioBox = document.querySelector("#answer-radios");
const diagramFrame = document.querySelector("#diagram-frame");
const questionImage = document.querySelector("#question-image");
const reasoningInput = document.querySelector("#reasoning-input");
const hintLevel = document.querySelector("#hint-level");
const hintButton = document.querySelector("#hint-button");
const checkButton = document.querySelector("#check-button");
const sampleButton = document.querySelector("#sample-button");
const inspectionButton = document.querySelector("#inspection-button");
const inspectionFrame = document.querySelector("#inspection-frame");
const inspectionMessage = document.querySelector("#inspection-message");
const setupMessage = document.querySelector("#setup-message");
const demoNotice = document.querySelector("#demo-notice");
const errorMessage = document.querySelector("#error-message");
const hintCard = document.querySelector("#hint-card");
const hintText = document.querySelector("#hint-text");
const feedbackCard = document.querySelector("#feedback-card");
const feedbackText = document.querySelector("#feedback-text");
const outcomeBadge = document.querySelector("#outcome-badge");
const resultFile = document.querySelector("#result-file");
const comparisonSplit = document.querySelector("#comparison-split");
const comparisonButton = document.querySelector("#comparison-button");
const comparisonAvailability = document.querySelector("#comparison-availability");
const comparisonProgress = document.querySelector("#comparison-progress");
const comparisonProgressBar = document.querySelector("#comparison-progress-bar");
const comparisonProgressText = document.querySelector("#comparison-progress-text");
const comparisonError = document.querySelector("#comparison-error");
const comparisonResult = document.querySelector("#comparison-result");
const comparisonFile = document.querySelector("#comparison-file");
const comparisonDisplay = document.querySelector("#comparison-display");
const comparisonPredictions = document.querySelector("#comparison-predictions");
const stageNames = ["evidence", "assessment", "response"];

let selectedQuestion = null;
let busy = false;
let comparisonBusy = false;
let comparisonAvailable = false;
let inspectionUrl = null;

function showMessage(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function updateRunButton() {
  const hasAnswer = document.querySelector('input[name="student-answer"]:checked');
  const unavailable = busy || comparisonBusy;
  hintButton.disabled = unavailable || !selectedQuestion;
  checkButton.disabled = unavailable || !selectedQuestion || !hasAnswer || !reasoningInput.value.trim();
  inspectionButton.disabled = unavailable || !selectedQuestion;
  hintButton.querySelector("span:first-child").textContent =
    unavailable ? "Please wait…" : "I'm stuck — give me a hint";
  checkButton.querySelector("span:first-child").textContent =
    unavailable ? "Please wait…" : "Check my answer";
  comparisonButton.disabled = unavailable || !comparisonAvailable;
  comparisonSplit.disabled = unavailable;
  comparisonButton.querySelector("span:first-child").textContent =
    comparisonBusy ? "Comparing all modes…" : "Compare all modes";
}

function clearResult() {
  for (const stage of document.querySelectorAll(".stage-card")) {
    stage.dataset.state = "";
    stage.querySelector(".stage-state").textContent = "Waiting";
    stage.querySelector(".stage-output").textContent = "";
    stage.querySelector("details").open = false;
  }
  hintCard.hidden = true;
  feedbackCard.hidden = true;
  errorMessage.hidden = true;
  resultFile.hidden = true;
}

function resetInspection() {
  inspectionUrl = null;
  inspectionFrame.hidden = true;
  inspectionFrame.removeAttribute("src");
  inspectionButton.textContent = "Show inspection report";
  inspectionButton.setAttribute("aria-expanded", "false");
  showMessage(inspectionMessage, "");
}

function renderQuestion(item) {
  selectedQuestion = item;
  questionText.textContent = item.question;
  contextText.textContent = item.context || "";
  contextText.hidden = !item.context;

  choicesBox.replaceChildren();
  radioBox.replaceChildren();
  for (const choice of item.choices) {
    const display = `${choice.id}. ${choice.text}`;
    const choiceItem = document.createElement("div");
    choiceItem.className = "choice-item";
    const letter = document.createElement("span");
    letter.className = "choice-letter";
    letter.textContent = choice.id;
    const text = document.createElement("span");
    text.textContent = choice.text;
    choiceItem.append(letter, text);
    choicesBox.append(choiceItem);

    const label = document.createElement("label");
    label.className = "answer-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "student-answer";
    input.value = choice.id;
    input.addEventListener("change", () => {
      demoNotice.hidden = true;
      updateRunButton();
    });
    const answerText = document.createElement("span");
    answerText.textContent = display;
    label.append(input, answerText);
    radioBox.append(label);
  }

  questionImage.src = item.image_url;
  questionImage.alt = `Diagram for ${item.item_id}`;
  diagramFrame.hidden = false;
  resetInspection();
  inspectionUrl = item.inspection_url || null;
  clearResult();
  updateRunButton();
}

async function loadQuestionList() {
  selectedQuestion = null;
  resetInspection();
  hintButton.disabled = true;
  checkButton.disabled = true;
  questionSelect.disabled = true;
  questionText.textContent = "Choose a prepared question to begin.";
  contextText.hidden = true;
  choicesBox.replaceChildren();
  radioBox.replaceChildren();
  diagramFrame.hidden = true;
  clearResult();
  questionSelect.replaceChildren(new Option("Loading questions…", ""));
  showMessage(setupMessage, "");

  try {
    const split = splitSelect.value;
    const data = await getJson(`/api/questions?split=${encodeURIComponent(split)}`);
    questionSelect.replaceChildren();
    for (const item of data.questions) {
      questionSelect.add(new Option(`${item.item_id} · ${item.question}`, item.item_id));
    }
    if (data.questions.length === 0) {
      questionSelect.add(new Option("No questions in this split", ""));
      showMessage(setupMessage, "There are no prepared questions in this split.");
    } else {
      questionSelect.disabled = false;
      await loadSelectedQuestion();
    }
  } catch (error) {
    questionSelect.replaceChildren(new Option("Questions unavailable", ""));
    showMessage(setupMessage, error.message);
  }
}

async function loadSelectedQuestion() {
  const itemId = questionSelect.value;
  if (!itemId) return;
  try {
    const query = new URLSearchParams({ split: splitSelect.value, item_id: itemId });
    const item = await getJson(`/api/question?${query}`);
    renderQuestion(item);
    showMessage(setupMessage, "");
  } catch (error) {
    selectedQuestion = null;
    updateRunButton();
    showMessage(setupMessage, error.message);
  }
}

function setStage(stage, state, label) {
  const card = document.querySelector(`.stage-card[data-stage="${stage}"]`);
  if (!card) return;
  card.dataset.state = state;
  card.querySelector(".stage-state").textContent = label;
}

function renderResult(result, jobId, outputFile) {
  document.querySelector("#evidence-output").textContent = result.evidence;
  if (result.mode === "hint") {
    document.querySelector("#assessment-output").textContent =
      "No answer submitted yet. The tutor gave an answer-independent hint.";
    setStage("assessment", "", "Not needed");
    outcomeBadge.textContent = "A place to start";
    outcomeBadge.className = "outcome-badge";
    feedbackText.textContent = "You can ask for another hint or try an answer when ready.";
  } else {
    const explanationLabels = {
      correct: "Reasoning appears supported",
      needs_work: "Reasoning needs work",
      uncertain: "Reasoning is unclear",
      not_assessed: "Not separately assessed",
    };
    const assessment = [
      `Selected answer: ${result.answer_correct ? "Correct" : "Not correct yet"}`,
      `Explanation: ${explanationLabels[result.explanation_status] || "Unclear"}`,
      result.diagnosis ? `Tutor's reasoning: ${result.diagnosis}` : "",
    ].filter(Boolean);
    document.querySelector("#assessment-output").textContent = assessment.join("\n\n");
    outcomeBadge.textContent = result.outcome === "correct" ? "Correct — well done" : "Keep thinking";
    outcomeBadge.className = result.outcome === "correct"
      ? "outcome-badge outcome-correct"
      : "outcome-badge outcome-retry";
    feedbackText.textContent = result.feedback;
    setStage("assessment", "complete", "Checked");
  }
  document.querySelector("#response-output").textContent =
    [result.feedback, result.hint].filter(Boolean).join("\n\n");
  for (const name of ["evidence", "response"]) setStage(name, "complete", "Complete");
  feedbackCard.hidden = false;
  hintText.textContent = result.hint || "";
  hintCard.hidden = !result.hint;
  resultFile.href = `/api/results/${encodeURIComponent(jobId)}`;
  resultFile.textContent = `Download saved result: ${outputFile}`;
  resultFile.hidden = false;
}

async function pollJob(jobId) {
  while (true) {
    const job = await getJson(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (job.stage === "evidence") {
      setStage("evidence", "active", "In progress");
    } else if (job.stage === "assessment") {
      setStage("evidence", "complete", "Complete");
      setStage("assessment", "active", "In progress");
    } else if (job.stage === "diagnosis") {
      setStage("evidence", "complete", "Complete");
      setStage("assessment", "active", "In progress");
    } else if (job.stage === "hint" || job.stage === "feedback") {
      setStage("evidence", "complete", "Complete");
      if (job.mode === "hint") setStage("assessment", "", "Not needed");
      else setStage("assessment", "complete", "Checked");
      setStage("response", "active", "In progress");
    } else if (job.stage === "loading_model") {
      setStage("evidence", "active", "Loading model");
    }
    if (job.status === "complete") {
      renderResult(job.result, job.job_id, job.output_file);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "The tutor run failed.");
    await new Promise(resolve => setTimeout(resolve, 1100));
  }
}

async function runTutor(mode) {
  if (busy || !selectedQuestion) return;
  const selected = document.querySelector('input[name="student-answer"]:checked');
  if (mode === "check" && (!selected || !reasoningInput.value.trim())) return;

  busy = true;
  errorMessage.hidden = true;
  clearResult();
  setStage("evidence", "active", "Starting");
  updateRunButton();
  try {
    const data = await getJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        split: splitSelect.value,
        item_id: selectedQuestion.item_id,
        student_answer: mode === "check" ? selected.value : null,
        explanation: mode === "check" ? reasoningInput.value.trim() : null,
        hint_level: Number(hintLevel.value),
      }),
    });
    await pollJob(data.job_id);
  } catch (error) {
    for (const stage of document.querySelectorAll(".stage-card")) {
      if (stage.dataset.state === "active") {
        stage.dataset.state = "failed";
        stage.querySelector(".stage-state").textContent = "Stopped";
      }
    }
    showMessage(errorMessage, error.message);
  } finally {
    busy = false;
    updateRunButton();
  }
}

function formatAccuracy(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderComparison(job) {
  const result = job.result;
  const identityAccuracy = result.results.identity.accuracy;
  const best = result.results[result.best_mode];
  document.querySelector("#comparison-verdict").textContent =
    `Best on ${result.split}: ${result.best_mode} (${formatAccuracy(best.accuracy)})`;
  document.querySelector("#comparison-count").textContent =
    `${result.questions} ${result.split} questions · trained ${result.trained_checkpoint.selector_config.layers}-layer selector, ` +
    `rho=${result.rho}, seed=${result.trained_checkpoint.seed}`;
  const checkpointValidation = result.trained_checkpoint.best_validation;
  document.querySelector("#comparison-checkpoint").textContent = checkpointValidation
    ? `Best checkpoint validation accuracy: ${formatAccuracy(checkpointValidation.accuracy)} ` +
      `(${checkpointValidation.questions} questions).`
    : "Checkpoint validation metrics are unavailable.";

  const labels = {
    identity: "Original features; unchanged baseline",
    soft: `Learned continuous feature blending (training target rho=${result.rho})`,
    hard: `Keep top-scoring ceil(N × ${result.rho}) features`,
    random: `Keep seeded random ceil(N × ${result.rho}) features`,
    remove: `Replace top-scoring ceil(N × ${result.rho}) features with null`,
    remove_random: `Replace seeded random ceil(N × ${result.rho}) features with null`,
    null: "Replace all visual features with the learned null feature",
    no_image: "Omit the image entirely",
  };
  const modeBody = document.querySelector("#comparison-modes-body");
  modeBody.replaceChildren();
  for (const mode of result.modes) {
    const metrics = result.results[mode];
    const paired = metrics.paired_vs_identity;
    const delta = (metrics.accuracy - identityAccuracy) * 100;
    const row = document.createElement("tr");
    row.dataset.best = String(mode === result.best_mode);
    const values = [
      mode,
      labels[mode] || "Selector evaluation mode",
      `${formatAccuracy(metrics.accuracy)} · ` +
        `${formatAccuracy(metrics.accuracy_ci95_image_group_bootstrap[0])}–` +
        `${formatAccuracy(metrics.accuracy_ci95_image_group_bootstrap[1])}`,
      `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`,
      `${paired.won} / ${paired.lost}`,
      paired.mcnemar_exact_p.toPrecision(3),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    modeBody.append(row);
  }
  document.querySelector("#comparison-note").textContent = result.note;
  comparisonFile.href = `/api/comparisons/${encodeURIComponent(job.job_id)}/result`;
  comparisonFile.hidden = false;
  comparisonDisplay.hidden = false;
  comparisonDisplay.textContent = "Display all mode predictions";
  comparisonDisplay.setAttribute("aria-expanded", "false");
  const predictionHead = document.querySelector("#predictions-head");
  const header = document.createElement("tr");
  for (const label of ["Question", "Correct answer", ...result.modes]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  }
  predictionHead.replaceChildren(header);
  const rowsByMode = Object.fromEntries(result.modes.map(
    mode => [mode, new Map(result.predictions[mode].map(row => [row.item_id, row]))],
  ));
  const body = document.querySelector("#predictions-body");
  body.replaceChildren();
  for (const original of result.predictions.identity) {
    const itemId = original.item_id;
    const row = document.createElement("tr");
    for (const value of [itemId, original.gold]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    for (const mode of result.modes) {
      const prediction = rowsByMode[mode].get(itemId);
      if (!prediction) throw new Error(`Missing ${mode} prediction for ${itemId}.`);
      const cell = document.createElement("td");
      cell.textContent = `${prediction.predicted} · ${prediction.correct ? "correct" : "wrong"}`;
      cell.dataset.correct = String(prediction.correct);
      row.append(cell);
    }
    body.append(row);
  }
  document.querySelector("#predictions-count").textContent =
    `${result.questions} questions · correct and wrong labels shown per mode`;
  comparisonResult.hidden = false;
}

function toggleComparisonPredictions() {
  const expanded = comparisonDisplay.getAttribute("aria-expanded") === "true";
  comparisonDisplay.setAttribute("aria-expanded", String(!expanded));
  comparisonDisplay.textContent = expanded
    ? "Display all mode predictions"
    : "Hide all mode predictions";
  comparisonPredictions.hidden = expanded;
}

async function pollComparison(jobId) {
  while (true) {
    const job = await getJson(`/api/comparisons/${encodeURIComponent(jobId)}`);
    comparisonProgressBar.value = job.progress || 0;
    comparisonProgressText.textContent = job.total
      ? `Scored ${job.completed} of ${job.total} questions`
      : "Loading the model and evaluation data…";
    if (job.status === "complete") {
      renderComparison(job);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "The comparison failed.");
    await new Promise(resolve => setTimeout(resolve, 900));
  }
}

async function runComparison() {
  if (busy || comparisonBusy || !comparisonAvailable) return;
  comparisonBusy = true;
  comparisonError.hidden = true;
  comparisonResult.hidden = true;
  comparisonFile.hidden = true;
  comparisonDisplay.hidden = true;
  comparisonPredictions.hidden = true;
  comparisonDisplay.setAttribute("aria-expanded", "false");
  comparisonProgress.hidden = false;
  comparisonProgressBar.value = 0;
  comparisonProgressText.textContent = "Starting paired evaluation…";
  updateRunButton();
  try {
    const data = await getJson("/api/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ split: comparisonSplit.value }),
    });
    await pollComparison(data.job_id);
  } catch (error) {
    showMessage(comparisonError, error.message);
  } finally {
    comparisonBusy = false;
    comparisonProgress.hidden = true;
    updateRunButton();
  }
}

function toggleInspection() {
  if (!selectedQuestion) return;
  if (!inspectionFrame.hidden) {
    inspectionFrame.hidden = true;
    inspectionButton.textContent = "Show inspection report";
    inspectionButton.setAttribute("aria-expanded", "false");
    return;
  }

  if (!inspectionUrl) {
    showMessage(inspectionMessage, "The inspection report is not available for this question.");
    return;
  }

  inspectionFrame.src = inspectionUrl;
  inspectionFrame.hidden = false;
  inspectionButton.textContent = "Hide inspection report";
  inspectionButton.setAttribute("aria-expanded", "true");
  updateRunButton();
}

function loadDemoResponse() {
  const firstOption = document.querySelector('input[name="student-answer"]');
  if (!firstOption) return;
  firstOption.checked = true;
  reasoningInput.value = "The extra nutrients may help the bacteria produce more insulin.";
  hintLevel.value = "1";
  demoNotice.hidden = false;
  updateRunButton();
}

async function initialize() {
  try {
    const status = await getJson("/api/status");
    document.querySelector("#model-status").textContent = status.selector?.label || "Local research prototype";
    comparisonAvailable = Boolean(status.comparison_available);
    if (!comparisonAvailable) {
      showMessage(
        comparisonAvailability,
        "To enable comparison, restart the UI with --selector-checkpoint pointing to a trained selector checkpoint, such as outputs/training/selector-1layer-rho05-seed43/best.",
      );
    }
    updateRunButton();
    for (const option of [...splitSelect.options]) {
      if (!(option.value in status.counts)) option.remove();
    }
    const badge = document.querySelector("#dataset-badge");
    const total = Object.values(status.counts).reduce((sum, count) => sum + count, 0);
    if (status.prepared) {
      badge.textContent = `${total.toLocaleString()} questions ready`;
      badge.classList.add("ready");
    } else {
      badge.textContent = "Dataset not prepared";
      badge.classList.add("unavailable");
    }
  } catch (error) {
    showMessage(setupMessage, error.message);
  }
  await loadQuestionList();
}

splitSelect.addEventListener("change", loadQuestionList);
questionSelect.addEventListener("change", loadSelectedQuestion);
reasoningInput.addEventListener("input", () => {
  demoNotice.hidden = true;
  updateRunButton();
});
sampleButton.addEventListener("click", loadDemoResponse);
inspectionButton.addEventListener("click", toggleInspection);
hintButton.addEventListener("click", () => runTutor("hint"));
checkButton.addEventListener("click", () => runTutor("check"));
comparisonButton.addEventListener("click", runComparison);
comparisonDisplay.addEventListener("click", toggleComparisonPredictions);
initialize();
