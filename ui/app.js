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
const stageNames = ["evidence", "assessment", "response"];

let selectedQuestion = null;
let busy = false;
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
  hintButton.disabled = busy || !selectedQuestion;
  checkButton.disabled = busy || !selectedQuestion || !hasAnswer || !reasoningInput.value.trim();
  inspectionButton.disabled = !selectedQuestion;
  hintButton.querySelector("span:first-child").textContent =
    busy ? "Tutor is thinking…" : "I'm stuck — give me a hint";
  checkButton.querySelector("span:first-child").textContent =
    busy ? "Tutor is thinking…" : "Check my answer";
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
initialize();
