"""Browser tutor stages for pre-answer hints and submitted-answer checks."""
import json

from hint_prompts import INITIAL_HINT_MAX_NEW_TOKENS, render_initial_hint

EVIDENCE_MAX_NEW_TOKENS = 320

# Illustrative categories only. Replace or refine from research-based inventories.
MISCONCEPTIONS = {
    "arrow_direction": "Misinterprets what a diagram arrow represents or its direction",
    "label_confusion": "Confuses the identity or meaning of labels",
    "scientific_relationship": "Misunderstands the scientific relationship depicted",
    "concept_application": "Recognizes components but incorrectly applies the underlying concept",
    "unrelated_visual_detail": "Focuses on irrelevant diagram details",
    "insufficient_evidence": "Student response does not support a confident diagnosis",
}


def evidence_prompt(question):
    # Structured public inputs keep context and options separate from visual evidence.
    context, options = "", ""
    if isinstance(question, dict):
        context = question.get("context", "")
        options = "\n".join(f'{choice["id"]}. {choice["text"]}' for choice in question["choices"])
        question = question["question"]
    return f"""You are describing a scientific image for an AI tutor that generates progressive hints.

Question: {question}
Context: {context}
Options: {options}
Image: [attached image]

Extract the minimum sufficient visual observations relevant to the question. Use the question and options only to determine relevance, never as visual evidence.

Rules:
1. Preserve relevant objects, labels, counts, measurements, units, colours, positions, connections, and directional relationships.
2. For comparisons, report the corresponding features of each relevant group or panel separately, including size or scale when relevant.
3. For graphs and tables, record relevant labels, axes, scales, legends, and directly readable values.
4. Preserve visible text exactly. For arrows, use "X -> Y" (tail to head).
5. Describe only what is visually observable. Do not infer scientific meaning, calculate derived values, answer the question, evaluate options, or generate hints.
6. Avoid irrelevant details, repetition, and unsupported assumptions. Omit unreadable information.
Output:
- Numbered observations (1., 2., 3., ...), maximum 10.
- Start each line with a type label: Text, Count, Object, Position, Arrow, Connection, Measurement, Graph, Table, Colour, or another appropriate type.
- Identify the relevant panel, region, or object.
- Keep observations concise but preserve all necessary evidence.
- Output only the numbered observations."""


def extract_evidence(tutor, image, question):
    return tutor.ask(image, evidence_prompt(question), max_new_tokens=EVIDENCE_MAX_NEW_TOKENS)


def diagnose(tutor, image, question, student_answer, explanation, evidence):
    categories = json.dumps(MISCONCEPTIONS, indent=2)
    return tutor.ask(image, f"""Science question: {question}
Visual evidence (model-produced; may contain errors): {evidence}
Student answer: {student_answer}
Student explanation: {explanation}
Possible misconception categories: {categories}
Classify a likely misunderstanding ONLY if it is supported by what the student wrote.
Otherwise choose insufficient_evidence. Explain briefly, express uncertainty.
Do not reveal the question's correct answer.
Return a JSON object with keys: category, justification, confidence (low/medium/high).""", max_new_tokens=250)


def generate_hint(tutor, image, question, evidence, diagnosis, level=1):
    hints = {1: "Direct attention to relevant diagram evidence", 2: "Guide a useful scientific concept", 3: "Suggest a reasoning step without giving the result"}
    if level not in hints:
        raise ValueError("level must be 1, 2, or 3")
    return tutor.ask(image, f"""Science question: {question}
Visual evidence (unverified): {evidence}
Tentative diagnosis (unverified): {diagnosis}
Generate ONE concise, supportive tutoring hint at level {level}: {hints[level]}.
If the diagnosis is uncertain, ask one clarifying question instead of assuming a mistake.
Only reference diagram details you can verify visually. Do not state the answer, option letter, or reveal the solution.
Output only the hint.""", max_new_tokens=160)


def generate_initial_hint(tutor, image, question, evidence, level=1):
    return tutor.ask(image, render_initial_hint(question, level, str(evidence)),
                     max_new_tokens=INITIAL_HINT_MAX_NEW_TOKENS)


def assess_explanation(tutor, image, question, answer_text, explanation,
                       reference):
    prompt = f"""Assess only whether the student's reasoning is scientifically supported.
Question: {question}
The student's selected answer text: {answer_text}
The student's explanation: {explanation}
Reference explanation for evaluation only (never repeat or reveal it): {reference}
Compare the student's explanation to the reference, allowing scientifically equivalent wording.
If the explanation demonstrates the key idea, output exactly SUPPORTED.
If it conflicts with or misunderstands the key idea, output exactly NEEDS_WORK.
If the explanation is too vague to judge, output exactly UNCERTAIN.
Do not reveal the correct choice, reference explanation, or solution.
Output one label only."""
    response = tutor.ask(image, prompt, max_new_tokens=8)
    label = response.strip().split()[0].strip(" .,:;!").upper() if response.strip() else ""
    if label not in {"SUPPORTED", "NEEDS_WORK", "UNCERTAIN"}:
        return "uncertain", response
    status = {
        "SUPPORTED": "correct",
        "NEEDS_WORK": "needs_work",
        "UNCERTAIN": "uncertain",
    }[label]
    return status, response


def run_initial_hint(tutor, image, question, hint_level=1, on_stage=None, *, evidence_inputs=None):
    """Offer useful diagram guidance before the learner submits an answer."""
    if hint_level not in (1, 2, 3):
        raise ValueError("hint_level must be 1, 2, or 3")
    if on_stage is not None:
        on_stage("evidence")
    evidence = extract_evidence(tutor, image, question if evidence_inputs is None else evidence_inputs)
    if on_stage is not None:
        on_stage("hint")
    hint = generate_initial_hint(tutor, image, question, evidence, hint_level)
    return {
        "mode": "hint",
        "outcome": "hint",
        "evidence": evidence,
        "diagnosis": None,
        "explanation_assessment": None,
        "feedback": None,
        "hint": hint,
    }


def run_answered_pipeline(tutor, image, question, student_answer,
                          student_explanation, answer_text, correct_answer_id,
                          reference, hint_level=1, on_stage=None, *, evidence_inputs=None):
    """Check a submitted answer privately, then give feedback or another hint."""
    if hint_level not in (1, 2, 3):
        raise ValueError("hint_level must be 1, 2, or 3")
    if on_stage is not None:
        on_stage("evidence")
    evidence = extract_evidence(tutor, image, question if evidence_inputs is None else evidence_inputs)
    if on_stage is not None:
        on_stage("assessment")

    answer_correct = student_answer == correct_answer_id
    explanation_status, explanation_assessment = assess_explanation(
        tutor, image, question, answer_text, student_explanation, reference
    )

    if answer_correct and explanation_status == "correct":
        feedback = (
            "Correct—your answer and explanation show a sound understanding. "
            "Nice work."
        )
        if on_stage is not None:
            on_stage("feedback")
        return {
            "mode": "check",
            "outcome": "correct",
            "answer_correct": True,
            "explanation_status": explanation_status,
            "explanation_assessment": explanation_assessment,
            "evidence": evidence,
            "diagnosis": None,
            "feedback": feedback,
            "hint": None,
        }

    if on_stage is not None:
        on_stage("diagnosis")
    diagnosis = diagnose(
        tutor, image, question, answer_text, student_explanation, evidence
    )
    if on_stage is not None:
        on_stage("hint")
    hint = generate_hint(tutor, image, question, evidence, diagnosis, hint_level)
    if not answer_correct and explanation_status == "correct":
        feedback = (
            "Your explanation includes a useful idea, but the selected choice "
            "doesn't match yet. Use this hint to connect your reasoning to the question."
        )
    elif not answer_correct:
        feedback = "Not quite yet. Use this hint to rethink your answer."
    else:
        feedback = (
            "Your selected answer matches, but the explanation may need "
            "another look. Try strengthening your reasoning with this hint."
        )
    return {
        "mode": "check",
        "outcome": "try_again",
        "answer_correct": answer_correct,
        "explanation_status": explanation_status,
        "explanation_assessment": explanation_assessment,
        "evidence": evidence,
        "diagnosis": diagnosis,
        "feedback": feedback,
        "hint": hint,
    }
