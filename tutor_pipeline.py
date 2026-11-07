"""Tutor flow using selected visual tensors, with no textual evidence-generation pass."""
import json

from hint_prompts import INITIAL_HINT_MAX_NEW_TOKENS, render_initial_hint
from legacy_evidence import MISCONCEPTIONS


def selection_details(tutor):
    selection = getattr(tutor, "last_selection", None)
    if not isinstance(selection, dict):
        return "Visual features prepared.", None
    label = selection.get("label", "Visual selector")
    detail = f"{label}. {selection['visual_tokens']} visual tokens; mean gate {selection['mean_gate']:.3f}."
    if selection.get("mode") == "identity":
        detail += " The selector passes all original features through unchanged."
    detail += " These are feature weights, not a list of verified observations."
    return detail, selection


def run_initial_hint(tutor, image, question, hint_level=1, on_stage=None, *, evidence_inputs=None):
    if hint_level not in (1,2,3):
        raise ValueError("hint level must be 1, 2, or 3")
    if on_stage:
        on_stage("evidence")
        on_stage("hint")
    hint = tutor.ask(image, render_initial_hint(question, hint_level), max_new_tokens=INITIAL_HINT_MAX_NEW_TOKENS,
                     evidence_inputs=evidence_inputs)
    evidence, selection = selection_details(tutor)
    return {"mode": "hint", "outcome": "hint", "evidence": evidence, "selection": selection,
            "diagnosis": None, "explanation_assessment": None, "feedback": None, "hint": hint}


def run_answered_pipeline(tutor, image, question, student_answer, student_explanation, answer_text,
                          correct_answer_id, reference, hint_level=1, on_stage=None, *, evidence_inputs=None):
    if hint_level not in (1,2,3):
        raise ValueError("hint level must be 1, 2, or 3")
    if on_stage:
        on_stage("evidence")
        on_stage("assessment")
    assessment = tutor.ask(image, f"""Assess only whether the student's reasoning is scientifically supported.
Question: {question}
Student's selected answer: {answer_text}
Student's reasoning: {student_explanation}
Reference for private evaluation only: {reference}
Output exactly SUPPORTED, NEEDS_WORK, or UNCERTAIN. Never reproduce the reference.""",
        max_new_tokens=8, evidence_inputs=evidence_inputs)
    label = assessment.strip().split()[0].strip(" .,:;!").upper() if assessment.strip() else ""
    status = {"SUPPORTED":"correct", "NEEDS_WORK":"needs_work"}.get(label, "uncertain")
    correct = student_answer == correct_answer_id
    diagnosis, hint = None, None
    if correct and status == "correct":
        feedback, outcome = "Correct—your answer and explanation show a sound understanding. Nice work.", "correct"
        if on_stage:
            on_stage("feedback")
    else:
        if on_stage:
            on_stage("diagnosis")
        diagnosis = tutor.ask(image, f"""Question: {question}
Student's answer: {answer_text}
Student's explanation: {student_explanation}
Categories: {json.dumps(MISCONCEPTIONS)}
Diagnose only what the student's words support; otherwise use insufficient_evidence.
Return JSON with category, justification and confidence (low/medium/high). Do not reveal the correct answer.""",
            max_new_tokens=250, evidence_inputs=evidence_inputs)
        if on_stage:
            on_stage("hint")
        instructions = {1:"Direct attention to relevant diagram details",2:"Connect a visible detail to a science concept",3:"Suggest one small reasoning step without its result"}
        hint = tutor.ask(image, f"""Question: {question}
Tentative diagnosis (unverified): {diagnosis}
Give one concise supportive level {hint_level} hint: {instructions[hint_level]}.
Only mention visible details you can verify. Do not state the answer or reveal the solution.
If the diagnosis is uncertain, give a general question-specific starting point.""",
            max_new_tokens=160, evidence_inputs=evidence_inputs)
        feedback = "Your selected answer matches, but the explanation needs another look." if correct else "Not quite yet. Use this hint to rethink your answer."
        outcome = "try_again"
    evidence, selection = selection_details(tutor)
    return {"mode":"check", "outcome":outcome, "answer_correct":correct, "explanation_status":status,
            "explanation_assessment":assessment, "evidence":evidence, "selection":selection,
            "diagnosis":diagnosis, "feedback":feedback, "hint":hint}
