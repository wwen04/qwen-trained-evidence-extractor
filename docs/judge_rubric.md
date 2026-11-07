# LLM-as-Judge Rubrics for Hint Evaluation (Mode 1: no learner attempt)

Version: draft v0.2. Tier definitions match the UI no-attempt hint prompt. Develop and refine on training/validation items only, then lock before final evaluation.

## How the judge is used

| Call | Unit judged | Purpose | Sees condition label? |
|---|---|---|---|
| Part A: Pointwise hint rubric | One hint (one question, one tier) | Primary results | No |
| Part B: Tier-distinctness check | The T1, T2, T3 hints of one system for one question | Is T3 more than a restatement of T2? | No |
| Part C: Evidence audit | The extracted evidence block (condition D only) | Evidence accuracy and answer leakage | n/a (D only) |
| Part D: Pairwise comparison | Two hints, same question and tier | Secondary head-to-head results | No |

Settings for all calls:
- Use a multimodal judge from a different model family than Qwen. Give it the image.
- Temperature 0. Structured JSON output only.
- Blind: strip condition labels, randomize the order of items, and never mention how a hint was generated.
- Do not ask the judge to measure length. Compute word counts in code and report them per condition.
- Run Part D in both orders (X/Y and Y/X).

---

## Shared tier definitions (used by the generator AND the judge)

Context: the student has not attempted the question and may not know where to begin. Exactly one short hint is shown at the requested tier.

The **Definition** column matches the generator's tier descriptions exactly. The compliant/non-compliant columns are **judge-only** guidance and are never passed to the hint generator.

| Tier | Definition | A compliant hint | A non-compliant hint |
|---|---|---|---|
| 1 | Invite the student to notice one relevant visual detail | Points to a specific element of the image (label, arrow, region, trend, object). | Explains the science concept, or gives a procedure or conclusion. |
| 2 | Connect a visible detail to a useful science idea | Links a specific visible detail to a relevant concept, without applying it to decide the answer. | Only says where to look (that is T1), or applies the concept to reach the answer. |
| 3 | Suggest a small reasoning step without solving the question | Names a concrete, doable action the student can perform next, tailored to this question. | A generic study tip, a restatement of the concept, or a step that completes the reasoning. |

---

## Part A: Pointwise hint rubric

### Inputs given to the judge
- Image
- Question text, context and ordered choices
- Gold answer and gold lecture/solution (reference only, never shown to the generator)
- Requested tier (1, 2 or 3) and the tier definition above
- The hint text

### Instructions to the judge
You are evaluating a tutoring hint for a school science question. The student has not attempted the question. Judge only the hint text against the criteria below. Use the image to verify visual claims. Use the gold answer and lecture/solution to judge scientific correctness and disclosure. Do not reward length, politeness or confident tone. Do not give credit for what the hint might have meant. Judge what it says. For each criterion, give a short quote or reason first, then the verdict.

### Criteria

**A1. Tier compliance** (yes / no)
- yes: the hint does what the requested tier's definition asks, and does not do the work of a different tier.
- no: the hint belongs to another tier, does several tiers' work at once, or is not a hint (for example, it is only a question back to the student, praise, or an answer).
- Note: a T2 hint that also points to a visual element still passes. A T1 hint that explains the concept fails. A T3 hint that carries out the whole reasoning chain fails.

**A2. Visual grounding** (yes / no)
- yes: the hint refers to at least one specific, identifiable element of this image (a named label, an arrow and its endpoints, a region, a bar, an axis, an object), and that reference is needed for the hint to make sense.
- no: the hint is generic and would be equally valid for any question on the topic, or refers to "the diagram" or "the picture" without identifying anything specific.
- This criterion exists so vague advice cannot score well just by avoiding errors.

**A3. Unsupported visual claims** (integer count, plus list)
- Count each statement about the image's content that is false, cannot be verified in the image, or misdescribes a label, direction, quantity or relation.
- List each such claim in `unsupported_claims`. A hint with no visual statements scores 0 here, which is why A2 is scored separately.
- Do not count correct statements, or scientific claims that are not about the image (those belong to A4).

**A4. Scientific correctness** (yes / no)
- yes: nothing in the hint contradicts the gold lecture/solution or established science at the school level.
- no: the hint contains at least one scientific error or a misleading statement of a concept.
- Do not penalize hints for omitting information.

**A5. Steers toward wrong answer** (yes / no)
- yes: following the hint as written would likely lead a student to a different answer than the gold answer, even if no single statement is false.
- no: otherwise.
- This is separate from A4 because a hint can be literally accurate and still misdirect.

**A6. Answer disclosure** (0 / 1 / 2)
- 0, none: the hint does not name the answer and does not favour or rule out any option. Pointing to a relevant region or concept is fine.
- 1, strong clue: the hint rules in or out one or more options, names the specific fact that decides between options, or reduces the remaining work to a trivial lookup, without stating the answer.
- 2, direct: the hint states the answer, names the correct option (letter or text) as correct, or states the conclusion that makes the answer immediate.
- Judge by effect on the student, not wording. An equivalent clue with no option letter still counts.
- Pointing to the entity that the correct answer is about is not by itself disclosure. Telling the student what that entity does, relative to the options, can be.

**A7. Actionability** (1-5)
- 1: no usable next move (praise, vague, restates the question).
- 2: a vague direction ("think about energy") with no concrete action.
- 3: a clear direction but the student must work out what to do with it.
- 4: a concrete action the student can do immediately, using something specific in this question.
- 5: a concrete, specific, doable action that leaves the key reasoning for the student, and is neither too small to matter nor large enough to solve the question.

### Output format (JSON only)

```json
{
  "tier_compliance": {"reason": "", "verdict": "yes|no"},
  "visual_grounding": {"reason": "", "verdict": "yes|no"},
  "unsupported_visual_claims": {"unsupported_claims": [], "count": 0},
  "scientific_correctness": {"reason": "", "verdict": "yes|no"},
  "steers_toward_wrong_answer": {"reason": "", "verdict": "yes|no"},
  "answer_disclosure": {"reason": "", "score": 0},
  "actionability": {"reason": "", "score": 1}
}
```

### Derived metrics (computed in code, not by the judge)

- **Good hint** (headline rate) = tier_compliance = yes AND visual_grounding = yes AND count = 0 AND scientific_correctness = yes AND steers_toward_wrong_answer = no AND disclosure = 0.
- Also report each component separately, per tier and per condition, plus mean actionability.
- Report everything split by Experiment A correctness (the model answered that question correctly vs incorrectly).
- Report disclosure as a distribution (share at 0, 1, 2), not only a mean.

---

## Part B: Tier-distinctness check

Run once per question per system, with the three hints shown together and labelled only T1, T2, T3.

### Inputs
Image, question, choices, and the three hint texts.

### Instructions to the judge
Decide whether the three hints form a progression in which each tier adds something the previous one does not.

### Criteria

- **t2_adds_beyond_t1** (yes / no): T2 adds a science connection that T1 does not already contain.
- **t3_adds_beyond_t2** (yes / no): T3 gives a concrete reasoning action that is not just a restatement of T2's concept.
- **t3_is_generic** (yes / no): T3 could be pasted unchanged onto many other questions (for example, "look at each option and compare to the diagram").
- **redundant_pairs**: list any pair of tiers that say substantially the same thing.

```json
{
  "t2_adds_beyond_t1": {"reason": "", "verdict": "yes|no"},
  "t3_adds_beyond_t2": {"reason": "", "verdict": "yes|no"},
  "t3_is_generic": {"reason": "", "verdict": "yes|no"},
  "redundant_pairs": []
}
```

---

## Part C: Evidence audit (condition D only)

### Inputs
Image, question, choices, gold answer, and the extracted evidence list.

### Criteria

- **incorrect_observations** (integer, plus list): observations that misread a label, an arrow direction, a quantity or a relation, or describe something not in the image.
- **missed_key_evidence** (yes / no): a feature of the image that is necessary for the question is absent from the evidence.
- **answer_implying** (yes / no): the evidence, taken alone, makes the correct answer easy to infer (for example, it states the key relation that decides between options). Evidence that merely lists what is visible is not answer-implying.
- **contains_interpretation** (yes / no): the evidence includes scientific interpretation or conclusions, not only observations.

```json
{
  "incorrect_observations": {"list": [], "count": 0},
  "missed_key_evidence": {"reason": "", "verdict": "yes|no"},
  "answer_implying": {"reason": "", "verdict": "yes|no"},
  "contains_interpretation": {"reason": "", "verdict": "yes|no"}
}
```

Use Part C for the error-propagation analysis: compare Part A outcomes for D hints when the evidence is accurate vs when it contains errors, and check whether D's disclosure cases come from answer-implying evidence.

---

## Part D: Pairwise comparison

### Inputs
Image, question, choices, gold answer and gold lecture/solution, requested tier with its definition, and two hints labelled Hint X and Hint Y.

### Instructions to the judge
Decide which hint is the better tutoring hint at the stated tier for a student who has not attempted the question and does not know where to begin. Apply these checks in order, and stop at the first one that separates the hints:

1. **Disqualifiers.** A hint loses if it discloses the answer (directly, or by a strong clue), makes false visual claims, contains a scientific error, or would steer a student to a wrong answer. If both have such problems, prefer the one with fewer and less severe ones.
2. **Tier compliance.** Prefer the hint that does what the tier asks and nothing else.
3. **Grounding.** Prefer the hint that refers to a specific, identifiable element of the image.
4. **Actionability.** Prefer the hint that gives the student a more concrete and more doable next move while leaving the reasoning to them.

Do not prefer a hint because it is longer, more detailed or more polite. If the hints are equivalent on all four checks, answer "tie".

```json
{
  "deciding_check": "disqualifier|tier_compliance|grounding|actionability|none",
  "reason": "",
  "winner": "X|Y|tie"
}
```

### Scoring pairwise results (in code)

- Run both orders. Count a win for a system only if it wins in both orders. Any disagreement between orders, or a tie in either order, counts as a tie.
- Report win / tie / loss rates for B vs D and C vs D, separately for each tier, with 95% paired bootstrap CIs resampling questions or shared-diagram groups.

---

## Before locking the rubric

1. Add 2-3 anchored examples per criterion (especially A6 disclosure and A7 actionability) taken from training/validation items, written into the judge prompt.
2. Run B, C and D on about 30 validation questions and check where the judge's reasons seem wrong or inconsistent. Tighten the wording and rerun.
3. Have one or two team members blind-label about 50-60 hints on A1, A2, A4, A6 and A7 and compute Cohen's kappa against the judge. Report it.
4. Run the judge twice on a subset to check stability.
5. Freeze the judge model name and version, prompt text, temperature and rubric version, and record them with the results.
