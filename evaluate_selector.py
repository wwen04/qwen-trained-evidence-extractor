"""Answer accuracy and feature-removal controls. Never trains or calls a judge."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from project_data import digest, file_hash, pixel_hash, load_questions, load_references, read_jsonl, shuffle_options, stable_seed, write_json


def comparison_fingerprint(provenance):
    """Exclude the experimental selector while fixing the backbone and inputs."""
    backbone = provenance["backbone"]
    return digest({"backbone": {"model_path": backbone["model_path"],
                               "revision": backbone.get("resolved_revision") or backbone.get("revision")},
                   **{key: provenance[key] for key in ("backbone_signature", "preprocessing", "precision",
                      "quantization", "letter_token_ids", "conditioning", "answer_instruction_sha256")}})


def bootstrap_accuracy(rows, seed=42, repetitions=2000):
    import numpy as np
    if not rows:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(int(row["correct"]))
    groups = list(groups.values())
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        indexes = rng.integers(0, len(groups), size=len(groups))
        samples.append(float(sums[indexes].sum() / counts[indexes].sum()))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(rows):
    return {"questions": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "accuracy_ci95_image_group_bootstrap": bootstrap_accuracy(rows),
            "failures": sum(r["status"] != "ok" for r in rows),
            "gold_letter_counts": dict(Counter(r["gold"] for r in rows)),
            "predicted_letter_counts": dict(Counter(r["predicted"] or "FAILED" for r in rows))}


def paired_comparison(rows, baseline):
    current = {r["item_id"]: r for r in rows}
    previous = {r["item_id"]: r for r in baseline}
    if len(current) != len(rows) or len(previous) != len(baseline):
        raise ValueError("Paired comparison requires exactly one prediction per question")
    if current.keys() != previous.keys():
        raise ValueError("Paired comparison requires identical question IDs")
    for key in current:
        if (not current[key].get("comparison_sha256") or not previous[key].get("comparison_sha256")
                or any(current[key][f] != previous[key][f] for f in
                       ("input_sha256", "gold", "image_sha256", "comparison_sha256"))):
            raise ValueError("Paired comparison inputs differ")
    won = sum(current[k]["correct"] and not previous[k]["correct"] for k in current)
    lost = sum(not current[k]["correct"] and previous[k]["correct"] for k in current)
    discordant = won + lost
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(won, lost)+1)) / 2**discordant) if discordant else 1.0
    return {"won": won, "lost": lost, "mcnemar_exact_p": p,
            "note": "McNemar treats questions as independent; shared diagrams limit that assumption. Accuracy CIs resample image groups."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scienceqa", "ai2d"], default="scienceqa")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=["identity", "no_image", "soft", "hard", "random", "remove", "remove_random", "null"], default="identity")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--mismatched-question", action="store_true")
    parser.add_argument("--faithfulness", action="store_true", help="Run identity, keep/remove/random controls at 0.1,0.3,0.5,0.7,1.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="An identity/no-image predictions.jsonl from a matched run")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    items = load_questions(args.dataset, args.split)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        items = items[:args.limit]
    if not items:
        parser.error("No evaluation items")
    if not 0 <= args.rho <= 1:
        parser.error("--rho must be in [0,1]")
    if args.mismatched_question and len(items) < 2:
        parser.error("Question mismatch needs at least two items")
    if args.preflight:
        print({"dataset": args.dataset, "split": args.split, "questions": len(items), "model_loaded": False})
        return
    import torch
    from model_runtime import QwenSelectorRuntime, read_runtime
    config = read_runtime(args.runtime_config, model_path=args.model_path, selector_checkpoint=str(args.checkpoint) if args.checkpoint else None)
    if (args.mode not in {"identity", "no_image"} or args.faithfulness) and not config.get("selector_checkpoint"):
        parser.error("Gate evaluation requires a checkpoint so all controls share its null token")
    runtime = QwenSelectorRuntime(config)
    provenance = runtime.provenance()
    comparison_sha256 = comparison_fingerprint(provenance)
    baseline = read_jsonl(args.compare) if args.compare else None
    if baseline is not None:
        if not baseline or len({r["condition"] for r in baseline}) != 1:
            raise ValueError("Comparison baseline must contain exactly one condition")
        if any(r.get("comparison_sha256") != comparison_sha256 for r in baseline):
            raise ValueError("Comparison backbone, preprocessing, precision or QA protocol differs; rerun a matched baseline")
    refs = load_references(args.dataset, args.split)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    conditions = [(args.mode, args.rho)]
    if args.faithfulness:
        if runtime.selector.config.variant == "free_transform":
            parser.error("Feature-removal tests require a gated selector")
        conditions = [("identity", 1.0)] + [(m, r) for r in (0.1, 0.3, 0.5, 0.7, 1.0) for m in ("hard", "random", "remove", "remove_random")]
    write_json(output / "run.json", {"status": "running", "dataset": args.dataset, "split": args.split,
               "item_ids": [i["item_id"] for i in items], "runtime": config, "provenance": provenance,
               "comparison_sha256": comparison_sha256,
               "checkpoint_weights_sha256": runtime.metadata["weights_sha256"] if runtime.metadata else None,
               "conditions": conditions, "shuffle_options": args.shuffle_options, "mismatched_question": args.mismatched_question,
               "seed": args.seed, "note": "Feature replacement does not reduce sequence length or compute. No localization annotations are included."})
    all_rows = []
    for mode, rho in conditions:
        for index, original in enumerate(items):
            gold_id = refs[original["item_id"]]["gold_option_id"]
            if args.shuffle_options:
                item, gold_index = shuffle_options(original, gold_id, stable_seed(args.seed, original["item_id"]))
                gold_id = chr(65+gold_index)
            else:
                item = original
            conditioning = items[(index+1) % len(items)] if args.mismatched_question else None
            row = {"item_id": item["item_id"], "condition": f"{mode}:{rho}", "mode": mode, "rho": rho,
                   "gold": gold_id, "input_sha256": digest({k:v for k,v in item.items() if k != "image_path"}),
                   "image_sha256": file_hash(item["image_path"]), "pixel_sha256": pixel_hash(item["image_path"]),
                   "comparison_sha256": comparison_sha256,
                   "conditioning_item_id": conditioning["item_id"] if conditioning else item["item_id"]}
            try:
                with torch.no_grad():
                    logits, gates = runtime.answer_forward(item, mode=mode, rho=rho,
                        seed=stable_seed(args.seed, item["item_id"]), conditioning_item=conditioning)
                    restricted = logits[:, runtime.letter_ids[:len(item["choices"])]].float()
                    if not torch.isfinite(restricted).all():
                        raise ValueError("Nonfinite option logits")
                    predicted = chr(65+restricted.argmax(-1).item())
                row.update(status="ok", predicted=predicted, correct=predicted == gold_id,
                           option_probabilities=restricted.softmax(-1)[0].cpu().tolist(),
                           mean_gate=float(gates.mean()) if len(gates) else None,
                           gates=gates.float().cpu().tolist())
            except (RuntimeError, ValueError) as error:
                row.update(status="failed", predicted=None, correct=False, error=f"{type(error).__name__}: {error}")
            all_rows.append(row)
            with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"{row['condition']} {item['item_id']}: {row['status']}", flush=True)
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["condition"]].append(row)
    report = {k: summarize(v) for k,v in grouped.items()}
    if baseline is not None:
        for key, rows in grouped.items():
            report[key]["paired_comparison"] = paired_comparison(rows, baseline)
    write_json(output / "metrics.json", report)
    from project_data import read_json
    run = read_json(output / "run.json")
    write_json(output / "run.json", {**run, "status": "complete", "failures": sum(r["status"] != "ok" for r in all_rows)})
    print(report)


if __name__ == "__main__":
    main()
