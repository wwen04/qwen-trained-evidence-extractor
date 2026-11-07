"""Train only the selector. Default execution is a CPU/data-only preflight.

No optimizer is constructed or stepped unless --train is explicitly supplied.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import random

from project_data import ROOT, file_hash, load_questions, load_references, read_json, shuffle_options, stable_seed, write_json


def validate_config(config):
    for key in ("epochs", "gradient_accumulation", "null_init_images", "eval_every_updates"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["target_rho"] <= 1:
        raise ValueError("target_rho must be in (0,1]")
    for key in ("warmup_fraction", "budget_anneal_fraction", "straight_through_fraction"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
    if config["learning_rate"] <= 0 or config["budget_weight"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if config["selector"].get("variant") == "free_transform" and (config["budget_weight"] != 0 or config["straight_through_fraction"] != 0):
        raise ValueError("Free-transform control requires budget_weight=0 and straight_through_fraction=0")
    if config.get("selector_checkpoint"):
        raise ValueError("Use --resume for training continuation, not selector_checkpoint in the training config")


def preflight(config, limit=None):
    validate_config(config)
    train = load_questions("scienceqa", "train")
    validation = load_questions("scienceqa", "validation")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        train = train[:limit]
    if not train or not validation:
        raise ValueError("Training and validation selections must be nonempty")
    refs = load_references("scienceqa", "train")
    for item in train:
        if refs[item["item_id"]]["gold_option_id"] not in {c["id"] for c in item["choices"]}:
            raise ValueError("Invalid training answer label")
    counts = Counter(refs[x["item_id"]]["gold_option_id"] for x in train)
    return train, validation, {"training_questions": len(train), "validation_questions": len(validation),
        "test_questions_loaded": 0, "gold_letter_counts_before_shuffle": dict(counts),
        "optimizer_updates_planned": math.ceil(len(train) / config["gradient_accumulation"]) * config["epochs"],
        "split_manifest_sha256": file_hash(ROOT / "manifests/selector/splits.json"),
        "mode": "preflight only; no model loaded or training run"}


def validate(runtime, items, references):
    import torch
    from selector import answer_loss
    runtime.selector.eval()
    runtime.model.eval()
    total_ce, correct, gates_total = 0.0, 0, 0.0
    with torch.no_grad():
        for item in items:
            logits, gates = runtime.answer_forward(item, mode="soft")
            gold = ord(references[item["item_id"]]["gold_option_id"]) - 65
            ce, restricted = answer_loss(logits, runtime.letter_ids[:len(item["choices"])], gold)
            total_ce += float(ce)
            correct += int(restricted.argmax(-1).item() == gold)
            gates_total += float(gates.mean())
    runtime.selector.train()
    if runtime.config.get("gradient_checkpointing", True):
        runtime.model.model.language_model.train()
    return {"questions": len(items), "accuracy": correct / len(items), "mean_ce": total_ce / len(items),
            "mean_gate": gates_total / len(items)}


def run_training(config, train, validation, report, *, resume=False):
    import torch
    from checkpoints import save_checkpoint, load_checkpoint
    from model_runtime import QwenSelectorRuntime
    from selector import answer_loss, budget_target

    output = Path(config["output_dir"])
    output = output if output.is_absolute() else ROOT / output
    run_config = {**config, "train_item_ids": [r["item_id"] for r in train],
                  "validation_item_ids": [r["item_id"] for r in validation],
                  "split_manifest_sha256": report["split_manifest_sha256"]}
    if resume:
        prior = read_json(output / "run.json")
        if prior["config"] != run_config:
            raise ValueError("Resume configuration, inputs or split provenance differ from the saved run")
        if prior["status"] == "complete":
            raise ValueError("This run is already complete; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run.json", {"status": "initializing", "config": run_config, "preflight": report})
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    runtime_config = {**config, "gate_mode": "soft", "rho": config["target_rho"]}
    if resume:
        from checkpoints import checkpoint_metadata
        prior_metadata = checkpoint_metadata(output / "last")
        runtime_config['revision'] = prior_metadata['backbone'].get('resolved_revision') or config.get('revision')
    runtime = QwenSelectorRuntime(runtime_config, training=True)
    identity = runtime.identity_check(train[0])
    letter_probe = runtime.letter_probe(train[0])
    refs = load_references("scienceqa", "train")
    validation_refs = load_references("scienceqa", "validation")
    total_updates = report["optimizer_updates_planned"]
    optimizer = torch.optim.AdamW([p for p in runtime.selector.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    warmup = math.ceil(total_updates * config["warmup_fraction"])
    def schedule(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, start_index, updates, best = 0, 0, 0, None
    if resume:
        metadata = load_checkpoint(output / "last", runtime.selector, runtime.model.config)
        # Only tensor/primitives optimizer state is accepted. Resume only your own run.
        state = torch.load(output / "last/optimizer.pt", map_location=runtime.device, weights_only=True)
        if state["training_updates"] != metadata["training_updates"]:
            raise ValueError("Interrupted checkpoint save; selector/optimizer steps differ")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_index, updates, best = state["epoch"], state["index"], state["training_updates"], state["best"]
    else:
        # Only training images participate in mean-null initialization.
        sample = train[:min(len(train), config["null_init_images"])]
        mean_sum, token_count = None, 0
        for item in sample:
            inputs, _ = runtime.prepare(item, "Read the diagram.")
            visual = runtime.visual_features(inputs)
            summed = visual.float().sum(0)
            mean_sum = summed if mean_sum is None else mean_sum + summed
            token_count += len(visual)
        runtime.selector.initialize_null(mean_sum / token_count)
        write_json(output / "initialization.json", {"identity_check": identity, "letter_probe":letter_probe,
                   "null_training_items": [x["item_id"] for x in sample],
                   "trainable_parameters": sum(p.numel() for p in runtime.selector.parameters() if p.requires_grad),
                   "frozen_backbone_parameters": sum(p.numel() for p in runtime.model.parameters())})
    provenance = {**runtime.provenance(), "target_rho": config["target_rho"],
                  "split_manifest_sha256": report["split_manifest_sha256"], "train_config": config}
    write_json(output / "run.json", {"status": "running", "config": run_config, "preflight": report})
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, config["epochs"]):
            order = list(range(len(train)))
            random.Random(stable_seed(config["seed"], "order", epoch)).shuffle(order)
            offset = start_index if epoch == start_epoch else 0
            while offset < len(order):
                group = order[offset:offset + config["gradient_accumulation"]]
                rho = budget_target(updates, total_updates, config["target_rho"], config["budget_anneal_fraction"])
                use_st = config["straight_through_fraction"] > 0 and updates >= total_updates * (1-config["straight_through_fraction"])
                ce_sum, penalty_sum, gate_sum = 0.0, 0.0, 0.0
                for index in group:
                    original = train[index]
                    gold_id = refs[original["item_id"]]["gold_option_id"]
                    if config["shuffle_options"]:
                        item, gold = shuffle_options(original, gold_id, stable_seed(config["seed"], epoch, original["item_id"]))
                    else:
                        item, gold = original, ord(gold_id)-65
                    logits, gates = runtime.answer_forward(item, mode="straight_through" if use_st else "soft", rho=rho)
                    ce, _ = answer_loss(logits, runtime.letter_ids[:len(item["choices"])], gold)
                    penalty = torch.relu(gates.float().mean() - rho)
                    loss = (ce + config["budget_weight"] * penalty) / len(group)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    ce_sum += float(ce.detach()); penalty_sum += float(penalty.detach()); gate_sum += float(gates.detach().mean())
                if any(p.grad is not None for p in runtime.model.parameters()):
                    raise RuntimeError("Backbone unexpectedly received parameter gradients")
                norm = torch.nn.utils.clip_grad_norm_(runtime.selector.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                if not any(p.grad is not None for p in runtime.selector.parameters()):
                    raise RuntimeError("Selector gradient path is disconnected")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                offset += len(group)
                log = {"update": updates, "epoch": epoch+1, "ce": ce_sum/len(group), "budget_penalty": penalty_sum/len(group),
                       "mean_gate": gate_sum/len(group), "rho": rho, "gradient_norm": float(norm), "straight_through": use_st}
                if updates % config["eval_every_updates"] == 0 or offset == len(order):
                    metrics = validate(runtime, validation, validation_refs)
                    log["validation"] = metrics
                    if best is None or (metrics["accuracy"], -metrics["mean_ce"]) > (best["accuracy"], -best["mean_ce"]):
                        best = metrics
                        save_checkpoint(output / "best", runtime.selector, {**provenance, "training_updates": updates, "validation": metrics})
                next_epoch, next_index = (epoch+1, 0) if offset == len(order) else (epoch, offset)
                save_checkpoint(output / "last", runtime.selector, {**provenance, "training_updates": updates, "validation_best": best})
                temporary = output / "last/optimizer.pt.partial"
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": next_epoch,
                            "index": next_index, "training_updates": updates, "best": best}, temporary)
                temporary.replace(output / "last/optimizer.pt")
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    import json
                    stream.write(json.dumps(log, allow_nan=False) + "\n")
                print(log, flush=True)
    except BaseException as error:
        write_json(output / "run.json", {"status": "interrupted", "config": run_config, "preflight": report,
                   "last_completed_update": updates, "error": f"{type(error).__name__}: {error}"})
        raise
    write_json(output / "run.json", {"status": "complete", "config": run_config, "preflight": report,
               "training_updates": updates, "best_validation": best})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_selector.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--check-model", action="store_true", help="Load Qwen and verify identity; never train")
    action.add_argument("--train", action="store_true", help="Explicitly start optimizer updates")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Training-only pilot subset; validation remains unchanged")
    parser.add_argument("--output-dir")
    parser.add_argument("--rho", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--no-option-shuffle", action="store_true", help="B5 control only")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    for key in ("output_dir", "seed", "epochs", "model_path"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.rho is not None:
        config["target_rho"] = args.rho
    if args.no_option_shuffle:
        config["shuffle_options"] = False
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    train, validation, report = preflight(config, args.limit)
    print(report, flush=True)
    if args.check_model:
        from model_runtime import QwenSelectorRuntime
        runtime = QwenSelectorRuntime({**config, "gate_mode": "identity", "rho": 1.0})
        print({"identity": runtime.identity_check(train[0]), "letter_token_ids": runtime.letter_ids,
               "letter_probe":runtime.letter_probe(train[0])})
    elif args.train:
        run_training(config, train, validation, report, resume=args.resume)


if __name__ == "__main__":
    main()
