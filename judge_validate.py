"""Validate individual seven-field Part A judgments; no machine-specific paths."""

import argparse
import json
from pathlib import Path


KEYS = {"tier_compliance", "visual_grounding", "unsupported_visual_claims", "scientific_correctness",
        "steers_toward_wrong_answer", "answer_disclosure", "actionability"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def read_judgment(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("seven top-level fields required")
    for key in ("tier_compliance", "visual_grounding", "scientific_correctness", "steers_toward_wrong_answer"):
        item = value[key]
        if (not isinstance(item, dict) or set(item) != {"reason", "verdict"}
                or not isinstance(item["reason"], str) or not item["reason"].strip()
                or item["verdict"] not in ("yes", "no")):
            raise ValueError(f"Invalid {key}")
    item = value["unsupported_visual_claims"]
    if (not isinstance(item, dict) or set(item) != {"unsupported_claims", "count"}
            or not isinstance(item["unsupported_claims"], list)
            or any(not isinstance(text, str) or not text.strip() for text in item["unsupported_claims"])
            or type(item["count"]) is not int or item["count"] < 0
            or item["count"] != len(item["unsupported_claims"])):
        raise ValueError("Invalid unsupported visual claim count/list")
    for key, scores in (("answer_disclosure", range(3)), ("actionability", range(1, 6))):
        item = value[key]
        if (not isinstance(item, dict) or set(item) != {"reason", "score"}
                or not isinstance(item["reason"], str) or not item["reason"].strip()
                or type(item["score"]) is not int or item["score"] not in scores):
            raise ValueError(f"Invalid {key}")
    return value


def validate(path):
    try:
        read_judgment(path)
        return None
    except (ValueError, OSError, KeyError) as error:
        return str(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Judgment JSON files or folders")
    args = parser.parse_args(argv)
    paths = [path for argument in args.files for path in
             (sorted(argument.glob("judgment_*.json")) if argument.is_dir() else [argument])]
    if not paths:
        parser.error("No judgment JSON files found")
    errors = {str(path): error for path in paths if (error := validate(path)) is not None}
    print(json.dumps({"total": len(paths), "valid": len(paths) - len(errors), "invalid": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
