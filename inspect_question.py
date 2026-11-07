"""Inspect an audited ScienceQA item without loading a model or changing data."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import webbrowser

from scienceqa_tutor import dataset_root, load_public_questions


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_item(item_id, root=ROOT, data_source="final"):
    if not re.fullmatch(r"(train|validation|test)_\d{6}", item_id):
        raise ValueError("Use an ID such as validation_000008.")
    split = item_id.rsplit("_", 1)[0]
    base = dataset_root(root, data_source)
    questions = load_public_questions(root, split, data_source)
    item = next((row for row in questions if row["item_id"] == item_id), None)
    if item is None:
        raise ValueError(f"{item_id} is not in the {data_source} dataset. Use --source processed to inspect an excluded candidate.")
    if item["split"] != split:
        raise ValueError("The stored split does not match the item ID.")
    image = (root / item["image_path"]).resolve()
    if not image.is_relative_to(root.resolve()) or not image.is_file():
        raise ValueError("The prepared image is missing or outside the project.")
    if data_source == "final" and not image.is_relative_to((base / "images").resolve()):
        raise ValueError("Final questions must use audited images under data/final/scienceqa/images")
    references = read_rows(base / "references" / f"{split}.jsonl")
    reference = next((row for row in references if row["item_id"] == item_id), {})
    audit_file = root / "reviews/scienceqa/annotations.jsonl"
    audit = next((row for row in read_rows(audit_file) if row["item_id"] == item_id), {}) if audit_file.exists() else {}
    if audit:
        # Finalization changes only the image path; audit fingerprints describe the source record.
        source_item = dict(item)
        if data_source == "final":
            source_item["image_path"] = f"data/processed/scienceqa/images/{item_id}.png"
        digest = hashlib.sha256(json.dumps(source_item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        image_digest = hashlib.sha256(image.read_bytes()).hexdigest()
        audit = {**audit, "source_status": "matches reviewed source" if digest == audit.get("source_row_sha256") and image_digest == audit.get("image_sha256") else "STALE: source changed since this review"}
    return item, reference, {**audit, "data_source": data_source}, image


def render(item, reference, audit, image, output):
    """Generate an inspection document, never used as a tutoring prompt."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        image_link = Path(os.path.relpath(image, output.parent)).as_posix().replace(" ", "%20")
    except ValueError:  # A user-selected output can be on another Windows drive.
        image_link = image.as_uri()
    choice_lines = [f"{c['id']}. {c['text']}" for c in item["choices"]]
    sections = [
        ("Question", item["question"]),
        ("Supplied context", item.get("context") or "(none)"),
        ("Choices", "\n".join(choice_lines)),
        ("Metadata", "\n".join(f"{key}: {value}" for key, value in item.get("metadata", {}).items())),
        ("Audit classification", audit.get("visual_need", "pending")),
        ("Audit reason", audit.get("rationale") or "(not recorded)"),
        ("Necessary visual evidence", audit.get("necessary_evidence") or "(not recorded)"),
        ("Audit source check", audit.get("source_status", "(not reviewed)")),
        ("Reference answer", reference.get("gold_option_id", "(unavailable)")),
        ("Reference lecture", reference.get("lecture") or "(none)"),
        ("Reference solution", reference.get("solution") or "(none)"),
    ]
    title = f"ScienceQA — {item['item_id']}"
    lines = [f"# {title}", "", f"Official split: {item['split']} · Dataset: {audit.get('data_source', 'final')}", ""]
    for name, value in sections[:3]:
        lines.extend([f"## {name}", "", value, ""])
    lines.extend(["## Image", "", f"![Question image](<{image_link}>)", "", f"Source: `{item['image_path']}`", ""])
    for name, value in sections[3:]:
        lines.extend([f"## {name}", "", value, ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    esc = html.escape
    def section(name, value):
        return f"<section><h2>{esc(name)}</h2><div class='text'>{esc(value)}</div></section>"
    body = "".join(section(*entry) for entry in sections[:3])
    body += f"<section><h2>Image</h2><a href='{esc(image_link, quote=True)}'><img src='{esc(image_link, quote=True)}' alt='Question image'></a><p>{esc(item['image_path'])}</p></section>"
    body += "".join(section(*entry) for entry in sections[3:8])
    body += "<details><summary>Reference answer, lecture and solution</summary>" + "".join(section(*entry) for entry in sections[8:]) + "</details>"
    html_path = output.with_suffix(".html")
    html_path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>body{{font:17px/1.55 system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 24px;color:#172332;background:#f7f8fb}}"
        "section,details{background:white;padding:20px 26px;margin:18px 0;border:1px solid #dce2e9;border-radius:10px}h2{font-size:19px;margin-top:0}.text{white-space:pre-wrap;overflow-wrap:anywhere}"
        "img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}p{overflow-wrap:anywhere}</style>"
        f"<h1>{esc(title)}</h1><p>Official split: {esc(item['split'])} · Dataset: {esc(audit.get('data_source', 'final'))} · Read-only dataset inspection</p>{body}</html>", encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item_id", nargs="?", help="Example: validation_000008; prompts if omitted")
    parser.add_argument("--source", choices=("final", "processed"), default="final", help="Default: audited final set. Use processed only to inspect excluded audit candidates.")
    parser.add_argument("--list", choices=SPLITS, metavar="SPLIT", help="List available IDs and question text")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inspection/current_question.md", help="Markdown output path; a matching HTML page is also generated")
    parser.add_argument("--open", action="store_true", help="Open the HTML inspection page in your browser")
    args = parser.parse_args()
    try:
        if args.list:
            for row in load_public_questions(ROOT, args.list, args.source):
                print(f"{row['item_id']}  {row['question'].replace(chr(10), ' ')}")
            if not args.item_id:
                return
        selected_id = args.item_id or input("Question ID (e.g. validation_000008): ").strip()
        item, reference, audit, image = load_item(selected_id, data_source=args.source)
        output = args.output.resolve()
        if output.suffix.lower() != ".md":
            raise ValueError("--output must end in .md")
        for protected in ("data", "manifests", "reviews", "configs"):
            if output.is_relative_to(ROOT / protected):
                raise ValueError("Choose an inspection output outside data, manifests, reviews and configs.")
        html_path = render(item, reference, audit, image, output)
        print(f"Markdown: {output}\nBrowser page: {html_path}")
        if args.open:
            webbrowser.open(html_path.as_uri())
    except (ValueError, OSError, EOFError) as error:
        parser.exit(1, f"Inspection failed: {error}\n")


if __name__ == "__main__":
    main()
