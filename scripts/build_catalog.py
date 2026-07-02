"""Turns raw scraped rows (name, url, test_type letters) into the
normalized catalog.json the app actually reads.

Raw rows look like:
    {"name": "SQL (New)", "url": "https://...", "test_type": ["K"]}

This adds a stable id and an expanded description used for embedding -
matters more for retrieval quality than the bare name alone. Defaults job
levels / remote testing / duration when the lightweight scrape doesn't
have them; if you ran scrape_catalog.py with detail-page enrichment those
get passed through instead.

    python scripts/build_catalog.py --input scripts/seed_catalog_raw.json --output app/catalog/catalog.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def slugify(url: str) -> str:
    m = re.search(r"/view/([^/]+)/?$", url.rstrip("/"))
    return m.group(1) if m else re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")


def synthesize_description(name: str, test_types: list[str]) -> str:
    # plain, template-based description for embedding purposes when a real
    # scraped one isn't available - never copied from any webpage
    clean_name = re.sub(r"\s*\(New\)\s*", "", name).strip()
    labels = [TEST_TYPE_LABELS[t] for t in test_types if t in TEST_TYPE_LABELS]
    label_text = " / ".join(labels) if labels else "Assessment"

    if "K" in test_types and len(test_types) == 1:
        return f"{clean_name} knowledge and skills test. Multiple-choice assessment of {clean_name} proficiency for technical hiring."
    if "S" in test_types:
        return f"{clean_name} simulation. Hands-on, job-realistic simulation exercise assessing practical {clean_name} skills."
    if "P" in test_types:
        return f"{clean_name}. Personality and behavioral assessment measuring workplace preferences, style, and role fit."
    if "A" in test_types:
        return f"{clean_name}. Cognitive ability / aptitude test measuring reasoning, problem-solving, or numerical/verbal ability."
    if "D" in test_types:
        return f"{clean_name}. Development and 360-degree feedback report used for growth planning and manager insight."
    if "B" in test_types:
        return f"{clean_name}. Biodata or situational judgement assessment predicting on-the-job behavior from past experience and scenario responses."
    if "C" in test_types:
        return f"{clean_name}. Competency-based assessment mapped to the Universal Competency Framework."
    if "E" in test_types:
        return f"{clean_name}. Assessment center exercise used in structured evaluation and development centers."
    return f"{clean_name}. {label_text} assessment from the SHL catalog."


def normalize(raw: dict) -> dict:
    test_types = raw.get("test_type", [])
    name = raw["name"]
    return {
        "id": slugify(raw["url"]),
        "name": name,
        "url": raw["url"],
        "test_type": test_types,
        "description": raw.get("description") or synthesize_description(name, test_types),
        "job_levels": raw.get("job_levels", []),
        "remote_testing": raw.get("remote_testing", False),
        "adaptive_irt": raw.get("adaptive_irt", False),
        "duration_minutes": raw.get("duration_minutes"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_items = json.loads(Path(args.input).read_text())

    seen_ids = set()
    normalized = []
    for raw in raw_items:
        item = normalize(raw)
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        normalized.append(item)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(normalized, indent=2))
    print(f"wrote {len(normalized)} catalog items to {args.output}")


if __name__ == "__main__":
    main()
