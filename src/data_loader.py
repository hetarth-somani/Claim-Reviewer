import csv
from pathlib import Path

def load_claims(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [{k.strip("\ufeff").strip(): v for k, v in r.items()} for r in rows]
    rows = [r for r in rows if any(v.strip() for v in r.values() if v)]
    return rows


def load_user_history(csv_path: Path) -> dict[str, dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {row["user_id"]: row for row in csv.DictReader(fh)}


def load_evidence_requirements(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def filter_evidence_reqs(reqs: list[dict], claim_object: str) -> list[dict]:
    return [r for r in reqs if r.get("claim_object") in (claim_object, "all")]
