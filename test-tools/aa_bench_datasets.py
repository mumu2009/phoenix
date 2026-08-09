"""Dataset loaders for aa_test benchmarks (C-Eval, CMMLU, etc.).

All datasets are expected under build/tmp/<dataset>/<split>/ as one CSV per subject.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _csv_paths(dir_: Path):
    return sorted(dir_.glob("*.csv")) if dir_.exists() else []


def _norm(subject: str, row: dict) -> dict:
    """Normalize column names across different dataset CSV schemas."""
    def get(*keys):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return ""

    # C-Eval: id,question,A,B,C,D,answer
    # CMMLU: <index>,Question,A,B,C,D,Answer
    question = get("question", "Question")
    a = get("A")
    b = get("B")
    c = get("C")
    d = get("D")
    answer = get("answer", "Answer")
    qid = get("id", "")
    if not qid:
        # For CMMLU, the first (unnamed) column is the index.
        for k in row.keys():
            if k is None or k == "":
                qid = str(row[k])
                break
    if not qid:
        qid = "0"
    return {
        "subject": subject,
        "id": str(qid),
        "question": question,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "answer": str(answer).strip().upper(),
    }


def _load_csvs(dataset_dir: Path, per_subject: int):
    questions = []
    for csv_path in _csv_paths(dataset_dir):
        subject = csv_path.stem
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        end = per_subject if per_subject > 0 else len(rows)
        for row in rows[:end]:
            questions.append(_norm(subject, row))
    return questions


DATASET_SPECS = {
    "ceval": {
        "dir": ROOT / "build" / "tmp" / "ceval" / "val",
        "split": "val",
    },
    "cmmlu": {
        "dir": ROOT / "build" / "tmp" / "cmmlu" / "test",
        "split": "test",
    },
}


def load_dataset(name: str, per_subject: int = 1):
    name = name.lower().strip()
    spec = DATASET_SPECS.get(name)
    if not spec:
        raise FileNotFoundError(f"Unknown benchmark dataset: {name}. Known: {list(DATASET_SPECS)}")
    dir_ = spec["dir"]
    if not dir_.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dir_}")
    return _load_csvs(dir_, per_subject)


def list_datasets():
    return [k for k, v in DATASET_SPECS.items() if v["dir"].exists()]
