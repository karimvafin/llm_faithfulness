import json
from dataclasses import asdict
from typing import Any

from .protocols import Dataset
from .types import EvalRecord, MetricsReport, Mode


def _mean_bool(values: list[bool | None]) -> float:
    filtered = [int(v) for v in values if v is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def score_records(dataset: Dataset, entries_by_index: dict[int, Any], records: list[EvalRecord]) -> None:
    """Mutates records in place — fills f_id, f_strong, perf using the dataset's score()."""
    for rec in records:
        entry = entries_by_index[rec.index]
        result = dataset.score(entry, rec)
        rec.f_id = result.get("f_id")
        rec.f_strong = result.get("f_strong")
        rec.perf = result.get("perf")


def aggregate(
    records: list[EvalRecord],
    *,
    model: str,
    dataset_name: str,
    mode: Mode,
    intervention_level: int,
) -> MetricsReport:
    return MetricsReport(
        model=model,
        dataset=dataset_name,
        mode=mode,
        intervention_level=intervention_level,
        n=len(records),
        faithfulness_id=_mean_bool([r.f_id for r in records]),
        faithfulness_strong=_mean_bool([r.f_strong for r in records]),
        performance=_mean_bool([r.perf for r in records]),
        records=records,
    )


def save_report(report: MetricsReport, path: str) -> None:
    payload = asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str, ensure_ascii=False, indent=2)


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
