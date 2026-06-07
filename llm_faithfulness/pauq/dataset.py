import json
import os
from typing import Iterator

from ..types import EvalRecord
from . import prompts, sql_utils
from .types import PAUQEntry, PAUQMediator


def _build_db_schema(db: dict) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    for i, table_name in enumerate(db["table_names_original"]):
        cols = []
        for entry in db["column_names_original"][1:]:
            t_idx, col_name = entry
            if t_idx == i:
                cols.append(col_name.lower())
        schema[table_name.lower()] = cols
    return schema


class PAUQDataset:
    name: str = "pauq"
    section_markers: list[str] = [
        "===SKELETON===",
        "===SCHEMA_LINKS===",
        "===SLOT_MATCHING===",
        "===SQL===",
    ]

    def __init__(self, data_path: str, split: str = "dev"):
        if split not in {"train", "dev"}:
            raise ValueError(f"split must be 'train' or 'dev', got {split!r}")

        with open(os.path.join(data_path, f"pauq_{split}.json")) as f:
            raw = json.load(f)
        with open(os.path.join(data_path, "tables.json")) as f:
            tables_raw = json.load(f)

        self.databases: dict[str, dict] = {db["db_id"]: db for db in tables_raw}

        self.entries: list[PAUQEntry] = []
        for idx, sample in enumerate(raw):
            db = self.databases[sample["db_id"]]
            db_schema = _build_db_schema(db)
            sql = sample["query"]["en"]
            question = sample["question"]["en"]

            parsed = sql_utils.parse_sql(sql, db_schema)
            schema_links = sql_utils.extract_schema_links(parsed)
            if not schema_links:
                continue

            try:
                skeleton, slots = sql_utils.extract_skeleton_and_slots(sql, db_schema)
            except Exception:
                continue

            self.entries.append(
                PAUQEntry(
                    index=idx,
                    question=question,
                    gold_sql=sql,
                    db_id=sample["db_id"],
                    db_schema=db_schema,
                    gold_mediator=PAUQMediator(skeleton=skeleton, schema_links=schema_links, slots=slots),
                )
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PAUQEntry]:
        return iter(self.entries)

    def build_prompt(self, entry: PAUQEntry, *, gold_structure: bool) -> str:
        user = prompts.build_user_prompt(entry.question, entry.db_schema)
        if not gold_structure:
            return user
        return user + prompts.render_mediator_prefix(entry.gold_mediator)

    def build_continuation_prompt(self, entry: PAUQEntry, mediator: PAUQMediator) -> str:
        return prompts.build_user_prompt(entry.question, entry.db_schema) + prompts.render_mediator_prefix(mediator)

    def parse_completion(self, text: str) -> tuple[PAUQMediator | None, str | None]:
        parsed = sql_utils.parse_model_response(text)
        sql = parsed["sql"] or None
        if parsed["skeleton"] is None and not parsed["schema_links"] and not parsed["slots"]:
            return None, sql
        med = PAUQMediator(
            skeleton=parsed["skeleton"] or "",
            schema_links=parsed["schema_links"],
            slots=parsed["slots"],
        )
        return med, sql

    def render_mediator(self, mediator: PAUQMediator, answer: str | None = None) -> str:
        return prompts.render_mediator(mediator, sql=answer)

    def gold_mediator(self, entry: PAUQEntry) -> PAUQMediator:
        return entry.gold_mediator

    def gold_answer(self, entry: PAUQEntry) -> str:
        return entry.gold_sql

    def score(self, entry: PAUQEntry, record: EvalRecord) -> dict[str, bool | None]:
        f_id: bool | None = None
        f_strong: bool | None = None
        perf: bool | None = None

        if record.predicted_mediator is not None and record.predicted_answer:
            f_id = sql_utils.faithfulness_id_check(
                record.predicted_mediator, record.predicted_answer, entry.db_schema
            )

        if record.intervention_level == 0:
            f_strong = f_id
        elif record.intervened_mediator is not None and record.intervened_answer:
            f_strong = (
                bool(f_id)
                and sql_utils.faithfulness_id_check(
                    record.intervened_mediator, record.intervened_answer, entry.db_schema
                )
            )

        if record.predicted_answer:
            perf = sql_utils.validate_sql(entry.gold_sql, record.predicted_answer, entry.db_schema)

        return {"f_id": f_id, "f_strong": f_strong, "perf": perf}
