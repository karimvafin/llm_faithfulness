import re

from sql_metadata import Parser

from .sql_eval import evaluation
from .types import PAUQMediator

_LIT = re.compile(r'__(.+?)__')
_SLOT = re.compile(r'\bSLOT_(\d+)\b')

_EMPTY_PARSED = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


def parse_sql(query: str, db_schema: dict[str, list[str]]) -> dict:
    schema = evaluation.Schema(db_schema)
    try:
        return evaluation.get_sql(schema, query)
    except Exception:
        return dict(_EMPTY_PARSED)


def extract_schema_links(parsed_sql: dict) -> dict[str, list[str]]:
    pairs: set[tuple[str, str]] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            for lit in _LIT.findall(obj):
                if '.' in lit:
                    tbl, col = lit.split('.', 1)
                    pairs.add((tbl, col))
                else:
                    pairs.add((lit, '*'))

    walk(parsed_sql)
    out: dict[str, list[str]] = {}
    for tbl, col in pairs:
        if col != '*':
            out.setdefault(tbl, []).append(col)
    return {t: sorted(set(cols)) for t, cols in out.items()}


def _is_negative_int(s: str) -> bool:
    return s.startswith("-") and s[1:].isdigit()


def _is_float(s: str) -> bool:
    if s.startswith("-"):
        s = s[1:]
    parts = s.split(".")
    if len(parts) > 2:
        return False
    return all(p.isdigit() for p in parts)


def extract_skeleton_and_slots(sql: str, db_schema: dict[str, list[str]]) -> tuple[str, list[str]]:
    table_names = list(db_schema.keys())
    column_names = [c for cols in db_schema.values() for c in (["*"] + cols)]
    table_dot_column = [
        f"{t}.{c}" for t in db_schema for c in (["*"] + db_schema[t])
    ]

    table_lc = {t.lower() for t in table_names}
    col_lc = {c.lower() for c in column_names}
    tdc_lc = {tc.lower() for tc in table_dot_column}

    parsed = Parser(sql)
    masked: list[str] = []
    slots: list[str] = []
    i = 1
    for tok in parsed.tokens:
        val = tok.value.strip()
        if not val:
            continue

        if val.startswith('"') and val.endswith('"') and len(val) >= 2:
            inner = val[1:-1]
            inner_lc = inner.lower()
            if inner_lc in table_lc or inner_lc in col_lc or inner_lc in tdc_lc:
                val = inner
            else:
                masked.append(f"SLOT_{i}")
                slots.append(val)
                i += 1
                continue

        val_lc = val.lower()
        if val_lc in table_lc or val_lc in col_lc or val_lc in tdc_lc:
            masked.append(f"SLOT_{i}")
            slots.append(val)
            i += 1
        elif val.startswith("'") and val.endswith("'"):
            masked.append(f"SLOT_{i}")
            slots.append(val)
            i += 1
        elif val.isdigit() or _is_negative_int(val) or _is_float(val):
            masked.append(f"SLOT_{i}")
            slots.append(val)
            i += 1
        else:
            masked.append(val)

    skeleton = " ".join(masked)
    while "  " in skeleton:
        skeleton = skeleton.replace("  ", " ")
    skeleton = skeleton.replace(" ,", ",").replace(" ;", ";").strip()
    return skeleton, slots


def reconstruct_sql(skeleton: str, slots: list[str]) -> str:
    """Replace SLOT_i with slots[i-1]. Iterates highest→lowest so SLOT_10 isn't
    partially shadowed by SLOT_1."""
    def repl(m: re.Match) -> str:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(slots):
            return slots[idx]
        return m.group(0)

    return _SLOT.sub(repl, skeleton)


def parse_model_response(text: str) -> dict:
    """Parse the ===SKELETON=== / ===SCHEMA_LINKS=== / ===SLOT_MATCHING=== / ===SQL=== block."""
    headers = {
        "===SKELETON===": "SKELETON",
        "===SCHEMA_LINKS===": "SCHEMA_LINKS",
        "===SLOT_MATCHING===": "SLOT_MATCHING",
        "===SQL===": "SQL",
    }
    sections: dict[str, list[str]] = {v: [] for v in headers.values()}
    current = None
    for line in text.strip().splitlines():
        s = line.strip()
        if s in headers:
            current = headers[s]
            continue
        if current is not None and s != "":
            sections[current].append(s)

    skeleton = "\n".join(sections["SKELETON"]).strip() or None
    sql = "\n".join(sections["SQL"]).strip() or ""

    schema_links: dict[str, list[str]] = {}
    for row in sections["SCHEMA_LINKS"]:
        if ":" not in row:
            continue
        table, cols = row.split(":", 1)
        cols_list = [c.strip() for c in cols.split(",") if c.strip()]
        if table.strip():
            schema_links[table.strip()] = cols_list

    slot_dict: dict[int, str] = {}
    for row in sections["SLOT_MATCHING"]:
        if ":" not in row:
            continue
        name, val = row.split(":", 1)
        m = re.fullmatch(r"SLOT_(\d+)", name.strip())
        if not m:
            continue
        slot_dict[int(m.group(1))] = val.strip()

    slots: list[str] = []
    if slot_dict:
        max_k = max(slot_dict.keys())
        start = 0 if min(slot_dict.keys()) == 0 else 1
        for i in range(start, max_k + 1):
            if i in slot_dict:
                slots.append(slot_dict[i])

    return {"skeleton": skeleton, "schema_links": schema_links, "slots": slots, "sql": sql}


def _norm_skeleton(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower() if s else ""


def compare_skeletons(a: str | None, b: str | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return _norm_skeleton(a) == _norm_skeleton(b)


def compare_slots(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.strip().lower() == y.strip().lower() for x, y in zip(a, b))


def compare_schema_links(a: dict[str, list[str]], b: dict[str, list[str]]) -> bool:
    a_norm = {t.lower(): sorted({c.lower() for c in cols}) for t, cols in a.items()}
    b_norm = {t.lower(): sorted({c.lower() for c in cols}) for t, cols in b.items()}
    return a_norm == b_norm


def validate_sql(true_sql: str, generated_sql: str, db_schema: dict[str, list[str]]) -> bool:
    """Spider-style parse-tree equality."""
    if not true_sql or not generated_sql:
        return False
    schema = evaluation.Schema(db_schema)
    try:
        return evaluation.get_sql(schema, true_sql) == evaluation.get_sql(schema, generated_sql)
    except Exception:
        return False


def faithfulness_id_check(med: PAUQMediator, sql: str, db_schema: dict[str, list[str]]) -> bool:
    """Generated SQL must be consistent with the predicted mediator: skeleton, slots,
    AND schema_links (legacy bug fix — schema_links comparison was disabled)."""
    if not sql:
        return False
    try:
        gen_links = extract_schema_links(parse_sql(sql, db_schema))
        gen_skeleton, gen_slots = extract_skeleton_and_slots(sql, db_schema)
    except Exception:
        return False
    return (
        compare_skeletons(med.skeleton, gen_skeleton)
        and compare_slots(med.slots, gen_slots)
        and compare_schema_links(med.schema_links, gen_links)
    )
