from .types import PAUQMediator

_INSTRUCTIONS = """You are an expert in natural language understanding and SQL queries generation.
Given a natural language question and a database schema, perform 4 steps:

1. SQL Skeleton Generation: Derive the abstract syntactic structure of the SQL query that answers the question. Replace all concrete identifiers (table names, column names, literals, etc.) with generic placeholders SLOT_1, SLOT_2, ..., in the order they appear. The skeleton must be valid SQL syntax except for the use of SLOT_* tokens.
2. Schema Linking: Based on the question identify which tables and columns in the schema are relevant to answer the question.
3. Slot Matching: For each SLOT_i in the skeleton, map it to the exact identifier from the provided schema or a literal value derived from the question. Use only:
Column names (e.g., capacity)
Table names (e.g., stadium)
Numeric or string literals (e.g., 56, '2020%')
Do not infer semantics; match based on the question intent and schema content.
4. SQL Generation: Substitute all SLOT_i tokens in the skeleton with their matched identifiers to produce a syntactically correct, executable SQL query. The query must:
- Use only tables and columns from the given schema
- Answer the question exactly
Be written in standard SQL (no markdown, no comments, no extra text)

Output Format
Return only the following, with no additional explanations or formatting:

Output:
===SKELETON===
[SQL skeleton with SLOT tokens]
===SCHEMA_LINKS===
table1:col1,col2
table2:col1,col2
...
===SLOT_MATCHING===
SLOT_1:[value]
SLOT_2:[value]
...
===SQL===
[final executable SQL query]

Rules:
- Never include real identifiers in the skeleton.
- Every SLOT_i must appear in both SKELETON and SLOT_MATCHING.
- Write a valid SQL query string without markdown, without extra text
- Only output in the specified format. Do not add explanations.
- Do not pay attention to the name semantics of columns and tables, rely more on slot matching.
- String literals must be enclosed in single quotes; numeric literals must not be quoted.

Few-Shot Examples
Example 1
Question: "How many heads of the departments are older than 56?"

Schema:
Table: head
Columns: head_id, name, age
Table: department
Columns: dept_id, name, budget

Output:
===SKELETON===
SELECT COUNT(SLOT_1) FROM SLOT_2 WHERE SLOT_3 > SLOT_4;
===SCHEMA_LINKS===
head:age
===SLOT_MATCHING===
SLOT_1:*
SLOT_2:head
SLOT_3:age
SLOT_4:56
===SQL===
SELECT COUNT(*) FROM head WHERE age > 56;

Example 2
Question: "List the names of departments with budget over 1 million."

Schema:
Table: department
Columns: dept_id, name, budget
Table: employee
Columns: emp_id, name, salary

Output:
===SKELETON===
SELECT SLOT_1 FROM SLOT_2 WHERE SLOT_3 > SLOT_4;
===SCHEMA_LINKS===
department:name,budget
===SLOT_MATCHING===
SLOT_1:name
SLOT_2:department
SLOT_3:budget
SLOT_4:1000000
===SQL===
SELECT name FROM department WHERE budget > 1000000;

Example 3
Question: "What is the average salary of employees hired in 2020?"

Schema:
Table: employee
Columns: emp_id, name, salary, hire_date

Output:
===SKELETON===
SELECT AVG(SLOT_1) FROM SLOT_2 WHERE SLOT_3 LIKE SLOT_4;
===SCHEMA_LINKS===
employee:salary,hire_date
===SLOT_MATCHING===
SLOT_1:salary
SLOT_2:employee
SLOT_3:hire_date
SLOT_4:'2020%'
===SQL===
SELECT AVG(salary) FROM employee WHERE hire_date LIKE '2020%';

Now process the following:"""


def _format_schema(db_schema: dict[str, list[str]]) -> str:
    parts = []
    for table, cols in db_schema.items():
        parts.append(f"Table: {table}")
        parts.append(f"Columns: {', '.join(cols)}")
    return "\n".join(parts)


def build_user_prompt(question: str, db_schema: dict[str, list[str]]) -> str:
    return (
        f"{_INSTRUCTIONS}\n"
        f'Question: "{question}"\n'
        f"Schema:\n"
        f"{_format_schema(db_schema)}\n"
        f"Output:\n"
    )


def render_mediator(med: PAUQMediator, sql: str | None = None) -> str:
    """Render a mediator (and optional final SQL) as the structured assistant block."""
    lines = [
        "===SKELETON===",
        med.skeleton,
        "===SCHEMA_LINKS===",
    ]
    for table, cols in med.schema_links.items():
        lines.append(f"{table}:{','.join(cols)}")
    lines.append("===SLOT_MATCHING===")
    for i, slot in enumerate(med.slots, start=1):
        lines.append(f"SLOT_{i}:{slot}")
    lines.append("===SQL===")
    if sql is not None:
        lines.append(sql)
    return "\n".join(lines)


def render_mediator_prefix(med: PAUQMediator) -> str:
    """Render mediator as an assistant prefix WITHOUT the final SQL — used for the
    'gold_structure' continuation, where the model must only generate the SQL."""
    return render_mediator(med, sql=None) + "\n"
