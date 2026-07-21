import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated
import operator

import requests
from azure.identity import InteractiveBrowserCredential
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# CONFIG
# ============================================================

AZURE_OPENAI_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
API_VERSION             = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

WORKSPACE_ID = os.environ.get("POWERBI_WORKSPACE_ID")
DATASET_ID   = os.environ.get("POWERBI_DATASET_ID")

IJP_MEASURES_FILE    = os.environ.get("IJP_MEASURES_FILE", "ijp_measures.json")
IJP_DIMENSIONS_FILE  = os.environ.get("IJP_DIMENSIONS_FILE", "ijp_dimension.json")

MAX_REPAIR_ATTEMPTS = 2
MAX_WORKFLOW_STEPS = 8
CURRENT_YEAR = datetime.today().year

POWERBI_RESOURCE = "https://analysis.windows.net/powerbi/api"
POWERBI_EXECUTE_URL = (
    f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
    f"/datasets/{DATASET_ID}/executeQueries"
)

MAIN_TABLE = "Internal Job Posting"

DEBUG = os.environ.get("IJP_AGENT_DEBUG", "0") == "1"


# ============================================================
# LOAD METADATA
# ============================================================

with open(IJP_MEASURES_FILE, "r", encoding="utf-8") as f:
    IJP_MEASURES = json.load(f)

with open(IJP_DIMENSIONS_FILE, "r", encoding="utf-8") as f:
    IJP_DIMENSIONS_DATA = json.load(f)
    IJP_DIMENSIONS = IJP_DIMENSIONS_DATA["tables"]

IJP_MEASURES_BY_NAME = {m["measure_name"]: m for m in IJP_MEASURES}
ALL_IJP_MEASURE_NAMES = set(IJP_MEASURES_BY_NAME.keys())

# Build column -> table mapping (uses EXACT casing from the dimension JSON)
COLUMN_TO_TABLE: Dict[str, str] = {}
for _table, _cols in IJP_DIMENSIONS.items():
    for _col in _cols:
        COLUMN_TO_TABLE[_col] = _table

ALL_COLUMNS = set(COLUMN_TO_TABLE.keys())

# Numeric columns
NUMERIC_COLUMNS = {"Year", "Month"}

# The exact "active employee" column name and value, as they appear in
# ijp_dimension.json ("IS ACTIVE") — keep this as the single source of
# truth so casing bugs like the one this patch fixes can't creep back in.
ACTIVE_COLUMN = "IS ACTIVE"
ACTIVE_VALUE = 1


# ============================================================
# TABLE / COLUMN RESOLUTION
# ============================================================

def qualify_table(table_name: str) -> str:
    """Wrap table name in quotes if it contains spaces."""
    return f"'{table_name}'" if " " in table_name else table_name


def resolve_column(col: str, table: Optional[str] = None) -> str:
    """Resolve column to fully-qualified 'Table'[Column] DAX reference."""
    if table:
        if col not in IJP_DIMENSIONS.get(table, []):
            raise ValueError(f"Column '{col}' not in table '{table}'")
        return f"{qualify_table(table)}[{col}]"

    resolved_table = COLUMN_TO_TABLE.get(col)
    if not resolved_table:
        raise ValueError(f"Unknown column '{col}'")
    return f"{qualify_table(resolved_table)}[{col}]"


def get_dimension_context() -> str:
    """Return formatted string of all tables and columns."""
    lines = []
    for table, cols in IJP_DIMENSIONS.items():
        lines.append(f"\nTABLE: {table}")
        lines.extend(f"  {c}" for c in cols)
    return "\n".join(lines)


# ============================================================
# BUSINESS RULES
# ============================================================

BUSINESS_RULES = {
    "default_snap_date": None,  # Fetched dynamically from Power BI
    "default_is_active": ACTIVE_VALUE,  # Active employees only by default
}

MEASURE_ALIASES = {
    "headcount": "Head Count",
    "employee count": "Head Count",
    "ijp eligible": "Same Level IJP Eligible",
    "same level eligible": "Same Level IJP Eligible",
    "one level up eligible": "One Level IJP Eligible",
    "promotion eligible": "Promotion Eligible",
    "mid year promotion eligible": "Mid Year Promotion Eligible",
    "same role eligible": "Same Role Eligible",
}


# ============================================================
# LLM & POWERBI CLIENT
# ============================================================

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=API_VERSION,
    temperature=0,
    max_tokens=4000,
)


class PowerBIClient:
    def __init__(self):
        self._credential = None
        self._token: Optional[str] = None
        self._latest_snap: Optional[str] = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        self._credential = InteractiveBrowserCredential()
        self._token = self._credential.get_token(f"{POWERBI_RESOURCE}/.default").token
        return self._token

    def execute_dax(self, dax_query: str) -> dict:
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"queries": [{"query": dax_query}], "serializerSettings": {"includeNulls": True}}
        resp = requests.post(POWERBI_EXECUTE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        tables = result.get("results", [{}])[0]
        if "error" in tables:
            raise ValueError(f"DAX error: {tables['error']}")
        return result

    def get_latest_snapshot(self) -> str:
        if self._latest_snap:
            return self._latest_snap
        dax = (
            f'EVALUATE ROW("LatestSnap", '
            f'CALCULATE(MAX(\'{MAIN_TABLE}\'[Snap Date]), ALL(\'{MAIN_TABLE}\')))'
        )
        try:
            result = self.execute_dax(dax)
            rows = result["results"][0]["tables"][0]["rows"]
            raw_date = rows[0]["[LatestSnap]"]
            date_part = raw_date.split("T")[0]
            y, m, d = date_part.split("-")
            self._latest_snap = f"{int(y)},{int(m)},{int(d)}"
        except Exception:
            self._latest_snap = f"{CURRENT_YEAR},6,30"
        return self._latest_snap


pbi = PowerBIClient()


# ============================================================
# AGENT STATE
# ============================================================

class WorkflowStep(TypedDict, total=False):
    step_id: str
    description: str
    table: str
    measures: List[str]
    measure_filters: Dict[str, Dict]
    measure_labels: Dict[str, str]
    groupby: List[str]
    filters: Dict
    top_n: Optional[int]
    sort: Optional[Dict]
    depends_on: Optional[str]
    depends_on_column: Optional[str]
    entity_column: Optional[str]


class IJPState(TypedDict):
    question: str
    intent_type: str
    workflow_plan: List[WorkflowStep]
    workflow_results: Dict
    workflow_dax: Dict
    retrieved_measures: List[Dict]
    dimensions: str
    intent: Dict
    enriched_intent: Dict
    dax_query: str
    execution_status: int
    execution_result: Dict
    final_dax: str
    answer: str
    repair_attempts: int
    error_messages: List[str]
    workflow_failed_step_id: Optional[str]
    workflow_failed_step_index: Optional[int]
    workflow_step_repair_dax: Optional[str]
    workflow_step_error: Optional[Dict]
    workflow_step_repair_attempts: int


# ============================================================
# DEFAULTS & VALIDATION
# ============================================================

def apply_defaults(question: str, filters: Dict, allow_is_active_default: bool = True,
                   table: Optional[str] = None) -> Dict:
    """Apply business-rule defaults to filter dict."""
    q = question.lower()
    filters = dict(filters)
    table_cols = set(IJP_DIMENSIONS.get(table, [])) if table else None

    def col_ok(col_name: str) -> bool:
        if table_cols is None:
            return True
        return col_name in table_cols

    # Snap Date default
    if "Snap Date" not in filters:
        filters["Snap Date"] = pbi.get_latest_snapshot()

    # IS ACTIVE default (active employees only unless user asks for inactive/all).
    # NOTE: column name must match ijp_dimension.json EXACTLY ("IS ACTIVE", all caps) —
    # this was previously checked as "Is Active" and silently never matched.
    include_inactive = any(x in q for x in ["inactive", "exited", "all employees", "separated"])
    if allow_is_active_default and ACTIVE_COLUMN not in filters and col_ok(ACTIVE_COLUMN) and not include_inactive:
        filters[ACTIVE_COLUMN] = ACTIVE_VALUE

    # Employment Type — only applied if the user actually said it, never forced.
    if "Employment Type" not in filters and col_ok("Employment Type"):
        if "full time" in q or "fte" in q:
            filters["Employment Type"] = "Full Time"
        elif "part time" in q:
            filters["Employment Type"] = "Part Time"

    return filters


def validate_ijp_measures(measures: List[str]) -> List[str]:
    """Validate measure names."""
    errors = []
    for m in measures:
        if m not in ALL_IJP_MEASURE_NAMES:
            errors.append(f"Unknown IJP measure: '{m}'")
    return errors


def validate_ijp_columns(cols: List[str]) -> List[str]:
    """Validate column names."""
    return [f"Unknown IJP column: '{c}'" for c in cols if c not in ALL_COLUMNS]


# ============================================================
# GENERIC DAX BUILDERS
# ============================================================

def _format_filter_value(value, numeric: bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str) and value.lower() in ("yes", "true"):
        return "1"
    if isinstance(value, str) and value.lower() in ("no", "false"):
        return "0"
    if isinstance(value, (int, float)):
        return str(value)
    if numeric:
        return str(value)
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def build_filter_clause(column: str, value, table: Optional[str] = None) -> str:
    """
    Emits a plain boolean predicate for use as a CALCULATE()/CALCULATETABLE()
    filter argument, e.g.  'Internal Job Posting'[IS ACTIVE] = 1

    Previously this wrapped everything in FILTER(ALL(col), col = value), which
    is unnecessarily fragile when combined with measures that already apply
    their own CALCULATE() filters internally. The plain form below matches
    the syntax already confirmed to work directly against this dataset.
    """
    col_ref = resolve_column(column, table)
    numeric = column in NUMERIC_COLUMNS

    if isinstance(value, (list, tuple, set)):
        formatted = ", ".join(_format_filter_value(v, numeric) for v in value)
        return f"{col_ref} IN {{{formatted}}}"

    # Special handling for Snap Date (DATE function)
    if column == "Snap Date" and isinstance(value, str) and "," in value:
        y, m, d = value.split(",")
        return f"{col_ref} = DATE({y},{m},{d})"

    formatted = _format_filter_value(value, numeric)
    return f"{col_ref} = {formatted}"


def build_filters(filters: Dict, table: Optional[str] = None) -> List[str]:
    return [build_filter_clause(k, v, table) for k, v in filters.items()]


def build_entity_in_clause(column: str, values: List, table: Optional[str] = None) -> str:
    return build_filter_clause(column, list(values), table=table)


def build_sort(sort: Optional[Dict], fallback_alias: Optional[str] = None) -> str:
    if not sort:
        return f'[{fallback_alias}], DESC' if fallback_alias else ""
    by = sort.get("by", "") or fallback_alias or ""
    if not by:
        return ""
    order = sort.get("order", "desc").upper()
    return f"[{by}], {order}" if "[" in by else f"[{by}], {order}"


def build_filtered_measure_expression(
    measure_name: str,
    extra_filters: Optional[Dict] = None,
    table: Optional[str] = None,
) -> str:
    base = f"[{measure_name}]"
    if not extra_filters:
        return base

    predicates = []
    for col, value in extra_filters.items():
        col_ref = resolve_column(col, table)
        numeric = col in NUMERIC_COLUMNS
        if isinstance(value, (list, tuple, set)):
            formatted = ", ".join(_format_filter_value(v, numeric) for v in value)
            predicates.append(f"{col_ref} IN {{{formatted}}}")
        else:
            formatted = _format_filter_value(value, numeric)
            predicates.append(f"{col_ref} = {formatted}")

    predicate_str = ",\n        ".join(predicates)
    return f"CALCULATE(\n        {base},\n        {predicate_str}\n    )"


def validate_dax_before_execution(dax: str) -> List[str]:
    """
    Validate DAX for common errors before execution.
    Returns list of validation errors (empty if valid).
    """
    errors = []

    if not dax.strip().startswith("EVALUATE"):
        errors.append("DAX must start with EVALUATE keyword")

    open_count = dax.count("(")
    close_count = dax.count(")")
    if open_count != close_count:
        errors.append(f"Unbalanced parentheses: {open_count} open, {close_count} close")

    single_quotes = dax.count("'")
    if single_quotes % 2 != 0:
        errors.append(f"Unbalanced single quotes: {single_quotes} total")

    double_quotes = dax.count('"')
    if double_quotes % 2 != 0:
        errors.append(f"Unbalanced double quotes: {double_quotes} total")

    if "SUMMARIZECOLUMN(" in dax:
        errors.append("Typo detected: SUMMARIZECOLUMN should be SUMMARIZECOLUMNS")

    if "EVALUTE" in dax:
        errors.append("Typo detected: EVALUTE should be EVALUATE")

    if "''" in dax or '[]' in dax:
        errors.append("Empty table or column reference detected")

    return errors


def render_metric_spec_dax(spec: Dict, entity_filter_clause: Optional[str] = None) -> str:
    """Unified DAX renderer for IJP specs."""
    table = spec.get("table") or MAIN_TABLE
    measures = spec.get("measures", [])

    if not measures:
        raise ValueError("metric spec requires at least one measure")

    measure_filters = spec.get("measure_filters", {}) or {}
    measure_labels = spec.get("measure_labels", {}) or {}
    groupby = spec.get("groupby", []) or []
    filters = spec.get("filters", {}) or {}
    top_n = spec.get("top_n")
    sort = spec.get("sort")

    dax_filters = build_filters(filters, table=table if table != MAIN_TABLE else None)
    if entity_filter_clause:
        dax_filters.append(entity_filter_clause)

    dax_groupby = [resolve_column(c, table=table if table != MAIN_TABLE else None) for c in groupby]

    # Scalar
    if not groupby and not top_n:
        rows = []
        for m in measures:
            label = measure_labels.get(m, m)
            expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
            calc_body = expr if not dax_filters else f"CALCULATE(\n        {expr},\n        {', '.join(dax_filters)}\n    )"
            rows.append(f'    "{label}",\n    {calc_body}')
        dax = "EVALUATE\nROW(\n" + ",\n".join(rows) + "\n)"
    elif not groupby and top_n:
        # TOPN requires a table, not ROW — with no groupby there's nothing to
        # rank over, so fall back to scalar.
        rows = []
        for m in measures:
            label = measure_labels.get(m, m)
            expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
            calc_body = expr if not dax_filters else f"CALCULATE(\n        {expr},\n        {', '.join(dax_filters)}\n    )"
            rows.append(f'    "{label}",\n    {calc_body}')
        dax = "EVALUATE\nROW(\n" + ",\n".join(rows) + "\n)"
    else:
        # Grouped
        summarize_parts = []
        summarize_parts.extend(dax_groupby)
        summarize_parts.extend(dax_filters)
        for m in measures:
            label = measure_labels.get(m, m)
            expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
            summarize_parts.append(f'"{label}"')
            summarize_parts.append(expr)

        summarize = "SUMMARIZECOLUMNS(\n    " + ",\n    ".join(summarize_parts) + "\n)"

        if top_n:
            primary_measure = measure_labels.get(measures[0], measures[0])
            sort_expr = build_sort(sort, fallback_alias=primary_measure)
            dax = f"EVALUATE\nTOPN(\n    {top_n},\n    {summarize},\n    {sort_expr}\n)"
        else:
            dax = f"EVALUATE\n{summarize}"

    validation_errors = validate_dax_before_execution(dax)
    if validation_errors:
        raise ValueError(f"DAX validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors))

    return dax


# ============================================================
# MEASURE RETRIEVAL
# ============================================================

def retrieve_ijp_measures(question: str, top_k: int = 20) -> List[Dict]:
    """Retrieve relevant measures using lexical scoring."""
    q = question.lower()
    scored = []

    for measure in IJP_MEASURES:
        text = (measure.get("measure_name", "") + " " +
                measure.get("description", "")).lower()
        score = 0

        for alias, target in MEASURE_ALIASES.items():
            if alias in q and target == measure["measure_name"]:
                score += 10

        for token in re.findall(r"[a-z0-9%]+", q):
            if len(token) > 2 and token in text:
                score += 1

        scored.append((score, measure))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# ============================================================
# GRAPH NODES
# ============================================================

def retrieve_ijp_measures_node(state: IJPState) -> IJPState:
    measures = retrieve_ijp_measures(state["question"])
    dims = get_dimension_context()
    return {**state, "retrieved_measures": measures, "dimensions": dims}


IJP_MASTER_INTENT_PROMPT = ChatPromptTemplate.from_template(
"""You are an IJP (Internal Job Posting) analytics expert.

USER QUESTION:
{question}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

Classify the question as ONE of: standard | topn | comparison | workflow
- "workflow" = the question needs MULTIPLE distinct metrics/steps, OR one step's
  output feeds into another (e.g. "top 3 BUs, then eligibility for each").
- Otherwise use standard / topn / comparison for a single self-contained query.

For STANDARD / TOPN / COMPARISON, return a single JSON object with this EXACT schema:
{{
  "intent_type": "standard",
  "table": "Internal Job Posting",
  "measures": ["Head Count"],
  "groupby": [],
  "filters": {{}},
  "top_n": null,
  "sort": null,
  "measure_filters": {{}},
  "measure_labels": {{}}
}}
Rules:
- measures MUST use EXACT names from AVAILABLE MEASURES.
- groupby MUST use EXACT column names from AVAILABLE DIMENSIONS.
- filters: only include a key if the user EXPLICITLY mentioned that value in
  the question (e.g. a specific Business Unit, Grade, Gender). Do NOT include
  IS ACTIVE, Snap Date, or Employment Type — those are applied automatically
  afterward based on business rules, not by you.
- top_n: integer only if user asked for "top N" / "bottom N"; else null.

For WORKFLOW, return a JSON object: {{"steps": [ ... ]}} where each step follows
this EXACT schema:
{{
  "step_id": "step1",
  "description": "short description",
  "table": "Internal Job Posting",
  "measures": ["Head Count"],
  "groupby": [],
  "filters": {{}},
  "top_n": null,
  "sort": null,
  "measure_filters": {{}},
  "measure_labels": {{}},
  "depends_on": null,
  "depends_on_column": null,
  "entity_column": null
}}
Rules:
- step_id is REQUIRED, unique, sequential ("step1", "step2", ...).
- Only set depends_on/depends_on_column/entity_column if this step needs the
  ENTITY VALUES produced by a prior step (e.g. the top-3 BU names).
- Same filter rules as above (no IS ACTIVE / Snap Date / Employment Type).

Example — "How many are eligible for same-level and one-level-up IJP?" (workflow,
independent steps, no dependency):
{{"steps": [
  {{"step_id": "step1", "description": "Same-level IJP eligible", "table": "Internal Job Posting",
    "measures": ["Same Level IJP Eligible"], "groupby": [], "filters": {{}},
    "top_n": null, "sort": null, "measure_filters": {{}}, "measure_labels": {{}},
    "depends_on": null, "depends_on_column": null, "entity_column": null}},
  {{"step_id": "step2", "description": "One-level-up IJP eligible", "table": "Internal Job Posting",
    "measures": ["One Level IJP Eligible"], "groupby": [], "filters": {{}},
    "top_n": null, "sort": null, "measure_filters": {{}}, "measure_labels": {{}},
    "depends_on": null, "depends_on_column": null, "entity_column": null}}
]}}

Output JSON only. No explanation. No markdown fences.
""")


def extract_ijp_intent_node(state: IJPState) -> IJPState:
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:15]])

    prompt = IJP_MASTER_INTENT_PROMPT.format(
        question=state["question"],
        measures=measures_ref,
        dimensions=state["dimensions"],
    )

    response = llm.invoke(prompt)
    raw = response.content.strip().strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    if DEBUG:
        print(f"📝 LLM Response (first 300 chars): {raw[:300]}...")

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, dict) and "steps" in parsed:
            intent_type = "workflow"
            intent = parsed
            if DEBUG:
                print(f"✅ Detected workflow with {len(intent.get('steps', []))} steps")
        elif isinstance(parsed, list):
            intent_type = "workflow"
            intent = {"intent_type": "workflow", "steps": parsed}
            if DEBUG:
                print(f"✅ Detected workflow (list format) with {len(parsed)} steps")
        else:
            intent = parsed
            intent_type = intent.get("intent_type", "standard")
            if DEBUG:
                print(f"✅ Detected {intent_type} query with measures: {intent.get('measures', [])}")

    except Exception as e:
        print(f"⚠️ Intent parse failed: {e}, using default")
        intent = {"intent_type": "standard", "table": MAIN_TABLE, "measures": ["Head Count"], "filters": {}}
        intent_type = "standard"

    return {**state, "intent": intent, "intent_type": intent_type}


def enrich_ijp_intent_node(state: IJPState) -> IJPState:
    intent = dict(state["intent"])
    intent_type = state["intent_type"]
    question = state["question"]

    # For workflow, skip enrichment here — it's done per-step in plan_ijp_workflow_node
    if intent_type == "workflow":
        return {**state, "enriched_intent": intent}

    if intent_type in ("standard", "topn", "comparison"):
        if "table" not in intent:
            intent["table"] = MAIN_TABLE
        if "measures" not in intent or not intent["measures"]:
            intent["measures"] = ["Head Count"]
        if "filters" not in intent:
            intent["filters"] = {}

        filters = apply_defaults(question, intent.get("filters", {}), table=intent.get("table"))
        intent["filters"] = filters

        measures = intent.get("measures", [])
        groupby = intent.get("groupby", [])

        measure_errors = validate_ijp_measures(measures)
        column_errors = validate_ijp_columns(groupby)

        if measure_errors or column_errors:
            error_msg = "\n".join(measure_errors + column_errors)
            print(f"❌ Validation errors:\n{error_msg}")
            return {**state, "enriched_intent": intent, "error_messages": state.get("error_messages", []) + [error_msg],
                    "answer": f"Validation failed:\n{error_msg}"}

    return {**state, "enriched_intent": intent}


def build_ijp_dax_node(state: IJPState) -> IJPState:
    try:
        spec = state["enriched_intent"]

        if DEBUG:
            print(f"📝 Building DAX from spec:")
            print(f"   Table: {spec.get('table')}")
            print(f"   Measures: {spec.get('measures')}")
            print(f"   Groupby: {spec.get('groupby')}")
            print(f"   Filters: {spec.get('filters')}")

        if not spec.get("measures"):
            raise ValueError("No measures found in enriched intent")

        dax = render_metric_spec_dax(spec)
        if DEBUG:
            print(f"✅ DAX generated successfully ({len(dax)} chars)")
        return {**state, "dax_query": dax, "final_dax": dax}
    except Exception as e:
        error_msg = f"DAX generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"   Intent: {state.get('enriched_intent')}")
        return {**state, "error_messages": state.get("error_messages", []) + [error_msg], "answer": error_msg}


def execute_ijp_dax_node(state: IJPState) -> IJPState:
    dax = state.get("final_dax") or state.get("dax_query")

    if not dax:
        error_msg = "No DAX query generated"
        print(f"❌ {error_msg}")
        return {**state, "execution_status": 500,
                "error_messages": state.get("error_messages", []) + [error_msg],
                "answer": error_msg}

    if DEBUG:
        print(f"\n📋 Executing DAX ({len(dax)} chars):")
        print("─" * 60)
        print(dax)
        print("─" * 60)

    try:
        result = pbi.execute_dax(dax)
        if DEBUG:
            print(f"✅ DAX execution successful")
            print(f"   Result: {json.dumps(result, indent=2)[:1000]}")
        return {**state, "execution_status": 200, "execution_result": result, "final_dax": dax}
    except Exception as e:
        error_msg = str(e)
        print(f"❌ DAX execution failed: {error_msg}")
        if DEBUG:
            print(f"   Failed DAX:")
            print("   " + "\n   ".join(dax.split("\n")))

        if state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
            return {**state, "execution_status": 500,
                    "error_messages": state.get("error_messages", []) + [error_msg]}
        else:
            return {**state, "execution_status": 500,
                    "error_messages": state.get("error_messages", []) + [f"Max retries. Error: {error_msg}"],
                    "answer": f"Query failed after {MAX_REPAIR_ATTEMPTS} attempts.\nError: {error_msg}\n\nGenerated DAX:\n{dax}"}


REPAIR_DAX_PROMPT = ChatPromptTemplate.from_template(
"""Fix this failed DAX query.

QUESTION: {question}
FAILED DAX: {dax}
ERROR: {error}

MEASURES: {measures}
DIMENSIONS: {dimensions}

Note: the active-employee filter column is exactly 'Internal Job Posting'[IS ACTIVE]
(all caps) — if the error involves an unknown "Is Active" column, that casing is the bug.

Return ONLY corrected DAX, no explanation.
""")


def repair_ijp_dax_node(state: IJPState) -> IJPState:
    failed_dax = state.get("final_dax") or state.get("dax_query")
    error_msgs = state.get("error_messages", [])
    error_msg = error_msgs[-1] if error_msgs else "Unknown error"
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:20]])

    prompt = REPAIR_DAX_PROMPT.format(
        question=state["question"], dax=failed_dax, error=error_msg,
        measures=measures_ref, dimensions=state["dimensions"]
    )

    response = llm.invoke(prompt)
    repaired_dax = response.content.strip().strip("`").strip()
    if repaired_dax.startswith("dax"):
        repaired_dax = repaired_dax[3:].strip()

    attempts = state.get("repair_attempts", 0) + 1
    if DEBUG:
        print(f"🔧 Repair attempt {attempts}")

    return {**state, "final_dax": repaired_dax, "repair_attempts": attempts}


WORKFLOW_PLANNER_PROMPT = ChatPromptTemplate.from_template(
"""Plan an IJP workflow as ordered steps.

QUESTION: {question}
MEASURES: {measures}
DIMENSIONS: {dimensions}

Return a JSON array. Each step MUST follow this exact schema:
{{
  "step_id": "step1",
  "description": "short description",
  "table": "Internal Job Posting",
  "measures": ["Head Count"],
  "groupby": ["Business Unit Name"],
  "filters": {{}},
  "top_n": null,
  "sort": null,
  "measure_filters": {{}},
  "measure_labels": {{}},
  "depends_on": null,
  "depends_on_column": null,
  "entity_column": null
}}

Rules:
- step_id is REQUIRED for every step, unique, format "step1", "step2", ...
- measures MUST use exact names from MEASURES list.
- groupby MUST use exact column names from DIMENSIONS.
- Only set depends_on/depends_on_column/entity_column if a step needs entities
  produced by a prior step (e.g. "top 3 BUs, then show eligibility for each").
- Do NOT include IS ACTIVE, Snap Date, or Employment Type in filters —
  those are applied automatically afterward.
- Max {max_steps} steps.

Example for "Top 3 BUs by headcount, then show IJP eligibility for each":
[
  {{"step_id": "step1", "description": "Top 3 BUs by headcount", "table": "Internal Job Posting",
    "measures": ["Head Count"], "groupby": ["Business Unit Name"], "filters": {{}},
    "top_n": 3, "sort": {{"by": "Head Count", "order": "desc"}},
    "measure_filters": {{}}, "measure_labels": {{}},
    "depends_on": null, "depends_on_column": null, "entity_column": null}},
  {{"step_id": "step2", "description": "IJP eligibility for those BUs", "table": "Internal Job Posting",
    "measures": ["Same Level IJP Eligible", "One Level IJP Eligible"], "groupby": ["Business Unit Name"],
    "filters": {{}}, "top_n": null, "sort": null, "measure_filters": {{}}, "measure_labels": {{}},
    "depends_on": "step1", "depends_on_column": "Business Unit Name",
    "entity_column": "Business Unit Name"}}
]

Return JSON array only, no explanation, no markdown fences.
""")


def plan_ijp_workflow_node(state: IJPState) -> IJPState:
    intent = state.get("intent", {})
    if "steps" in intent:
        workflow_plan = intent["steps"]
        if DEBUG:
            print(f"✅ Using workflow plan from intent ({len(workflow_plan)} steps)")
    else:
        measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:15]])

        prompt = WORKFLOW_PLANNER_PROMPT.format(
            question=state["question"], measures=measures_ref,
            dimensions=state["dimensions"], max_steps=MAX_WORKFLOW_STEPS
        )

        response = llm.invoke(prompt)
        raw = response.content.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            workflow_plan = json.loads(raw)
            if not isinstance(workflow_plan, list):
                workflow_plan = []
        except Exception as e:
            print(f"⚠️ Workflow plan parse failed: {e}")
            workflow_plan = []

    # Defensive normalization: guarantee step_id/table/filters exist on every
    # step BEFORE anything else does step["step_id"] etc. A malformed LLM
    # step previously crashed the whole graph with an uncaught KeyError.
    for i, step in enumerate(workflow_plan):
        if "step_id" not in step or not step["step_id"]:
            step["step_id"] = f"step{i + 1}"
        if "table" not in step or not step["table"]:
            step["table"] = MAIN_TABLE
        if "filters" not in step or step["filters"] is None:
            step["filters"] = {}
        if "measures" not in step:
            step["measures"] = []
        if "groupby" not in step:
            step["groupby"] = []

    # Apply defaults per step — allow_is_active_default is now True, so
    # multi-part questions get the active-employee scope too.
    for step in workflow_plan:
        step["filters"] = apply_defaults(
            state["question"], step.get("filters", {}),
            allow_is_active_default=True, table=step.get("table")
        )

    # Validate steps
    for step in workflow_plan:
        measures = step.get("measures", [])
        groupby = step.get("groupby", [])
        measure_errors = validate_ijp_measures(measures)
        column_errors = validate_ijp_columns(groupby)
        if measure_errors or column_errors:
            error_msg = f"Step {step['step_id']} validation failed:\n" + "\n".join(measure_errors + column_errors)
            print(f"❌ {error_msg}")
            return {**state, "workflow_plan": [],
                    "error_messages": state.get("error_messages", []) + [error_msg],
                    "answer": error_msg}

    return {**state, "workflow_plan": workflow_plan}


def extract_ijp_column_values(result: Dict, column_label: str) -> List:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
    except Exception:
        return []
    if not rows:
        return []

    matching_key = None
    for key in rows[0].keys():
        if key == column_label or key.endswith(f"[{column_label}]") or key.endswith(f".{column_label}"):
            matching_key = key
            break

    if not matching_key:
        matching_key = list(rows[0].keys())[0]

    seen, values = set(), []
    for row in rows:
        v = row.get(matching_key)
        if v is not None and v not in seen:
            seen.add(v)
            values.append(v)
    return values


def execute_ijp_workflow_node(state: IJPState) -> IJPState:
    workflow_plan = state["workflow_plan"]
    workflow_results = dict(state.get("workflow_results", {}))
    workflow_dax = dict(state.get("workflow_dax", {}))

    if state.get("workflow_step_repair_dax"):
        failed_idx = state["workflow_failed_step_index"]
        step = workflow_plan[failed_idx]
        dax = state["workflow_step_repair_dax"]

        try:
            result = pbi.execute_dax(dax)
            workflow_results[step["step_id"]] = result
            workflow_dax[step["step_id"]] = dax
            state = {**state, "workflow_results": workflow_results, "workflow_dax": workflow_dax,
                     "workflow_step_repair_dax": None, "workflow_failed_step_id": None,
                     "workflow_failed_step_index": None, "workflow_step_error": None}
            start_idx = failed_idx + 1
        except Exception as e:
            error_msg = str(e)
            return {**state, "workflow_step_error": {"message": error_msg},
                    "error_messages": [f"Step {step['step_id']} failed after repair"],
                    "answer": f"Workflow failed at {step['step_id']} after repair"}
    else:
        start_idx = len(workflow_results)

    for idx in range(start_idx, len(workflow_plan)):
        step = workflow_plan[idx]
        step_id = step["step_id"]

        entity_filter_clause = None
        if step.get("depends_on"):
            dep_step_id = step["depends_on"]
            dep_column = step.get("depends_on_column")
            entity_column = step.get("entity_column")

            if dep_step_id not in workflow_results:
                error_msg = f"{step_id} depends on missing {dep_step_id}"
                return {**state, "error_messages": [error_msg], "answer": error_msg}

            entity_values = extract_ijp_column_values(workflow_results[dep_step_id], dep_column)
            if not entity_values:
                error_msg = f"No entities in {dep_step_id} for {dep_column}"
                return {**state, "workflow_results": workflow_results, "workflow_dax": workflow_dax,
                        "error_messages": [error_msg], "answer": f"Workflow stopped: {error_msg}"}

            entity_filter_clause = build_entity_in_clause(entity_column, entity_values, table=step.get("table"))

        try:
            dax = render_metric_spec_dax(step, entity_filter_clause)
            if DEBUG:
                print(f"\n📋 Executing step '{step_id}' DAX:\n{dax}\n")
            result = pbi.execute_dax(dax)
            workflow_results[step_id] = result
            workflow_dax[step_id] = dax
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Step '{step_id}' failed: {error_msg}")
            if state.get("workflow_step_repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
                return {**state, "workflow_results": workflow_results, "workflow_dax": workflow_dax,
                        "workflow_failed_step_id": step_id, "workflow_failed_step_index": idx,
                        "workflow_step_error": {"message": error_msg, "dax": dax if 'dax' in locals() else ""}}
            else:
                return {**state, "workflow_results": workflow_results, "workflow_dax": workflow_dax,
                        "error_messages": [f"{step_id} failed after max retries"],
                        "answer": f"Workflow failed at {step_id}: {error_msg}"}

    return {**state, "workflow_results": workflow_results, "workflow_dax": workflow_dax, "execution_status": 200}


def repair_ijp_workflow_step_node(state: IJPState) -> IJPState:
    step_id = state["workflow_failed_step_id"]
    step_idx = state["workflow_failed_step_index"]
    step = state["workflow_plan"][step_idx]
    error_info = state["workflow_step_error"]

    failed_dax = error_info.get("dax", "")
    error_msg = error_info.get("message", "Unknown")
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:20]])

    prompt = REPAIR_DAX_PROMPT.format(
        question=f"Step {step_id}: {step.get('description', '')}",
        dax=failed_dax, error=error_msg, measures=measures_ref, dimensions=state["dimensions"]
    )

    response = llm.invoke(prompt)
    repaired_dax = response.content.strip().strip("`").strip()
    if repaired_dax.startswith("dax"):
        repaired_dax = repaired_dax[3:].strip()

    attempts = state.get("workflow_step_repair_attempts", 0) + 1
    return {**state, "workflow_step_repair_dax": repaired_dax, "workflow_step_repair_attempts": attempts}


ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template(
"""You are an IJP analytics analyst.

QUESTION: {question}
TYPE: {intent_type}
RESULT: {result}

Write clear, business-readable answer.
Use ASCII tables for multi-row. Round percentages. Highlight insights.
""")


def generate_ijp_answer_node(state: IJPState) -> IJPState:
    if state.get("answer"):
        return state

    intent_type = state["intent_type"]

    if intent_type == "workflow":
        workflow_results = state.get("workflow_results", {})
        workflow_plan = state.get("workflow_plan", [])
        result_summary = []
        for step in workflow_plan:
            step_id = step["step_id"]
            if step_id in workflow_results:
                result_summary.append(f"\n[{step_id}] {step.get('description', '')}")
                result_summary.append(json.dumps(workflow_results[step_id], indent=2))
        result_str = "\n".join(result_summary)
    else:
        result_str = json.dumps(state.get("execution_result", {}), indent=2)

    prompt = ANSWER_GENERATION_PROMPT.format(
        question=state["question"], intent_type=intent_type, result=result_str
    )

    response = llm.invoke(prompt)
    return {**state, "answer": response.content.strip()}


# ============================================================
# ROUTING LOGIC
# ============================================================

def route_by_intent(state: IJPState) -> str:
    if state.get("error_messages"):
        return "generate_answer"
    intent_type = state.get("intent_type", "standard")
    return "plan_workflow" if intent_type == "workflow" else "build_dax"


def route_after_execute(state: IJPState) -> str:
    status = state.get("execution_status", 0)
    if status == 200:
        return "generate_answer"
    elif state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        return "repair_dax"
    else:
        return "generate_answer"


def route_after_workflow_execute(state: IJPState) -> str:
    if state.get("execution_status") == 200:
        return "generate_answer"
    elif state.get("workflow_failed_step_id"):
        if state.get("workflow_step_repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
            return "repair_workflow_step"
        else:
            return "generate_answer"
    else:
        return "generate_answer"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_ijp_graph():
    """Build the IJP agent LangGraph."""
    graph = StateGraph(IJPState)

    graph.add_node("retrieve_measures", retrieve_ijp_measures_node)
    graph.add_node("extract_intent", extract_ijp_intent_node)
    graph.add_node("enrich_intent", enrich_ijp_intent_node)
    graph.add_node("build_dax", build_ijp_dax_node)
    graph.add_node("execute_dax", execute_ijp_dax_node)
    graph.add_node("repair_dax", repair_ijp_dax_node)
    graph.add_node("plan_workflow", plan_ijp_workflow_node)
    graph.add_node("execute_workflow", execute_ijp_workflow_node)
    graph.add_node("repair_workflow_step", repair_ijp_workflow_step_node)
    graph.add_node("generate_answer", generate_ijp_answer_node)

    graph.set_entry_point("retrieve_measures")

    graph.add_edge("retrieve_measures", "extract_intent")
    graph.add_edge("extract_intent", "enrich_intent")

    graph.add_conditional_edges(
        "enrich_intent",
        route_by_intent,
        {"build_dax": "build_dax", "plan_workflow": "plan_workflow", "generate_answer": "generate_answer"}
    )

    graph.add_edge("build_dax", "execute_dax")
    graph.add_conditional_edges(
        "execute_dax",
        route_after_execute,
        {"generate_answer": "generate_answer", "repair_dax": "repair_dax"}
    )
    graph.add_edge("repair_dax", "execute_dax")

    graph.add_edge("plan_workflow", "execute_workflow")
    graph.add_conditional_edges(
        "execute_workflow",
        route_after_workflow_execute,
        {"generate_answer": "generate_answer", "repair_workflow_step": "repair_workflow_step"}
    )
    graph.add_edge("repair_workflow_step", "execute_workflow")

    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=MemorySaver())


# ============================================================
# PUBLIC API
# ============================================================

def ask(question: str, session_id: str) -> str:
    """
    Main entry point for IJP agent.

    Args:
        question: User question
        session_id: Session identifier for tracking context

    Returns:
        Business-readable answer string
    """
    print("\n" + "=" * 80)
    print("IJP QUESTION:")
    print(question)
    print("=" * 80)

    app = build_ijp_graph()

    initial_state: IJPState = {
        "question": question,
        "intent_type": "",
        "workflow_plan": [],
        "workflow_results": {},
        "workflow_dax": {},
        "retrieved_measures": [],
        "dimensions": "",
        "intent": {},
        "enriched_intent": {},
        "dax_query": "",
        "execution_status": 0,
        "execution_result": {},
        "final_dax": "",
        "answer": "",
        "repair_attempts": 0,
        "error_messages": [],
        "workflow_failed_step_id": None,
        "workflow_failed_step_index": None,
        "workflow_step_repair_dax": None,
        "workflow_step_error": None,
        "workflow_step_repair_attempts": 0,
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = app.invoke(initial_state, config)

    return final_state["answer"]


# ============================================================
# TESTING & CLI
# ============================================================

def test_ijp_pipeline():
    """Test the IJP agent with sample questions."""
    import uuid

    session_id = str(uuid.uuid4())

    test_questions = [
        "What is the headcount?",
        "How many employees are eligible for same-level IJP?",
        "What is the promotion eligible headcount by business unit?",
        "How many employees are eligible for same-level and one-level-up IJP?",
        "Top 3 BUs by headcount, then show IJP eligibility for each",
        "Show one-level-up eligible count by grade for full-time employees",
        "How many inactive employees are there?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 80}\nTEST: {q}\n{'=' * 80}")
        try:
            answer = ask(q, session_id)
            print(f"\nANSWER:\n{answer}\n")
        except Exception as e:
            print(f"\nERROR: {e}\n")


if __name__ == "__main__":
    import uuid

    print("=" * 80)
    print("IJP Dashboard Agent v5 (Production Grade, patched)")
    if DEBUG:
        print("[DEBUG MODE ON]")
    print("=" * 80)

    session_id = str(uuid.uuid4())
    print(f"📋 Session ID: {session_id}\n")

    print("Type a question, or 'test' to run the built-in smoke test suite, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "test":
            test_ijp_pipeline()
            continue

        try:
            answer = ask(user_input, session_id)
            print(f"\n{'=' * 80}\nRESULT:\n{answer}\n{'=' * 80}\n")
        except Exception as e:
            print(f"Error: {e}")
