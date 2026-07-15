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
# CONFIG  (same env-var pattern as v6.py)
# ============================================================

AZURE_OPENAI_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
API_VERSION             = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

WORKSPACE_ID = os.environ.get("POWERBI_WORKSPACE_ID")
DATASET_ID   = os.environ.get("POWERBI_DATASET_ID")

SOC_MEASURES_FILE    = os.environ.get("SOC_MEASURES_FILE",    "soc_measures.json")
SOC_DIMENSIONS_FILE  = os.environ.get("SOC_DIMENSIONS_FILE",  "soc_dimensions.json")

MAX_REPAIR_ATTEMPTS = 2
CURRENT_YEAR        = datetime.today().year
CURRENT_MONTH       = datetime.today().month

# Month number → name string (as stored in 'Dim Cal'[Month])
MONTH_NUMBER_TO_NAME = {
    1: "January",  2: "February",  3: "March",
    4: "April",    5: "May",       6: "June",
    7: "July",     8: "August",    9: "September",
    10: "October", 11: "November", 12: "December",
}


# ============================================================
# LOAD METADATA
# ============================================================

with open(SOC_MEASURES_FILE,   "r", encoding="utf-8") as f:
    SOC_MEASURES = json.load(f)

with open(SOC_DIMENSIONS_FILE, "r", encoding="utf-8") as f:
    SOC_DIMENSIONS = json.load(f)

SOC_MEASURES_BY_NAME: Dict[str, Dict] = {m["measure_name"]: m for m in SOC_MEASURES}
ALL_SOC_MEASURE_NAMES = set(SOC_MEASURES_BY_NAME.keys())

# ── Column → table resolution ────────────────────────────────────────────
# Built once from soc_dimensions.json so every query resolves columns to
# the right table automatically.  Replaces the old hardcoded DIMENSION_MAP
# of only 6 entries.
SOC_COLUMN_TO_TABLE: Dict[str, str] = {}
for _tbl, _cols in SOC_DIMENSIONS["tables"].items():
    for _col in _cols:
        SOC_COLUMN_TO_TABLE[_col] = _tbl

ALL_SOC_COLUMNS = set(SOC_COLUMN_TO_TABLE.keys())

# Columns that live in 'Dim Cal' (for date/year FILTER(ALL(...)) treatment)
DIM_CAL_COLS = set(SOC_DIMENSIONS["tables"].get("Dim Cal", []))

# Measures that rely on SELECTEDVALUE() slicer context — they are NOT
# safely aggregatable in SUMMARIZECOLUMNS from the API.
# Detected automatically from formula content below.
SELECTEDVALUE_MEASURES: set = set()
for _m in SOC_MEASURES:
    if "SELECTEDVALUE" in _m.get("formula", "").upper():
        SELECTEDVALUE_MEASURES.add(_m["measure_name"])

# ── SELECTEDVALUE rewrite map ─────────────────────────────────────────────
# These measures use SELECTEDVALUE() and only work with a visual slicer
# context — they cannot be aggregated via executeQueries API.
# When the LLM picks one, we rewrite the spec to use the correct API-safe
# measure based on the user's question context.
#
# Direct_Repotee_Count / Direct_Reportee_Count_Male / Direct_Reportee_Count_Female
#   → counts reportees under a SINGLE selected manager (slicer-driven).
#   For API use: Level_N measures count employees at each level, which is the
#   closest semantically correct substitute. For gender breakdown, use
#   Employees_Head_Count grouped by Gender.
#
# Layers  → a SWITCH on two slicers (View[Value] and Level[Value]).
#   For API use: Level_1 … Level_10 for employee counts per level, or
#   Level1_Manager … Level9_Manager for manager counts per level.
#
# Complete_Span  → counts all employees under a selected manager (slicer).
#   For API use: Employees_Head_Count (full org headcount, no slicer needed).

# Maps "Level_N" tokens in the question → the correct Level_N measure name
LEVEL_NUMBER_TO_MEASURE = {str(n): f"Level_{n}" for n in range(1, 11)}
LEVEL_NUMBER_TO_MANAGER_MEASURE = {str(n): f"Level{n}_Manager" for n in range(1, 10)}

# Groupby column to use when drilling into a specific level's sub-entities
LEVEL_NUMBER_TO_NAME_COL = {
    "1": "Level1 Name", "2": "Level2 Name", "3": "Level3 Name",
    "4": "Level4 Name", "5": "Level5 Name", "6": "Level6 Name",
    "7": "Level7 Name", "8": "Level8 Name", "9": "Level9 Name",
}


def _extract_level_number(question: str) -> Optional[str]:
    """Extract the first Level number mentioned in the question, e.g. 'Level_7' → '7'."""
    m = re.search(r'\blevel[_\s\-]?(\d{1,2})\b', question, re.IGNORECASE)
    return m.group(1) if m else None


def rewrite_selectedvalue_spec(spec: Dict, question: str) -> Dict:
    """
    Rewrite a metric spec that contains SELECTEDVALUE-dependent measures into
    an equivalent spec that can be executed via the executeQueries API.

    Called automatically in build_soc_dax_node and plan_soc_workflow_node
    before validation, so the validator never sees the bad measures.

    Rewrite rules (applied in order, first match wins per measure):

    Direct_Repotee_Count / Direct_Reportee_Count_Male / Direct_Reportee_Count_Female
      - If question mentions a specific level (Level_N):
          → replace with Level_N employee measure; if Male/Female add groupby Gender
      - Otherwise:
          → replace with Employees_Head_Count; if Male/Female add groupby Gender

    Complete_Span
      → replace with Employees_Head_Count

    Layers
      - If question mentions a specific level (Level_N) and "manager":
          → replace with Level{N}_Manager
      - If question mentions a specific level (Level_N):
          → replace with Level_N
      - If groupby is set (e.g. grouped by Level5 Name):
          → replace with Employees_Head_Count (count per group)
      - Otherwise:
          → generate a comparison spec across all Level_1..Level_10 measures
            (returned as a scalar ROW with each level as a column); set a
            special flag so the caller knows a multi-measure ROW was chosen.

    Analyse
      → replace with Employees_Head_Count
    """
    measures  = list(spec.get("measures") or [])
    groupby   = list(spec.get("groupby") or [])
    filters   = dict(spec.get("filters") or {})
    top_n     = spec.get("top_n")
    sort      = spec.get("sort")
    q         = question.lower()
    level_num = _extract_level_number(question)   # e.g. "7"

    rewritten_measures = []
    add_gender_groupby = False
    rewrote = False

    for m in measures:
        if m not in SELECTEDVALUE_MEASURES:
            rewritten_measures.append(m)
            continue

        rewrote = True
        print(f"[SELECTEDVALUE-REWRITE] '{m}' → rewriting for API compatibility")

        # ── Direct_Repotee_Count / Male / Female ──────────────────────────
        if m in ("Direct_Repotee_Count",
                 "Direct_Reportee_Count_Male",
                 "Direct_Reportee_Count_Female"):
            is_male   = m == "Direct_Reportee_Count_Male"
            is_female = m == "Direct_Reportee_Count_Female"

            if level_num and level_num in LEVEL_NUMBER_TO_MEASURE:
                substitute = LEVEL_NUMBER_TO_MEASURE[level_num]
            else:
                substitute = "Employees_Head_Count"

            rewritten_measures.append(substitute)

            if is_male or is_female:
                # Gender breakdown: add Gender to groupby and filter to M/F
                if "Gender" not in groupby:
                    add_gender_groupby = True
                gender_val = "M" if is_male else "F"
                if "Gender" not in filters:
                    filters["Gender"] = gender_val
            elif "male" in q or "female" in q or "gender" in q:
                if "Gender" not in groupby:
                    add_gender_groupby = True

        # ── Complete_Span ─────────────────────────────────────────────────
        elif m == "Complete_Span":
            rewritten_measures.append("Employees_Head_Count")

        # ── Layers ───────────────────────────────────────────────────────
        elif m == "Layers":
            is_manager = "manager" in q
            if level_num:
                if is_manager and level_num in LEVEL_NUMBER_TO_MANAGER_MEASURE:
                    rewritten_measures.append(LEVEL_NUMBER_TO_MANAGER_MEASURE[level_num])
                elif level_num in LEVEL_NUMBER_TO_MEASURE:
                    rewritten_measures.append(LEVEL_NUMBER_TO_MEASURE[level_num])
                else:
                    rewritten_measures.append("Employees_Head_Count")
            elif groupby:
                # Grouped by some dimension → just count employees per group
                rewritten_measures.append("Employees_Head_Count")
            else:
                # No level context → expand to all Level_1…Level_10 measures
                # so the caller gets a side-by-side row of all levels
                all_level_measures = [f"Level_{n}" for n in range(1, 11)]
                rewritten_measures.extend(all_level_measures)

        # ── Analyse ──────────────────────────────────────────────────────
        elif m == "Analyse":
            rewritten_measures.append("Employees_Head_Count")

        else:
            # Unknown SELECTEDVALUE measure — safest fallback
            rewritten_measures.append("Employees_Head_Count")

    if not rewrote:
        return spec   # nothing changed

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for m in rewritten_measures:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    rewritten_measures = deduped

    if add_gender_groupby and "Gender" not in groupby:
        groupby.append("Gender")

    # If the user asked for a specific level entity groupby (e.g. "Level_6
    # with highest direct reportee count") and no groupby is set yet, add the
    # Level-Name column for that level so results are grouped meaningfully.
    if level_num and not groupby and level_num in LEVEL_NUMBER_TO_NAME_COL:
        groupby.append(LEVEL_NUMBER_TO_NAME_COL[level_num])
        # If there's a top_n from the original intent keep it; otherwise
        # if the question says "highest"/"top" default to top 1
        if top_n is None and any(w in q for w in ("highest", "most", "top", "largest")):
            top_n = 1
            sort = {"by": rewritten_measures[0], "order": "desc"}

    new_spec = dict(spec)
    new_spec["measures"] = rewritten_measures
    new_spec["groupby"]  = groupby
    new_spec["filters"]  = filters
    new_spec["top_n"]    = top_n
    new_spec["sort"]     = sort
    return new_spec

# ============================================================

BUSINESS_RULES = {
    "default_year":  CURRENT_YEAR,
    "default_month": MONTH_NUMBER_TO_NAME[CURRENT_MONTH],  # e.g. "June"
}

# Month names for suppression-signal detection
_MONTH_NAMES = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
]
_MONTH_SUPPRESS_SIGNALS = [
    "by month", "per month", "each month", "all months",
    "monthly", "month over month", "trend", "over time",
    "ytd", "year to date", "full year",
]


def apply_soc_defaults(question: str, filters: Dict) -> Dict:
    """
    Inject Year and Month defaults into a SOC filter dict if not already set.

    Rules:
    • Year  — defaults to CURRENT_YEAR unless the question contains a 4-digit
              year (e.g. "2025") or already has Year in filters.
    • Month — 'Dim Cal'[Month] stores the full month NAME string (e.g. "June"),
              NOT a number. The default is MONTH_NUMBER_TO_NAME[CURRENT_MONTH].
              Suppressed when:
              - Month is already in filters (user/LLM set it explicitly)
              - The question asks for trend/all-months analysis
              - The question mentions an explicit month name or number
              - 'Dim Cal' does not have a Month column (safety check)

    Called from build_soc_dax_node (standard/topn/comparison) and from
    plan_soc_workflow_node (applied per step before validation).
    """
    filters = dict(filters)
    q_lower = question.lower()

    # ── Year ──────────────────────────────────────────────────────────────
    if "Year" not in filters:
        year_match = re.search(r"\b(20\d{2})\b", question)
        filters["Year"] = (int(year_match.group(1)) if year_match
                           else BUSINESS_RULES["default_year"])

    # ── Month (name string, e.g. "June") ──────────────────────────────────
    # Only inject if 'Dim Cal' actually has a Month column.
    if "Month" not in filters and "Month" in DIM_CAL_COLS:
        explicit_month = (
            any(m in q_lower for m in _MONTH_NAMES)
            or bool(re.search(r"\bmonth\s*\d{1,2}\b", q_lower))
        )
        suppress = (
            any(sig in q_lower for sig in _MONTH_SUPPRESS_SIGNALS)
            or explicit_month
        )
        if not suppress:
            # Store the name string — 'Dim Cal'[Month] is a text column
            filters["Month"] = BUSINESS_RULES["default_month"]   # e.g. "June"

    return filters



llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=API_VERSION,
)


# ============================================================
# GRAPH STATE
# ============================================================

class SOCState(TypedDict):
    question:           str
    intent_type:        str          # "standard" | "comparison" | "topn" | "workflow"
    intent:             Dict
    retrieved_measures: List[Dict]
    dax_query:          str
    final_dax:          str
    execution_status:   int
    execution_result:   Dict
    answer:             str
    repair_attempts:    int
    error_messages:     Annotated[List[str], operator.add]
    # ── Workflow fields ────────────────────────────────────────────────────
    workflow_plan:                   List[Dict]   # ordered step specs from planner
    workflow_results:                Dict         # {step_id: {description, result}}
    workflow_dax:                    Dict         # {step_id: final_dax_string}
    workflow_failed_step_id:         Optional[str]
    workflow_failed_step_index:      Optional[int]
    workflow_step_error:             Optional[Dict]
    workflow_step_repair_dax:        Optional[str]
    workflow_step_repair_attempts:   int


# ============================================================
# COLUMN / TABLE UTILITIES
# ============================================================

def qualify_table(table: str) -> str:
    return f"'{table}'" if " " in table else table


def resolve_col(col: str) -> str:
    """
    Return fully-qualified 'Table'[Column] DAX reference for any column
    in soc_dimensions.json.  Raises ValueError on unknown columns so
    hallucinated names are caught before DAX generation.
    """
    tbl = SOC_COLUMN_TO_TABLE.get(col)
    if not tbl:
        raise ValueError(f"Unknown SOC column: '{col}'")
    return f"{qualify_table(tbl)}[{col}]"


def get_soc_dimension_context() -> str:
    lines = []
    for table, cols in SOC_DIMENSIONS["tables"].items():
        lines.append(f"\nTABLE: {table}")
        lines.extend(f"  {c}" for c in cols)
    return "\n".join(lines)


# ============================================================
# MEASURE RETRIEVAL  (same lexical scoring as v6)
# ============================================================

SOC_MEASURE_ALIASES = {
    "managers":                    "Managers_Head_Count",
    "manager count":               "Managers_Head_Count",
    "manager headcount":           "Managers_Head_Count",
    "employees":                   "Employees_Head_Count",
    "employee count":              "Employees_Head_Count",
    "employee headcount":          "Employees_Head_Count",
    "headcount":                   "Employees_Head_Count",
    "fte":                         "Employees_Head_Count",
    "fte count":                   "Employees_Head_Count",
    "full time":                   "Employees_Head_Count",
    "full time employees":         "Employees_Head_Count",
    "full time headcount":         "Employees_Head_Count",
    "span":                        "Average Direct Reportee Span",
    "average span":                "Average Direct Reportee Span",
    "span of control":             "Average Direct Reportee Span",
    "soc":                         "Average Direct Reportee Span",
    "median span":                 "Median_SOC",
    "layers":                      "Layers",
    "direct reportee count":       "Direct_Repotee_Count",
    "reportees":                   "Direct_Repotee_Count",
    "reportee count":              "Direct_Repotee_Count",
    "direct reportees":            "Direct_Repotee_Count",
    "fte reportees":               "Direct_Repotee_Count",
    "full time reportees":         "Direct_Repotee_Count",
    "complete span":               "Complete_Span",
    "span ratio":                  "Span_Ratio",
    "female span":                 "Complete_Span_Female",
    "male span":                   "Complete_Span_Male",
    "headcount by level":          "Employees_Head_Count",
    "headcount by levels":         "Employees_Head_Count",
    "level headcount":             "Employees_Head_Count",
}


def retrieve_soc_measures(question: str, top_k: int = 20) -> List[Dict]:
    q = question.lower()
    scored = []
    for measure in SOC_MEASURES:
        text = (measure.get("measure_name", "") + " " +
                measure.get("description", "")).lower()
        score = 0
        for alias, target in SOC_MEASURE_ALIASES.items():
            if alias in q and target == measure["measure_name"]:
                score += 10
        for token in re.findall(r"[a-z0-9]+", q):
            if len(token) > 2 and token in text:
                score += 1
        scored.append((score, measure))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# ============================================================
# VALIDATION
# ============================================================

def validate_soc_measures(measures: List[str]) -> List[str]:
    """
    Validate measure names against soc_measures.json.
    SELECTEDVALUE-dependent measures are NOT rejected here — they are
    rewritten to API-safe equivalents by rewrite_selectedvalue_spec()
    BEFORE this validator is called. If any slip through (shouldn't happen),
    they will fail at Power BI execution and get repaired by the repair loop.
    """
    errors = []
    for m in measures:
        if m not in ALL_SOC_MEASURE_NAMES:
            errors.append(f"Unknown SOC measure: '{m}'")
        # SELECTEDVALUE check intentionally removed — rewriter handles these.
    return errors


def validate_soc_columns(cols: List[str]) -> List[str]:
    return [f"Unknown SOC column: '{c}'" for c in cols if c not in ALL_SOC_COLUMNS]


# ============================================================
# GENERALIZED DAX FILTER BUILDING  (TREATAS + FILTER/ALL)
# ============================================================
# SOC model uses TREATAS for dimension value filters (e.g. Grade = "M3")
# because the 'Direct Reportee' table is related to other tables and
# TREATAS is the idiomatic cross-filter injection for this model.
# Year / Date filters use FILTER(ALL(...)) on Dim Cal, same as v6.

def build_soc_filter_clause(column: str, value) -> str:
    """
    Single filter clause.
    - Dim Cal columns → FILTER(ALL('Dim Cal'[col]), ...) idiom
    - Everything else → TREATAS({value}, 'Table'[col]) idiom
    Value can be scalar or list.
    """
    col_ref = resolve_col(column)

    # ── Date/year columns: FILTER(ALL(...)) ──────────────────────────────
    if column in DIM_CAL_COLS:
        if isinstance(value, (list, tuple, set)):
            vals = ", ".join(str(v) for v in value)
            return f"FILTER(ALL({col_ref}), {col_ref} IN {{{vals}}})"
        num = isinstance(value, (int, float)) and not isinstance(value, bool)
        fv  = str(value) if num else f'"{value}"'
        return f"FILTER(ALL({col_ref}), {col_ref} = {fv})"

    # ── Dimension columns: TREATAS ────────────────────────────────────────
    tbl = SOC_COLUMN_TO_TABLE[column]
    if isinstance(value, (list, tuple, set)):
        rows = ", ".join(f'{{"{v}"}}' for v in value)
        return f"TREATAS({{{rows}}}, {col_ref})"
    return f'TREATAS({{"{value}"}}, {col_ref})'


def build_soc_filters(filters: Dict) -> List[str]:
    return [build_soc_filter_clause(col, val) for col, val in filters.items()]


def extract_soc_column_values(result: Dict, column_label: str) -> List:
    """
    Pull distinct non-null values of `column_label` from a Power BI
    executeQueries result.  Matches flexibly on column key suffix because
    Power BI prefixes returned column names with the table alias, e.g.
    "'Direct Reportee'[Function Label]" or "[Function Label]".
    Falls back to the first column if no suffix match is found.
    """
    try:
        rows = result["results"][0]["tables"][0]["rows"]
    except Exception:
        return []
    if not rows:
        return []

    matching_key = None
    for key in rows[0].keys():
        if (key == column_label
                or key.endswith(f"[{column_label}]")
                or key.endswith(f".{column_label}")):
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


# ============================================================
# GENERALIZED DAX BUILDER
# ============================================================
# Single renderer for all SOC query shapes, replacing the three
# separate DaxBuilder static methods that were incomplete (no TOPN,
# no sort, no multi-dimension groupby support).

def render_soc_dax(spec: Dict) -> str:
    """
    Render a SOC metric spec to DAX.

    Spec shape:
    {
      "measures":  ["Managers_Head_Count", ...],
      "groupby":   ["Function Label"],        # optional
      "filters":   {"Grade": "M3"},           # optional
      "top_n":     10,                        # optional
      "sort":      {"by": "Average Direct Reportee Span", "order": "desc"}
    }

    Three output shapes:
      - No groupby, single measure      → ROW(...)
      - No groupby, multiple measures   → ROW(..., ..., ...)
      - groupby present                 → SUMMARIZECOLUMNS(...)
                                          wrapped in TOPN if top_n set
    """
    measures = spec.get("measures") or []
    groupby  = spec.get("groupby")  or []
    filters  = spec.get("filters")  or {}
    top_n    = spec.get("top_n")
    sort     = spec.get("sort")     or {}

    if not measures:
        raise ValueError("SOC spec must have at least one measure")

    filter_clauses = build_soc_filters(filters)

    # ── Scalar (no groupby) ───────────────────────────────────────────────
    if not groupby:
        pairs = []
        for m in measures:
            if filter_clauses:
                expr = f"CALCULATE([{m}], {', '.join(filter_clauses)})"
            else:
                expr = f"[{m}]"
            pairs.append(f'"{m}",\n    {expr}')
        return "EVALUATE\nROW(\n    " + ",\n    ".join(pairs) + "\n)"

    # ── Grouped (SUMMARIZECOLUMNS) ────────────────────────────────────────
    args = []

    # 1. Group-by columns
    for col in groupby:
        args.append(resolve_col(col))

    # 2. Filter clauses
    args.extend(filter_clauses)

    # 3. Measure name + expression pairs
    for m in measures:
        args.append(f'"{m}"')
        args.append(f"[{m}]")

    summarize = "SUMMARIZECOLUMNS(\n    " + ",\n    ".join(args) + "\n)"

    if top_n:
        sort_by    = sort.get("by") or measures[0]
        sort_order = (sort.get("order") or "desc").upper()
        return (
            f"EVALUATE\n"
            f"TOPN(\n"
            f"    {top_n},\n"
            f"    {summarize},\n"
            f"    [{sort_by}], {sort_order}\n"
            f")"
        )

    if sort and sort.get("by"):
        sort_by    = sort["by"]
        sort_order = (sort.get("order") or "desc").upper()
        return (
            f"EVALUATE\n"
            f"CALCULATETABLE(\n"
            f"    {summarize}\n"
            f")\n"
            f"ORDER BY [{sort_by}] {sort_order}"
        )

    return f"EVALUATE\n{summarize}"


# ============================================================
# MASTER INTENT PROMPT  (classify + extract in one LLM call)
# ============================================================

SOC_MASTER_INTENT_PROMPT = ChatPromptTemplate.from_template(
"""
You are a Span-of-Control (SOC) HR Analytics expert.
Given the user question, classify it AND extract the full query spec.

═══════════════════════════════════════════════
USER QUESTION:
{question}
═══════════════════════════════════════════════

TODAY'S YEAR: {current_year}
TODAY'S MONTH: {current_month} (full name, e.g. "June")
NOTE: The system auto-applies Year={current_year} and Month="{current_month}"
as defaults when the user does not specify a period. Do NOT add Year or Month
to the filters yourself — omit them and defaults will be applied automatically.
Only include Year/Month when the user explicitly asks for a different period.
For trend/over-time questions ("by month", "trend", "all months") do NOT
include Month in filters — the system suppresses the month default for those.

AVAILABLE MEASURES (use EXACT names only, never invent one):
{measures}

AVAILABLE TABLES AND COLUMNS (use EXACT names only):
{dimensions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose exactly ONE intent type:

"standard"   — One or more metrics, optionally grouped by a column,
               optionally filtered. No ranking.
               Examples: "Show managers by function",
               "Show average span by grade for M3",
               "Employee headcount by business unit"

"topn"       — Ranking question: Top/Bottom N entities by a metric.
               Examples: "Top 10 managers by span",
               "Bottom 5 functions by employee headcount"

"comparison" — Compare two or more metrics side-by-side WITHOUT a groupby,
               typically scalar totals.
               Examples: "Compare managers and employees",
               "Show manager count vs employee count"

"workflow"   — The question requires MULTIPLE dependent result sets, most
               commonly: select top/bottom N entities FIRST, then drill into
               each of those entities with sub-metric or sub-dimension
               breakdowns. Any question with layered bullet-style asks where
               later asks depend on the result of an earlier ask.
               Examples:
               "Top 5 managers by span — for each show gender breakdown of
                their direct reportees"
               "For the top 3 functions by headcount, show span and manager
                count split by grade"
               "Show me the top 3 BUs by employee count, then for each BU
                show headcount by level"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — EXTRACT SPEC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUSINESS TERM → MEASURE MAPPING (apply these when user uses informal terms):
  Managers / Manager Count / Manager Headcount → Managers_Head_Count
  Employees / Employee Count / Employee Headcount / Headcount → Employees_Head_Count
  FTE / Full Time / Full Time Employees / FTE Count / FTE Headcount → Employees_Head_Count
  Span / Average Span / Span Of Control / SOC → Average Direct Reportee Span
  Median Span → Median_SOC
  Reportees / Direct Reportees / FTE Reportees / Full Time Reportees /
    Direct Reportee Count / Reportee Count → use Level_N measure for the
    level mentioned (e.g. Level_7 employees), NOT Direct_Repotee_Count
  Layers (by level N, employees) → Level_N  (e.g. Level_5, Level_7)
  Layers (by level N, managers)  → LevelN_Manager  (e.g. Level5_Manager)
  Layers (no specific level)     → use Level_1 through Level_10 as comparison
  Female Span → Complete_Span_Female
  Male Span → Complete_Span_Male

CRITICAL — SLICER-ONLY MEASURES (NEVER use these, use substitutes below):
  Direct_Repotee_Count       → requires a single manager to be selected in a
                               visual slicer. NOT usable via API.
                               SUBSTITUTE: Level_N where N is the level in the
                               question, or Employees_Head_Count otherwise.
  Direct_Reportee_Count_Male → same restriction. SUBSTITUTE: Employees_Head_Count
                               with groupby Gender.
  Direct_Reportee_Count_Female → same. SUBSTITUTE: Employees_Head_Count with
                               groupby Gender.
  Complete_Span              → slicer-only. SUBSTITUTE: Employees_Head_Count.
  Layers                     → slicer-only switch. SUBSTITUTE: Level_N or
                               LevelN_Manager depending on context.
  Analyse                    → slicer-only. SUBSTITUTE: Employees_Head_Count.

  If you find yourself about to use any of the above, stop and use the
  substitute measure instead.

BUSINESS TERM → COLUMN MAPPING (apply these for groupby and filters):
  Function / Function Label → "Function Label"  (in 'Direct Reportee')
  Business Unit / BU → "Business Unit Name"  (in 'Direct Reportee')
  Grade → "Grade"  (in 'Direct Reportee')
  Gender → "Gender"  (in 'Direct Reportee')
  Layer → "Layer"  (in 'Direct Reportee')
  Manager Name / Manager → "Manager full name"  (in 'Direct Reportee')
  Level / Levels / Level Name / Employee Level → "Employee Level"  (in 'Direct Reportee')
  Level 1 / L1 → "Level 1"  (in 'Direct Reportee')
  Level 2 / L2 → "Level 2"  (in 'Direct Reportee')
  Level 3 / L3 → "Level 3"  (in 'Direct Reportee')
  (Level N → "Level N" for any N, with a space before the number)
  Year → "Year"  (in 'Dim Cal')

RULES:
1. measures → list of EXACT measure names from the list above. Required.
2. groupby  → list of EXACT column names. Use [] for scalar / comparison.
3. filters  → dict of {{column: value}}. Use {{}} if none.
4. top_n    → integer or null.
5. sort     → {{"by": "measure name", "order": "asc"|"desc"}} or null.
6. NEVER put "Employment Type" or "IS ACTIVE" in filters — the measures
   already handle Full Time / active filtering internally. Adding them
   causes incorrect DAX.
7. NEVER invent a measure name. "FTE" is a column, NOT a measure —
   use "Employees_Head_Count" for FTE/full-time headcount instead.
8. NEVER use "Level Name" or "Levels" as a groupby column — they do not
   exist. Use "Employee Level" for level grouping, or "Level 1", "Level 2",
   etc. (with a space) for specific level hierarchy columns.
9. For a "list of managers and their reportees" style query, use:
   measures=["Managers_Head_Count","Direct_Repotee_Count"],
   groupby=["Manager full name"].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON ONLY, no markdown, no commentary.

For "workflow" — return only intent_type + reasoning (planner handles the rest):
{{
    "intent_type": "workflow",
    "reasoning":   "one sentence"
}}

For "standard" / "topn" / "comparison":
{{
    "intent_type": "standard" | "topn" | "comparison",
    "reasoning":   "one sentence",
    "measures":    ["Exact Measure Name"],
    "groupby":     ["Exact Column Name"],
    "filters":     {{"Column": "value"}},
    "top_n":       null,
    "sort":        null
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Show managers by function
A: {{"intent_type":"standard","reasoning":"Managers grouped by function.","measures":["Managers_Head_Count"],"groupby":["Function Label"],"filters":{{}},"top_n":null,"sort":null}}

Q: Show average span by grade for grade M3
A: {{"intent_type":"standard","reasoning":"Span grouped by grade, filtered to M3.","measures":["Average Direct Reportee Span"],"groupby":["Grade"],"filters":{{"Grade":"M3"}},"top_n":null,"sort":null}}

Q: Top 10 managers by span
A: {{"intent_type":"topn","reasoning":"Top N ranking of managers by span.","measures":["Average Direct Reportee Span"],"groupby":["Manager full name"],"filters":{{}},"top_n":10,"sort":{{"by":"Average Direct Reportee Span","order":"desc"}}}}

Q: Compare managers and employees
A: {{"intent_type":"comparison","reasoning":"Side-by-side scalar totals, no groupby.","measures":["Managers_Head_Count","Employees_Head_Count"],"groupby":[],"filters":{{}},"top_n":null,"sort":null}}

Q: Show employee headcount grouped by Level 1
A: {{"intent_type":"standard","reasoning":"Employee headcount grouped by Level 1 hierarchy.","measures":["Employees_Head_Count"],"groupby":["Level 1"],"filters":{{}},"top_n":null,"sort":null}}

Q: Show managers by function for grade M3
A: {{"intent_type":"standard","reasoning":"Managers grouped by function filtered to grade M3.","measures":["Managers_Head_Count"],"groupby":["Function Label"],"filters":{{"Grade":"M3"}},"top_n":null,"sort":null}}

Q: Bottom 5 functions by employee headcount
A: {{"intent_type":"topn","reasoning":"Bottom 5 ranking of functions.","measures":["Employees_Head_Count"],"groupby":["Function Label"],"filters":{{}},"top_n":5,"sort":{{"by":"Employees_Head_Count","order":"asc"}}}}

Q: Show manager and employee headcount by business unit
A: {{"intent_type":"standard","reasoning":"Two metrics grouped by BU.","measures":["Managers_Head_Count","Employees_Head_Count"],"groupby":["Business Unit Name"],"filters":{{}},"top_n":null,"sort":null}}

Q: list of managers and their Full Time reportees
A: {{"intent_type":"standard","reasoning":"Managers with their direct reportee count grouped by manager name. Reportees uses Direct_Repotee_Count which already filters Full Time internally.","measures":["Managers_Head_Count","Direct_Repotee_Count"],"groupby":["Manager full name"],"filters":{{}},"top_n":null,"sort":null}}

Q: list of managers and their FTE reportees
A: {{"intent_type":"standard","reasoning":"Same as above — FTE reportees maps to Direct_Repotee_Count.","measures":["Managers_Head_Count","Direct_Repotee_Count"],"groupby":["Manager full name"],"filters":{{}},"top_n":null,"sort":null}}

Q: tell me the headcount by levels
A: {{"intent_type":"standard","reasoning":"Employee headcount grouped by Employee Level column.","measures":["Employees_Head_Count"],"groupby":["Employee Level"],"filters":{{}},"top_n":null,"sort":null}}

Q: Compare Direct Reportee Count for Male vs Female employees at Level 7
A: {{"intent_type":"standard","reasoning":"Level 7 employee count split by gender. Direct_Repotee_Count is slicer-only; use Level_7 measure grouped by Gender instead.","measures":["Level_7"],"groupby":["Gender"],"filters":{{}},"top_n":null,"sort":null}}

Q: Display the distribution of Layers across all levels
A: {{"intent_type":"comparison","reasoning":"Layers is slicer-only; compare Level_1 through Level_10 measures as a scalar row instead.","measures":["Level_1","Level_2","Level_3","Level_4","Level_5","Level_6","Level_7","Level_8","Level_9","Level_10"],"groupby":[],"filters":{{}},"top_n":null,"sort":null}}

Q: Display the distribution of Layers by Level 5
A: {{"intent_type":"standard","reasoning":"Layers is slicer-only; use Level_5 employee count grouped by Level5 Name instead.","measures":["Level_5"],"groupby":["Level5 Name"],"filters":{{}},"top_n":null,"sort":null}}

Q: Identify the Level 6 with the highest Direct Reportee Count
A: {{"intent_type":"topn","reasoning":"Direct_Repotee_Count is slicer-only; use Level_6 measure grouped by Level6 Name, top 1 descending.","measures":["Level_6"],"groupby":["Level6 Name"],"filters":{{}},"top_n":1,"sort":{{"by":"Level_6","order":"desc"}}}}

Q: Top 3 functions by headcount — for each show span and manager count by grade
A: {{"intent_type":"workflow","reasoning":"First selects top-3 functions, then drills into each with dependent sub-breakdowns."}}

Return JSON ONLY.
"""
)


def extract_soc_intent(question: str, measures: List[Dict], dimensions: str) -> Dict:
    response = llm.invoke(
        SOC_MASTER_INTENT_PROMPT.format(
            question=question,
            current_year=CURRENT_YEAR,
            current_month=MONTH_NUMBER_TO_NAME[CURRENT_MONTH],  # e.g. "June"
            measures=json.dumps([m["measure_name"] for m in measures], indent=2),
            dimensions=dimensions,
        )
    )
    content = response.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(content)


# ============================================================
# WORKFLOW PLANNING PROMPT
# ============================================================
# For complex multi-step SOC questions the LLM proposes the full ordered
# list of metric-spec steps. Python validates, resolves entity dependencies
# (e.g. top-3 function names threading into dependent steps via TREATAS),
# renders DAX, and executes each step with its own repair loop.
#
# Each step is a standard SOC metric spec PLUS dependency fields:
#   depends_on        — step_id whose first groupby column provides the
#                       entity list to inject into this step's filters
#   depends_on_column — which column of the prior step holds entity values
#   entity_column     — column in THIS step to filter via those entities
#
# Entity injection uses TREATAS (the SOC model's filter idiom), consistent
# with how all other dimension filters are expressed in this model.

MAX_WORKFLOW_STEPS = 6

SOC_WORKFLOW_PLAN_PROMPT = ChatPromptTemplate.from_template(
"""
You are a Span-of-Control (SOC) HR Analytics workflow planner.
The user's question requires MULTIPLE dependent DAX queries. Break it into
an ordered list of steps, where later steps can filter their results to the
entities produced by earlier steps.

═══════════════════════════════════════════════
USER QUESTION:
{question}
═══════════════════════════════════════════════

AVAILABLE MEASURES (EXACT names only, never invent one):
{measures}

AVAILABLE TABLES AND COLUMNS (EXACT names only):
{dimensions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEASURES
• EXACT names only. Never invent a measure.
• Managers/Manager Count → Managers_Head_Count
• Employees/Headcount → Employees_Head_Count
• Span/Average Span/SOC → Average Direct Reportee Span
• Median Span → Median_SOC

COLUMNS
• EXACT names only. Never invent a column.
• Function Label, Business Unit Name, Grade, Gender, Layer,
  Manager full name, Level 1 … Level 9 (with space), Year
• DO NOT add Employment Type or IS ACTIVE to filters.

FILTERS
• Dimension value filters: express as {{"column": "value"}} — the renderer
  will wrap these in TREATAS automatically.
• Year filters: {{"Year": 2026}} — the renderer uses FILTER(ALL(...)).

DEPENDENCIES
• The FIRST step is usually the "select top-N entities" step.
• Later steps that should be restricted to those entities MUST set:
    "depends_on":        "<step_id of the entity-producing step>"
    "depends_on_column": "<groupby column of that step, e.g. 'Function Label'>"
    "entity_column":     "<column in THIS step to filter by those entities>"
• Steps without an entity dependency set all three to null.
• Steps are executed in order; depends_on must reference an earlier step_id.
• Max {max_steps} steps total.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON ONLY, no markdown.

{{
  "steps": [
    {{
      "step_id":           "top_functions",
      "description":       "Top 3 functions by employee headcount",
      "measures":          ["Employees_Head_Count"],
      "groupby":           ["Function Label"],
      "filters":           {{}},
      "top_n":             3,
      "sort":              {{"by": "Employees_Head_Count", "order": "desc"}},
      "depends_on":        null,
      "depends_on_column": null,
      "entity_column":     null
    }},
    {{
      "step_id":           "function_span_by_grade",
      "description":       "Span and manager count by grade for top functions",
      "measures":          ["Average Direct Reportee Span", "Managers_Head_Count"],
      "groupby":           ["Function Label", "Grade"],
      "filters":           {{}},
      "top_n":             null,
      "sort":              null,
      "depends_on":        "top_functions",
      "depends_on_column": "Function Label",
      "entity_column":     "Function Label"
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Top 5 managers by span — for each show gender breakdown of direct reportees

Steps:
1. step_id="top_managers": Top 5 managers by Average Direct Reportee Span,
   groupby=["Manager full name"], top_n=5, sort desc. depends_on=null.
2. step_id="manager_gender_breakdown": Employee count by gender for those
   managers. groupby=["Manager full name","Gender"],
   measures=["Employees_Head_Count"], depends_on="top_managers",
   depends_on_column="Manager full name", entity_column="Manager full name".

Q: For the top 3 functions by headcount, show headcount by level and also
   average span by grade

Steps:
1. step_id="top_functions": Top 3 functions by Employees_Head_Count.
   groupby=["Function Label"], top_n=3, sort desc. depends_on=null.
2. step_id="headcount_by_level": Headcount by Level within top functions.
   groupby=["Function Label","Level 1"], measures=["Employees_Head_Count"],
   depends_on="top_functions", depends_on_column="Function Label",
   entity_column="Function Label".
3. step_id="span_by_grade": Average span by grade within top functions.
   groupby=["Function Label","Grade"], measures=["Average Direct Reportee Span"],
   depends_on="top_functions", depends_on_column="Function Label",
   entity_column="Function Label".

Return JSON ONLY.
"""
)


def plan_soc_workflow(question: str, measures: List[Dict], dimensions: str) -> List[Dict]:
    response = llm.invoke(
        SOC_WORKFLOW_PLAN_PROMPT.format(
            question=question,
            measures=json.dumps([m["measure_name"] for m in measures], indent=2),
            dimensions=dimensions,
            max_steps=MAX_WORKFLOW_STEPS,
        )
    )
    content = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed  = json.loads(content)
    steps   = parsed.get("steps", [])
    if not steps:
        raise ValueError("Workflow planner returned no steps")
    if len(steps) > MAX_WORKFLOW_STEPS:
        raise ValueError(
            f"Workflow planner returned {len(steps)} steps, cap is {MAX_WORKFLOW_STEPS}"
        )
    return steps


def validate_soc_workflow_plan(steps: List[Dict]) -> List[str]:
    errors   = []
    seen_ids = set()
    for i, step in enumerate(steps):
        sid = step.get("step_id")
        if not sid:
            errors.append(f"Step {i} missing step_id")
            continue
        if sid in seen_ids:
            errors.append(f"Duplicate step_id: '{sid}'")
        seen_ids.add(sid)

        # Measures
        errors.extend(
            f"[{sid}] {e}"
            for e in validate_soc_measures(step.get("measures") or [])
        )
        # Columns
        errors.extend(
            f"[{sid}] {e}"
            for e in validate_soc_columns(
                (step.get("groupby") or []) +
                list((step.get("filters") or {}).keys())
            )
        )
        # Dependency ordering
        dep = step.get("depends_on")
        if dep and dep not in seen_ids:
            errors.append(
                f"[{sid}] depends_on '{dep}' is not an earlier step_id"
            )
        if dep:
            if not step.get("entity_column"):
                errors.append(f"[{sid}] depends_on set but entity_column missing")
            elif step["entity_column"] not in ALL_SOC_COLUMNS:
                errors.append(
                    f"[{sid}] entity_column '{step['entity_column']}' is not a known column"
                )
    return errors


# ============================================================
# REPAIR PROMPT  (same structure as v6 REPAIR_PROMPT)
# ============================================================

SOC_REPAIR_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert Power BI DAX developer specialising in Span-of-Control models.

The following DAX query failed when sent to the Power BI executeQueries API.

ORIGINAL DAX:
{dax}

POWER BI ERROR:
{error}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

RULES:
1. Fix ONLY the DAX — preserve the original intent exactly.
2. Use ONLY measure names from AVAILABLE MEASURES.
3. Use ONLY column names from AVAILABLE DIMENSIONS.
4. For dimension-value filters use TREATAS({{"value"}}, 'Table'[Column]).
5. For Year / Dim Cal filters use FILTER(ALL('Dim Cal'[Year]), ...).
6. Return DAX ONLY — no markdown, no explanation, no commentary.
"""
)


def repair_soc_dax(dax: str, error: Dict, measures: List[Dict], dimensions: str) -> str:
    response = llm.invoke(
        SOC_REPAIR_PROMPT.format(
            dax=dax,
            error=json.dumps(error, indent=2),
            measures=json.dumps([m["measure_name"] for m in measures], indent=2),
            dimensions=dimensions,
        )
    )
    fixed = response.content.strip().replace("```DAX", "").replace("```dax", "").replace("```", "")
    return fixed.strip()


# ============================================================
# ANSWER GENERATION PROMPT
# ============================================================

SOC_ANSWER_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert HR Analyst specialising in Span-of-Control (SOC) reporting.

User Question:
{question}

Analysis Results:
{result}

RULES:
1. Answer using ONLY the provided results — never invent numbers.
2. If results are empty, say so plainly.
3. Use clear business language; use a table or bullets where it aids clarity.
4. Do not mention DAX, Power BI, or internal technical details.
5. Percentages and ratios: use as given.

Return the business answer only.
"""
)


def generate_soc_answer(question: str, result: Dict) -> str:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        if not rows:
            result_text = "No records returned."
        else:
            lines = []
            for row in rows:
                lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            result_text = "\n".join(lines)
    except Exception:
        result_text = json.dumps(result, indent=2)

    response = llm.invoke(
        SOC_ANSWER_PROMPT.format(question=question, result=result_text)
    )
    return response.content


# ============================================================
# POWER BI EXECUTION
# ============================================================

def execute_soc_dax_query(dax_query: str):
    """Execute a DAX query against Power BI, return (status_code, body)."""
    credential   = InteractiveBrowserCredential()
    token        = credential.get_token(
        "https://analysis.windows.net/powerbi/api/.default"
    ).token

    response = requests.post(
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{DATASET_ID}/executeQueries",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        json={
            "queries": [{"query": dax_query}],
            "serializerSettings": {"includeNulls": True},
        },
        verify=False,
    )

    try:
        body = response.json()
    except Exception:
        body = {"error": response.text}

    return response.status_code, body


# ============================================================
# LANGGRAPH NODES
# ============================================================

def retrieve_soc_measures_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 1: RETRIEVING MEASURES =====")
    retrieved = retrieve_soc_measures(state["question"], top_k=20)
    print("Retrieved:", [m["measure_name"] for m in retrieved])
    state["retrieved_measures"] = retrieved
    return state


def extract_soc_intent_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 2: CLASSIFYING INTENT + EXTRACTING SPEC =====")
    dimensions_ctx = get_soc_dimension_context()
    try:
        parsed = extract_soc_intent(
            state["question"], state["retrieved_measures"], dimensions_ctx
        )
    except Exception as e:
        state["error_messages"] = [f"SOC intent extraction failed: {e}"]
        return state

    intent_type = parsed.get("intent_type", "standard")
    print(f"Classified as: {intent_type} — {parsed.get('reasoning', '')}")

    state["intent_type"] = intent_type
    state["intent"]      = parsed
    return state


def build_soc_dax_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 3: BUILDING DAX =====")

    # Columns that the SOC measures already handle internally — never let the
    # LLM inject them as step-wide filters or they corrupt DAX with double-
    # filtering and TREATAS on restricted sets.
    INTERNAL_FILTER_COLS = {"Employment Type", "IS ACTIVE", "IS Manager",
                            "Employment Type (Complete Span)"}

    try:
        spec = dict(state["intent"])

        # ── Strip columns the measures handle internally ──────────────────
        raw_filters = spec.get("filters") or {}
        cleaned_filters = {k: v for k, v in raw_filters.items()
                           if k not in INTERNAL_FILTER_COLS}
        if len(cleaned_filters) != len(raw_filters):
            stripped = set(raw_filters) - set(cleaned_filters)
            print(f"[AUTO-FIX] Stripped internal-filter columns from filters: {stripped}")

        # ── Sanitise measures — replace known bad hallucinations ──────────
        # The LLM sometimes picks 'FTE' (a column name) instead of a measure.
        # Map it to the correct measure before validation.
        MEASURE_FIXUPS = {
            "FTE":                    "Employees_Head_Count",
            "Direct Reportee Count":  "Direct_Repotee_Count",
            "Reportees":              "Direct_Repotee_Count",
            "Direct Reportees":       "Direct_Repotee_Count",
            "FTE Reportees":          "Direct_Repotee_Count",
            "Full Time Reportees":    "Direct_Repotee_Count",
            "Employee Level Count":   "Employees_Head_Count",
        }
        raw_measures = spec.get("measures") or []
        fixed_measures = [MEASURE_FIXUPS.get(m, m) for m in raw_measures]
        if fixed_measures != raw_measures:
            print(f"[AUTO-FIX] Measure(s) corrected: {raw_measures} → {fixed_measures}")

        # ── Sanitise groupby — replace known bad column hallucinations ─────
        GROUPBY_FIXUPS = {
            "Level Name":          "Employee Level",
            "Level":               "Employee Level",
            "Levels":              "Employee Level",
            "Level_Name":          "Employee Level",
            "Layer Name":          "Layer",
        }
        raw_groupby = spec.get("groupby") or []
        fixed_groupby = [GROUPBY_FIXUPS.get(c, c) for c in raw_groupby]
        if fixed_groupby != raw_groupby:
            print(f"[AUTO-FIX] Groupby column(s) corrected: {raw_groupby} → {fixed_groupby}")

        spec["measures"] = fixed_measures
        spec["groupby"]  = fixed_groupby
        spec["filters"]  = cleaned_filters

        # ── Rewrite SELECTEDVALUE-dependent measures ───────────────────────
        # Must run BEFORE validate_soc_measures so the validator never sees
        # the unrewritable measures and immediately errors out.
        spec = rewrite_selectedvalue_spec(spec, state["question"])

        # ── Apply Year / Month defaults ───────────────────────────────────
        spec["filters"] = apply_soc_defaults(
            state["question"], spec["filters"]
        )
        print(f"Filters after defaults: {spec['filters']}")

        # ── Validate measures ─────────────────────────────────────────────
        measure_errors = validate_soc_measures(spec["measures"])
        if measure_errors:
            state["error_messages"] = state.get("error_messages", []) + measure_errors
            return state

        # ── Validate groupby + filter columns ────────────────────────────
        col_errors = validate_soc_columns(
            spec["groupby"] + list(spec["filters"].keys())
        )
        if col_errors:
            state["error_messages"] = state.get("error_messages", []) + col_errors
            return state

        dax = render_soc_dax(spec)
        print("Generated DAX:\n", dax)

        state["dax_query"]       = dax
        state["final_dax"]       = dax
        state["repair_attempts"] = 0

    except Exception as e:
        state["error_messages"] = state.get("error_messages", []) + [
            f"SOC DAX build failed: {e}"
        ]
    return state


def execute_soc_dax_node(state: SOCState) -> SOCState:
    print(f"\n===== SOC STEP 4: EXECUTING DAX "
          f"(Attempt {state['repair_attempts'] + 1}) =====")
    print("DAX:\n", state["final_dax"])

    status, result = execute_soc_dax_query(state["final_dax"])
    print(f"Status: {status}")

    state["execution_status"] = status
    state["execution_result"] = result
    return state


def repair_soc_dax_node(state: SOCState) -> SOCState:
    print(f"\n===== SOC STEP 5: REPAIRING DAX "
          f"(Attempt {state['repair_attempts'] + 1}/{MAX_REPAIR_ATTEMPTS}) =====")

    repaired = repair_soc_dax(
        state["final_dax"],
        state["execution_result"],
        state["retrieved_measures"],
        get_soc_dimension_context(),
    )
    print("Repaired DAX:\n", repaired)

    state["final_dax"]       = repaired
    state["repair_attempts"] += 1
    return state


def plan_soc_workflow_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 3W: PLANNING WORKFLOW =====")
    try:
        steps = plan_soc_workflow(
            state["question"],
            state["retrieved_measures"],
            get_soc_dimension_context(),
        )

        # ── Auto-fix + apply defaults to every step ───────────────────────
        INTERNAL_FILTER_COLS = {"Employment Type", "IS ACTIVE", "IS Manager",
                                "Employment Type (Complete Span)"}
        MEASURE_FIXUPS = {
            "FTE":                   "Employees_Head_Count",
            "Direct Reportee Count": "Direct_Repotee_Count",
            "Reportees":             "Direct_Repotee_Count",
            "Direct Reportees":      "Direct_Repotee_Count",
            "FTE Reportees":         "Direct_Repotee_Count",
            "Full Time Reportees":   "Direct_Repotee_Count",
            "Employee Level Count":  "Employees_Head_Count",
        }
        GROUPBY_FIXUPS = {
            "Level Name":  "Employee Level",
            "Level":       "Employee Level",
            "Levels":      "Employee Level",
            "Level_Name":  "Employee Level",
            "Layer Name":  "Layer",
        }
        for step in steps:
            step["filters"]  = {k: v for k, v in (step.get("filters") or {}).items()
                                if k not in INTERNAL_FILTER_COLS}
            step["measures"] = [MEASURE_FIXUPS.get(m, m) for m in (step.get("measures") or [])]
            step["groupby"]  = [GROUPBY_FIXUPS.get(c, c) for c in (step.get("groupby") or [])]
            # Rewrite SELECTEDVALUE-dependent measures before validation
            step = rewrite_selectedvalue_spec(step, state["question"])
            step["filters"]  = apply_soc_defaults(
                state["question"], step["filters"]
            )

        errors = validate_soc_workflow_plan(steps)
        if errors:
            state["error_messages"] = state.get("error_messages", []) + errors
            return state

        print(f"Planned {len(steps)} steps:")
        for s in steps:
            dep = f" (depends_on={s['depends_on']})" if s.get("depends_on") else ""
            print(f"  [{s['step_id']}] {s.get('description','')}{dep} | filters: {s['filters']}")

        state["workflow_plan"]                 = steps
        state["workflow_results"]              = {}
        state["workflow_dax"]                  = {}
        state["workflow_failed_step_id"]       = None
        state["workflow_failed_step_index"]    = None
        state["workflow_step_error"]           = None
        state["workflow_step_repair_dax"]      = None
        state["workflow_step_repair_attempts"] = 0

    except Exception as e:
        state["error_messages"] = state.get("error_messages", []) + [
            f"SOC workflow planning failed: {e}"
        ]
    return state


def execute_soc_workflow_node(state: SOCState) -> SOCState:
    """
    Executes each workflow step in order.

    On first entry: runs all steps from the beginning.
    On re-entry after a repair: skips already-completed steps, resumes at
    the failed step using the repaired DAX from workflow_step_repair_dax,
    then continues with any remaining steps.

    Entity dependency injection uses TREATAS (the SOC filter idiom),
    consistent with all other dimension filters in this model.
    """
    print("\n===== SOC STEP 4W: EXECUTING WORKFLOW =====")

    # ── Restore prior completed results on resume ────────────────────────
    prior = state.get("workflow_results") or state.get("execution_result") or {}
    workflow_results = dict(prior)
    if "workflow_steps" not in workflow_results:
        workflow_results["workflow_steps"] = {}
    workflow_dax = dict(state.get("workflow_dax") or {})

    # Re-build entity lists for steps that already ran
    step_entities: Dict[str, List] = {}
    for step in state["workflow_plan"]:
        sid = step["step_id"]
        if sid in workflow_results["workflow_steps"]:
            gb = step.get("groupby") or []
            if gb:
                step_entities[sid] = extract_soc_column_values(
                    workflow_results["workflow_steps"][sid]["result"], gb[0]
                )

    resume_step_id = state.get("workflow_failed_step_id")
    resume_dax     = state.get("workflow_step_repair_dax")
    is_resume      = bool(resume_step_id and resume_dax)

    for idx, step in enumerate(state["workflow_plan"]):
        step_id = step["step_id"]

        # Skip steps already completed
        if step_id in workflow_results["workflow_steps"]:
            print(f"--- Skipping already-completed step: {step_id} ---")
            continue

        print(f"\n--- Executing step: {step_id} ({step.get('description','')}) ---")

        # ── Resolve entity dependency via TREATAS ────────────────────────
        entity_filter: Optional[Dict] = None      # extra filter dict for render_soc_dax
        depends_on = step.get("depends_on")
        if depends_on:
            entity_values = step_entities.get(depends_on)
            if not entity_values:
                state["error_messages"] = state.get("error_messages", []) + [
                    f"Step '{step_id}' depends on '{depends_on}' but no "
                    f"entities were captured from it."
                ]
                state["execution_status"] = 500
                state["execution_result"] = workflow_results
                state["workflow_results"] = workflow_results
                state["workflow_dax"]     = workflow_dax
                state["workflow_failed_step_id"] = None
                state["workflow_step_repair_dax"] = None
                return state
            # Inject entity list as an extra filter so TREATAS wraps it
            entity_col = step["entity_column"]
            entity_filter = {entity_col: entity_values}

        # ── Build DAX ────────────────────────────────────────────────────
        if is_resume and step_id == resume_step_id:
            dax_query = resume_dax
            print(f"[RESUME] Using repaired DAX for '{step_id}'")
            is_resume = False
        else:
            try:
                merged_filters = dict(step.get("filters") or {})
                if entity_filter:
                    merged_filters.update(entity_filter)
                spec = dict(step)
                spec["filters"] = merged_filters
                dax_query = render_soc_dax(spec)
            except Exception as e:
                state["error_messages"] = state.get("error_messages", []) + [
                    f"Step '{step_id}' DAX build failed: {e}"
                ]
                state["execution_status"] = 500
                state["execution_result"] = workflow_results
                state["workflow_results"] = workflow_results
                state["workflow_dax"]     = workflow_dax
                state["workflow_failed_step_id"] = None
                state["workflow_step_repair_dax"] = None
                return state

        print(f"DAX:\n{dax_query}")
        workflow_dax[step_id] = dax_query

        # ── Execute ──────────────────────────────────────────────────────
        status, result = execute_soc_dax_query(dax_query)
        print(f"Status: {status}")

        if status != 200:
            print(f"Step '{step_id}' failed — routing to repair node")
            state["execution_status"]          = 500
            state["execution_result"]          = workflow_results
            state["workflow_results"]          = workflow_results
            state["workflow_dax"]              = workflow_dax
            state["workflow_failed_step_id"]   = step_id
            state["workflow_failed_step_index"] = idx
            state["workflow_step_error"]       = result
            state["workflow_step_repair_dax"]  = None
            return state

        # ── Step succeeded ───────────────────────────────────────────────
        workflow_results["workflow_steps"][step_id] = {
            "description": step.get("description", step_id),
            "result":      result,
        }
        gb = step.get("groupby") or []
        if gb:
            step_entities[step_id] = extract_soc_column_values(result, gb[0])
            print(f"Captured entities from '{step_id}': {step_entities[step_id]}")

        # Clear repair state once a step succeeds
        state["workflow_failed_step_id"]       = None
        state["workflow_step_repair_dax"]      = None
        state["workflow_step_error"]           = None
        state["workflow_step_repair_attempts"] = 0

    # ── All steps done ───────────────────────────────────────────────────
    state["execution_status"] = 200
    state["execution_result"] = workflow_results
    state["workflow_results"] = workflow_results
    state["workflow_dax"]     = workflow_dax
    state["workflow_failed_step_id"]  = None
    state["workflow_step_repair_dax"] = None
    return state


def repair_soc_workflow_step_node(state: SOCState) -> SOCState:
    """
    Repairs one failed workflow step's DAX and stores the fixed version in
    workflow_step_repair_dax so execute_soc_workflow_node can resume at
    exactly that step without re-executing earlier steps.
    """
    failed_step_id   = state.get("workflow_failed_step_id")
    current_attempts = state.get("workflow_step_repair_attempts") or 0

    print(f"\n===== SOC REPAIRING WORKFLOW STEP: '{failed_step_id}' "
          f"(Attempt {current_attempts + 1}/{MAX_REPAIR_ATTEMPTS}) =====")

    if current_attempts >= MAX_REPAIR_ATTEMPTS:
        state["error_messages"] = state.get("error_messages", []) + [
            f"Step '{failed_step_id}' could not be repaired after "
            f"{MAX_REPAIR_ATTEMPTS} attempt(s). "
            f"Last error: {json.dumps(state.get('workflow_step_error', {}))[:400]}"
        ]
        state["execution_status"]    = 500
        state["workflow_step_repair_dax"] = None
        return state

    failing_dax = (state.get("workflow_dax") or {}).get(failed_step_id, "")
    error_body  = state.get("workflow_step_error") or {}

    if not failing_dax:
        state["error_messages"] = state.get("error_messages", []) + [
            f"Repair node: no DAX found for step '{failed_step_id}'"
        ]
        state["execution_status"] = 500
        return state

    print(f"Failing DAX:\n{failing_dax}")
    print(f"PBI Error:\n{json.dumps(error_body, indent=2)[:500]}")

    repaired = repair_soc_dax(
        failing_dax, error_body,
        state["retrieved_measures"],
        get_soc_dimension_context(),
    )
    print(f"Repaired DAX:\n{repaired}")

    state["workflow_step_repair_dax"]      = repaired
    state["workflow_step_repair_attempts"] = current_attempts + 1
    return state


def generate_soc_answer_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 6: GENERATING ANSWER =====")

    result = state["execution_result"]

    # ── Workflow: combine all step results into one text block ────────────
    if isinstance(result, dict) and "workflow_steps" in result:
        text_parts = []
        for step_id, payload in result["workflow_steps"].items():
            desc = payload.get("description", step_id)
            step_result = payload.get("result", {})
            try:
                rows = step_result["results"][0]["tables"][0]["rows"]
                if not rows:
                    rows_text = "No records returned."
                else:
                    rows_text = "\n".join(
                        " | ".join(f"{k}: {v}" for k, v in row.items())
                        for row in rows
                    )
            except Exception:
                rows_text = json.dumps(step_result, indent=2)[:800]
            text_parts.append(f"SECTION — {desc}:\n{rows_text}")
        combined = "\n\n".join(text_parts)

        response = llm.invoke(
            SOC_ANSWER_PROMPT.format(question=state["question"], result=combined)
        )
        state["answer"] = response.content
        return state

    # ── Single query ──────────────────────────────────────────────────────
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        if not rows:
            state["answer"] = "No records returned for the applied filters."
            return state
    except Exception:
        pass

    answer = generate_soc_answer(state["question"], result)
    print("Answer:\n", answer)
    state["answer"] = answer
    return state


def soc_error_handler_node(state: SOCState) -> SOCState:
    print("\n===== SOC ERROR HANDLER =====")
    msg  = "Unable to process the SOC request:\n"
    msg += "\n".join(f"  - {e}" for e in state.get("error_messages", []))
    if state.get("execution_result"):
        detail = json.dumps(state["execution_result"], indent=2)[:1200]
        msg   += f"\n\nLast Power BI response:\n{detail}"
    state["answer"] = msg
    return state


# ============================================================
# ROUTING
# ============================================================

def route_after_intent(state: SOCState) -> str:
    if state.get("error_messages"):
        return "error_handler"
    if state.get("intent_type") == "workflow":
        return "plan_workflow"
    return "build_dax"


def route_after_plan(state: SOCState) -> str:
    if state.get("error_messages"):
        return "error_handler"
    return "execute_workflow"


def route_after_build(state: SOCState) -> str:
    if state.get("error_messages"):
        return "error_handler"
    return "execute_dax"


def route_after_execution(state: SOCState) -> str:
    if state["execution_status"] == 200:
        return "generate_answer"
    if state["repair_attempts"] < MAX_REPAIR_ATTEMPTS:
        return "repair_dax"
    state["error_messages"] = state.get("error_messages", []) + [
        f"DAX execution failed after {MAX_REPAIR_ATTEMPTS} repair attempt(s)."
    ]
    return "error_handler"


def route_after_repair(state: SOCState) -> str:
    return "execute_dax"


def route_after_workflow_execution(state: SOCState) -> str:
    """
    Three outcomes after execute_soc_workflow_node returns:
      "generate_answer"      — all steps succeeded (status=200)
      "repair_workflow_step" — a step failed AND repair attempts not exhausted
      "error_handler"        — dependency/build error (no step_id set)
                               OR repair attempts exhausted
    """
    if state["execution_status"] == 200:
        return "generate_answer"

    failed_step_id = state.get("workflow_failed_step_id")
    if not failed_step_id:
        # dependency or DAX-build error — not repairable via DAX repair
        return "error_handler"

    current_attempts = state.get("workflow_step_repair_attempts") or 0
    if current_attempts < MAX_REPAIR_ATTEMPTS:
        return "repair_workflow_step"

    return "error_handler"


def route_after_workflow_repair(state: SOCState) -> str:
    """
    After repair_soc_workflow_step_node:
      "execute_workflow" — repaired DAX ready, resume
      "error_handler"    — repair exhausted (node cleared workflow_step_repair_dax)
    """
    if state.get("workflow_step_repair_dax") and state["execution_status"] == 500:
        return "execute_workflow"
    return "error_handler"


# ============================================================
# BUILD LANGGRAPH
# ============================================================
#
# Full graph topology:
#
#   retrieve_measures
#         │
#   extract_intent ──(error)──────────────────────────► error_handler ──► END
#         │                  │
#    (standard/topn/    (workflow)
#     comparison)            │
#         │             plan_workflow ──(error)──────► error_handler
#         │                  │
#     build_dax         execute_workflow ◄──────────────────────────────┐
#         │(error)►eh        │ (step failed,          repair_workflow_step
#         │             attempts left) ──────────────────────────────────┘
#   execute_dax               │ (all done / exhausted)
#         │(200)►ga           ▼
#         │(fail)►   generate_answer ──► END
#     repair_dax
#         │
#   execute_dax (retry)

def build_soc_graph():
    g = StateGraph(SOCState)

    # ── Nodes ─────────────────────────────────────────────────────────────
    g.add_node("retrieve_measures",    retrieve_soc_measures_node)
    g.add_node("extract_intent",       extract_soc_intent_node)

    # standard / topn / comparison path
    g.add_node("build_dax",            build_soc_dax_node)
    g.add_node("execute_dax",          execute_soc_dax_node)
    g.add_node("repair_dax",           repair_soc_dax_node)

    # workflow path
    g.add_node("plan_workflow",        plan_soc_workflow_node)
    g.add_node("execute_workflow",     execute_soc_workflow_node)
    g.add_node("repair_workflow_step", repair_soc_workflow_step_node)

    # shared terminal nodes
    g.add_node("generate_answer",      generate_soc_answer_node)
    g.add_node("error_handler",        soc_error_handler_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    g.set_entry_point("retrieve_measures")
    g.add_edge("retrieve_measures", "extract_intent")

    # Intent → branch
    g.add_conditional_edges(
        "extract_intent", route_after_intent,
        {
            "build_dax":     "build_dax",
            "plan_workflow": "plan_workflow",
            "error_handler": "error_handler",
        },
    )

    # Standard path
    g.add_conditional_edges(
        "build_dax", route_after_build,
        {"execute_dax": "execute_dax", "error_handler": "error_handler"},
    )
    g.add_conditional_edges(
        "execute_dax", route_after_execution,
        {
            "generate_answer": "generate_answer",
            "repair_dax":      "repair_dax",
            "error_handler":   "error_handler",
        },
    )
    g.add_conditional_edges(
        "repair_dax", route_after_repair,
        {"execute_dax": "execute_dax"},
    )

    # Workflow path
    g.add_conditional_edges(
        "plan_workflow", route_after_plan,
        {"execute_workflow": "execute_workflow", "error_handler": "error_handler"},
    )
    g.add_conditional_edges(
        "execute_workflow", route_after_workflow_execution,
        {
            "generate_answer":      "generate_answer",
            "repair_workflow_step": "repair_workflow_step",
            "error_handler":        "error_handler",
        },
    )
    g.add_conditional_edges(
        "repair_workflow_step", route_after_workflow_repair,
        {"execute_workflow": "execute_workflow", "error_handler": "error_handler"},
    )

    # Terminal
    g.add_edge("generate_answer", END)
    g.add_edge("error_handler",   END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def ask(question: str, session_id: str) -> str:
    """
    Ask the SOC agent a question.
    
    Args:
        question: User question
        session_id: Session identifier for tracking conversation context
    
    Returns:
        Business-readable answer string
    """
    print("\n" + "=" * 80)
    print("SOC QUESTION:")
    print(question)
    print("=" * 80)

    app = build_soc_graph()

    initial_state: SOCState = {
        "question":           question,
        "intent_type":        "",
        "intent":             {},
        "retrieved_measures": [],
        "dax_query":          "",
        "final_dax":          "",
        "execution_status":   0,
        "execution_result":   {},
        "answer":             "",
        "repair_attempts":    0,
        "error_messages":     [],
        # Workflow fields
        "workflow_plan":                   [],
        "workflow_results":                {},
        "workflow_dax":                    {},
        "workflow_failed_step_id":         None,
        "workflow_failed_step_index":      None,
        "workflow_step_error":             None,
        "workflow_step_repair_dax":        None,
        "workflow_step_repair_attempts":   0,
    }

    config      = {"configurable": {"thread_id": session_id}}
    final_state = app.invoke(initial_state, config)
    return final_state["answer"]


# ============================================================
# TEST / DEMO
# ============================================================

def test_soc_pipeline():
    import uuid
    
    session_id = str(uuid.uuid4())
    
    questions = [
        # Standard
        "Tell me the manager headcount",
        "Show the total Employees Headcount grouped by Level 1",
        "Show average span by grade",
        "Show managers by function for grade M3",
        # Comparison
        "Compare managers and employees",
        # Top-N
        "Top 10 managers by span",
        "Bottom 5 functions by employee headcount",
        # Standard multi-measure
        "Show manager and employee headcount by business unit",
        # Workflow (complex / dependent)
        "Top 5 managers by span — for each show headcount split by gender",
        "Show the top 3 functions by employee headcount, then for each function "
        "show the average span and manager count broken down by grade",
        "For the top 3 business units by headcount, show span by level and "
        "also manager count by gender",
    ]
    for q in questions:
        print(f"\n{'='*80}\nTEST: {q}\n{'='*80}")
        try:
            answer = ask(q, session_id)
            print(f"ANSWER:\n{answer}")
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    import uuid
    
    # Quick smoke test with the query from the original code
    session_id = str(uuid.uuid4())
    print(f"📋 Session ID: {session_id}\n")
    
    result = ask("Show the total Employees Headcount grouped by Level 1.", session_id)
    print("\nFINAL ANSWER:\n", result)
