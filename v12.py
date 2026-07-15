"""
HR Analytics Pipeline - Generalized LangGraph Implementation (v6)

WHY v6 EXISTS
-------------
v5 broke on complex multi-part questions because its "workflow engine" was not
actually general: Python hardcoded exactly one workflow shape
(top_bu_hp_analysis) built from exactly four fixed step types. Any question
that didn't match that one template produced an empty plan or silently wrong
DAX (e.g. a hardcoded "Emp_Master[Rating]" filter regardless of what the LLM
actually wanted, no support for "annualized" metrics, no real dependency
resolution between steps, defaults never applied to workflow steps).

WHAT CHANGED IN v6
------------------
1. ONE generic "metric spec" shape is used for standard queries, comparison
   queries, and every workflow step. There are no more Python-side step
   "types" to keep adding to - the LLM expresses what it needs (measures,
   groupby, filters, top_n, sort, table) and a single DAX builder renders it.
2. The LLM plans the *entire* workflow (ordered list of metric specs) for
   complex questions. Python's job is to validate, resolve dependencies
   between steps, render DAX, execute, repair on failure, and assemble the
   answer - not to guess which canned template fits.
3. Filters support composable wrapping (e.g. "Annualized Attrition %" wrapped
   with Exit_Type = Vol-Reg, Rating IN {4,5}, Gender = Female) using the same
   CALCULATE-wrap idiom your model already uses internally (see
   "Female Attrition %", "Voluntary Regretted Female Exits", etc. in
   measures.json) - instead of inventing new measure-name string hacks.
4. Entity-list dependencies ("for these top-3 BUs...") are resolved generically
   via a `depends_on` field on any step, not hardcoded to step index 0.
5. The Year/date filter idiom (FILTER(ALL(DimCal[Year]), DimCal[Year]=N)) is
   applied consistently everywhere, because a large fraction of your measures
   (Headcount, Annualized Attrition %, High Performer Annualized Attrition %,
   etc.) are sensitive to MAX(DimCal[Date])/TOTALYTD and need a real filter
   context on DimCal, not an ad hoc per-branch hack.
6. The repair loop now applies per-step inside workflows too (v5 only repaired
   single-query standard/comparison paths; a failing workflow step just
   aborted the whole question).
7. Defaults (year, employment type, exit type) are applied uniformly to every
   step of every intent type, sourced from one BUSINESS_RULES dict.

Architecture Flow:
question -> retrieve_measures -> extract_intent -> router
          -> [standard | comparison | workflow] -> execute (+ repair per step)
          -> generate_answer
"""

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
# Kept as environment-driven, same as v5 expected these names to exist
# somewhere in the runtime environment. Fill these in via env vars rather
# than hardcoding secrets in source.



MEASURES_FILE = os.environ.get("MEASURES_FILE", "measures.json")
DIMENSIONS_FILE = os.environ.get("DIMENSIONS_FILE", "dimensions.json")

MAX_REPAIR_ATTEMPTS = 2
MAX_WORKFLOW_STEPS = 8  # safety cap so a runaway plan can't fire unbounded DAX calls
CURRENT_YEAR = datetime.today().year
TODAY = datetime.today()


# ============================================================
# GRAPH STATE
# ============================================================

class WorkflowStep(TypedDict, total=False):
    step_id: str
    description: str
    table: str                      # "Emp_Master" | "DimCal" | "People Manager Complete Span"
    measures: List[str]             # measure names, evaluated against MEASURES
    measure_filters: Dict[str, Dict]  # per-measure extra CALCULATE-wrap filters {measure: {col: value_or_list}}
    groupby: List[str]
    filters: Dict                   # base filters (Year, Employment_Type, Exit_Type, Rating, Gender, ...)
    top_n: Optional[int]
    sort: Optional[Dict]
    depends_on: Optional[str]       # step_id whose result entities feed this step's entity filter
    depends_on_column: Optional[str]  # which column of the dependency's result holds the entity values
    entity_column: Optional[str]    # column in THIS step to filter using the dependency's entity values


class HRAnalyticsState(TypedDict):
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
    error_messages: Annotated[List[str], operator.add]
    # ── Workflow step-level repair fields ─────────────────────────────────
    # When a workflow step fails these are populated so repair_workflow_step_node
    # has everything it needs: which step, what DAX it tried, what error PBI returned.
    # execute_workflow_node checks workflow_step_repair_dax on entry to know
    # whether it is resuming after a repair (pick up at the failed step with
    # the fixed DAX) vs starting/continuing normally.
    workflow_failed_step_id: Optional[str]      # step_id that failed
    workflow_failed_step_index: Optional[int]   # index in workflow_plan for fast resume
    workflow_step_repair_dax: Optional[str]     # repaired DAX to try on resume
    workflow_step_error: Optional[Dict]         # raw PBI error body for the repair prompt
    workflow_step_repair_attempts: int          # per-step repair counter


# ============================================================
# LOAD METADATA AND LLM
# ============================================================

with open(MEASURES_FILE, "r", encoding="utf-8") as f:
    MEASURES = json.load(f)

with open(DIMENSIONS_FILE, "r", encoding="utf-8") as f:
    DIMENSIONS = json.load(f)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=API_VERSION,
)


# ============================================================
# TABLE / COLUMN RESOLUTION
# ============================================================
# Built once from dimensions.json so every column is resolved to its real
# table automatically, instead of being guessed/hardcoded per branch as v5 did.

TABLE_NAMES = list(DIMENSIONS["tables"].keys())

# column_name -> table_name  (last writer wins only on genuine name clashes;
# Emp_Master / DimCal / People Manager Complete Span columns are namespaced
# distinctly enough in your model that this is safe)
COLUMN_TO_TABLE: Dict[str, str] = {}
for _table, _cols in DIMENSIONS["tables"].items():
    for _col in _cols:
        COLUMN_TO_TABLE[_col] = _table

ALL_MEASURE_NAMES = {m["measure_name"] for m in MEASURES}
ALL_COLUMNS = set(COLUMN_TO_TABLE.keys())

MEASURES_BY_NAME = {m["measure_name"]: m for m in MEASURES}


def qualify_table(table_name: str) -> str:
    """Wrap a table name in single quotes for DAX if it contains spaces."""
    return f"'{table_name}'" if " " in table_name else table_name


def resolve_column(col: str, table: Optional[str] = None) -> str:
    """
    Resolve a bare column name to a fully-qualified 'Table[Column]' DAX
    reference. If `table` is given it is trusted (used for People Manager
    Complete Span queries where column names are ambiguous-looking), otherwise
    the column is looked up against dimensions.json directly so this works
    for ANY column in ANY table without per-branch special-casing.
    """
    if table:
        if col not in DIMENSIONS["tables"].get(table, []):
            raise ValueError(f"Column '{col}' is not part of table '{table}'")
        return f"{qualify_table(table)}[{col}]"

    resolved_table = COLUMN_TO_TABLE.get(col)
    if not resolved_table:
        raise ValueError(f"Unknown column '{col}' - not found in any known table")
    return f"{qualify_table(resolved_table)}[{col}]"


def column_table(col: str, table: Optional[str] = None) -> str:
    """Return just the table name a column belongs to."""
    if table:
        return table
    resolved_table = COLUMN_TO_TABLE.get(col)
    if not resolved_table:
        raise ValueError(f"Unknown column '{col}' - not found in any known table")
    return resolved_table


def get_dimension_context() -> str:
    context = []
    for table, columns in DIMENSIONS["tables"].items():
        context.append(f"\nTABLE: {table}\n" + "\n".join(columns))
    return "\n".join(context)


def validate_measure_exists(measure_name: str) -> bool:
    return measure_name in ALL_MEASURE_NAMES


def get_measure_expression(measure_name: str) -> str:
    """Base (unfiltered) reference to a measure. Override hook kept for any
    measure that needs a literally different expression than '[Name]'."""
    if measure_name in METRIC_OVERRIDES:
        return METRIC_OVERRIDES[measure_name]
    if not validate_measure_exists(measure_name):
        raise ValueError(f"Unknown measure: '{measure_name}'")
    return f"[{measure_name}]"


# ============================================================
# BUSINESS RULES AND OVERRIDES
# ============================================================

BUSINESS_RULES = {
    "default_year": CURRENT_YEAR,
    "default_employment_type": "Full Time",
    "default_exit_type": "Vol-Reg",
}

# Only needed for measures whose correct DAX is NOT simply "[Measure Name]" -
# e.g. a measure that doesn't exist in the model under that exact name but is
# a well-known composite. Kept intentionally tiny: prefer fixing measure
# *selection* (picking the right existing measure) over inventing overrides.
METRIC_OVERRIDES: Dict[str, str] = {}

# Numeric / non-string columns. Anything NOT in this set is treated as a
# string filter value and quoted. This generalizes the old code's manual
# "if key == Overall_Rating: don't quote" special case to any column.
NUMERIC_COLUMNS = {
    "Rating", "Overall_Rating_CS", "Year", "Month", "Age_IN_Years",
    "Age_IN_Month", "Tenure", "Grade_Level",
}

MEASURE_ALIASES = {
    "headcount": "Headcount",
    "fte headcount": "Headcount",
    "fte": "Headcount",
    "total employees": "Headcount",
    "workforce size": "Headcount",
    "attrition count": "Attrition Count",
    "exits": "Attrition Count",
    "leavers": "Attrition Count",
    "attrition rate": "Attrition %",
    "attrition %": "Attrition %",
    "turnover": "Attrition %",
    "annualized attrition": "Annualized Attrition %",
    "annualized attrition %": "Annualized Attrition %",
    "average headcount": "Average Headcount",
    "diversity": "Diversity %",
    "diversity %": "Diversity %",
    "average tenure": "Average Tenure",
    "high performer headcount": "High Performers Overall",
    "high performers": "High Performers Overall",
    "high performer annualized attrition": "High Performer Annualized Attrition %",
    "high performer attrition": "High Performer Attrition %",
    "span":                          "Average Headcount (Span)",
    "direct span":                   "Average Headcount (Span)",
    "largest span":                  "Average Headcount (Span)",
    "manager span":                  "Average Headcount (Span)",
    "complete span":                 "Average Headcount (Span)",
    "span headcount":                "Average Headcount (Span)",
    "span of control":               "Average Headcount (Span)",
    "average headcount span":        "Average Headcount (Span)",
    "span attrition":                "Overall Attrition % (Span)",
    "span annualized attrition":     "Annualized Attrition % (Span)",
    "span voluntary attrition":      "Voluntary Annualized Attrition (Span)",
    "span high performer":           "Average High Performer Headcount (Span)",
    "span female headcount":         "Average People Manger Female Headcount (Span)",
    "span exits":                    "Total Exits (Span)",
}


# ============================================================
# HRBP_EXIT_REASON MAPPING
# ============================================================
# Built from pv360_mapping.xlsx. Maps user query phrases (both formal
# synonyms and colloquial phrases) → HRBP_Exit_Reason DAX filter value.
# Called by resolve_hrbp_exit_reason() which is invoked inside
# apply_defaults() and the master/workflow prompts.

HRBP_EXIT_REASON_MAP = {
    # key (HRBP_Exit_Reason value) : list of trigger phrases (lowercase)
    "Absconding": [
        "abandonment", "unauthorized absence", "job desertion", "awol",
    ],
    "Alpha": [
        "top performer", "high potential", "star employee",
    ],
    "Behaviour Issues": [
        "conduct issues", "attitude problems", "interpersonal conflicts",
        "professional misconduct",
    ],
    "BGV - Red": [
        "background verification failure", "failed background check", "credential fraud",
    ],
    "Career/Growth": [
        "career advancement", "growth opportunity", "vertical mobility",
        "professional development", "career and growth",
    ],
    "Change of Career Track": [
        "career pivot", "domain switch", "industry change", "career transition",
    ],
    "Compensation": [
        "salary dissatisfaction", "pay gap", "remuneration issues",
        "below market pay", "comp",
    ],
    "Contract Conversion - Regular": [
        "permanent absorption", "full-time conversion", "regularization",
    ],
    "Contract Conversion - TPC": [
        "third party contract", "vendor payroll transfer", "outsourced employment",
    ],
    "Culture Fitment": [
        "cultural mismatch", "values misalignment", "poor cultural fit",
    ],
    "Death": [
        "deceased", "demise", "mortality", "passed away",
    ],
    "Early Attrition": [
        "premature exit", "quick quit", "early resignation", "honeymoon attrition",
    ],
    "Early Retirement": [
        "voluntary early exit", "pre-mature retirement", "vrs",
        "voluntary retirement scheme",
    ],
    "End of Contract": [
        "contract expiry", "project completion exit", "term end",
    ],
    "End of Internship": [
        "internship completion", "trainee exit", "intern offboarding",
    ],
    "Health Reasons": [
        "medical exit", "health-related separation", "illness-based resignation",
    ],
    "Higher Studies": [
        "further education", "academic pursuit", "post-graduate studies",
        "educational sabbatical",
    ],
    "Issues with Org Direction/Management (Skip)": [
        "leadership misalignment", "strategic disagreement", "management conflict",
    ],
    "Lack of Role Clarity": [
        "ambiguous job role", "undefined responsibilities", "job ambiguity",
    ],
    "Manager Issues": [
        "supervisor conflict", "poor leadership", "toxic manager", "bad boss",
    ],
    "Misconduct": [
        "policy violation", "ethical breach", "disciplinary issue",
        "code of conduct violation",
    ],
    "Non Performance": [
        "underperformance", "productivity failure", "performance default",
    ],
    "Other": [
        "miscellaneous", "unclassified reason", "uncategorized exit",
    ],
    "Performance Reasons": [
        "performance-based exit", "pip failure", "productivity issues",
    ],
    "Personal": [
        "personal circumstances", "individual reasons", "private matter",
    ],
    "Personal - Family Reasons": [
        "family obligation", "domestic responsibility", "caregiving duty",
    ],
    "Relocation": [
        "geographic move", "city change", "place of residence change",
    ],
    "Retirement": [
        "superannuation", "end of service", "age-based exit",
    ],
    "RIF": [
        "reduction in force", "layoff", "workforce reduction", "downsizing",
    ],
    "Role Redundancy": [
        "position eliminated", "job obsolescence", "organizational restructuring",
    ],
    "Termination - Other": [
        "involuntary exit", "employment termination", "dismissal",
    ],
    "Transferred Within Group": [
        "internal mobility", "intra-group transfer", "lateral move",
    ],
    "Vendor Rationalization": [
        "vendor consolidation", "outsourcing reduction", "third-party exit",
    ],
    "Work Life Balance": [
        "burnout", "overwork", "personal time conflict", "lifestyle mismatch",
    ],
}

# Reverse lookup: phrase (lowercase) → HRBP_Exit_Reason value
# Built once at startup for O(1) lookup during query processing.
_HRBP_PHRASE_TO_REASON: Dict[str, str] = {}
for _reason, _phrases in HRBP_EXIT_REASON_MAP.items():
    for _phrase in _phrases:
        _HRBP_PHRASE_TO_REASON[_phrase.lower()] = _reason


def resolve_hrbp_exit_reason(question: str) -> Optional[str]:
    """
    Scan the user question for any known HRBP_Exit_Reason trigger phrase
    and return the matching HRBP_Exit_Reason value (as stored in the column),
    or None if no match is found.

    Matching is case-insensitive and uses substring search so multi-word
    phrases (e.g. "top gun") are found inside longer sentences.

    When multiple phrases match, the longest match wins (most specific first)
    to avoid "compensation" matching inside "deferred compensation plan".
    """
    q = question.lower()
    best_phrase, best_reason = "", None

    for phrase, reason in _HRBP_PHRASE_TO_REASON.items():
        if phrase in q and len(phrase) > len(best_phrase):
            best_phrase = phrase
            best_reason = reason

    if best_reason:
        print(f"[HRBP_EXIT_REASON] Matched phrase '{best_phrase}' → '{best_reason}'")

    return best_reason



def apply_defaults(question: str, filters: Dict, allow_exit_type_default: bool = True,
                   table: Optional[str] = None) -> Dict:
    """
    Apply business-rule defaults to a filter dict, without clobbering
    anything the LLM/user already specified explicitly.

    `table` — when provided, column-level defaults (Employment_Type,
    Exit_Type) are only injected if that column actually exists in the
    given table. This prevents, e.g., injecting Employment_Type for a
    "People Manager Complete Span" query where the column is named
    Employment_Type_CS instead.

    `allow_exit_type_default` is False for workflow steps. Reason: a
    workflow step's "filters" apply to EVERY measure in that step
    (Headcount AND Attrition % might both live in the same step). Defaulting
    Exit_Type onto the whole step would silently restrict Headcount to only
    voluntarily-exited employees too, which is wrong. For workflow steps,
    Vol-Reg scoping must instead be expressed per-measure via
    measure_filters (the planner prompt instructs the LLM to do this).
    """
    q = question.lower()
    filters = dict(filters)
    table_cols = set(DIMENSIONS["tables"].get(table, [])) if table else None

    def col_ok(col_name: str) -> bool:
        """True if the column is valid for the target table (or no table specified)."""
        if table_cols is None:
            return True
        return col_name in table_cols

    if "Year" not in filters:
        year_found = re.search(r"\b(20\d{2})\b", q)
        if year_found:
            filters["Year"] = int(year_found.group(1))
        else:
            filters["Year"] = BUSINESS_RULES["default_year"]
    # ── HRBP_Exit_Reason ─────────────────────────────────────────────────
    # Auto-detect from question text if not already explicitly set.
    if "HRBP_Exit_Reason" not in filters and col_ok("HRBP_Exit_Reason"):
        matched_reason = resolve_hrbp_exit_reason(question)
        if matched_reason:
            filters["HRBP_Exit_Reason"] = matched_reason

    employment_terms = ["intern", "contract", "temporary", "part time", "part-time"]
    if "Employment_Type" not in filters and col_ok("Employment_Type"):
        if "full time" in q or "full-time" in q or "fte" in q:
            filters["Employment_Type"] = "Full Time"
        elif not any(x in q for x in employment_terms):
            filters["Employment_Type"] = BUSINESS_RULES["default_employment_type"]

    if allow_exit_type_default and "Exit_Type" not in filters and col_ok("Exit_Type"):
        if "attrition" in q and "vol-reg" not in q and "vol reg" not in q and "exit type" not in q:
            filters["Exit_Type"] = BUSINESS_RULES["default_exit_type"]
        elif "vol-reg" in q or "vol reg" in q or "voluntary" in q:
            filters["Exit_Type"] = "Vol-Reg"

    if ("ytd" in q or "year to date" in q) and "Start_Date" not in filters:
        filters["Start_Date"] = f"{CURRENT_YEAR}-01-01"
        filters["End_Date"] = TODAY.strftime("%Y-%m-%d")

    return filters


def retrieve_measures(question: str, top_k: int = 25) -> List[Dict]:
    q = question.lower()
    scored = []

    # Detect if this is a span-of-control / People Manager Complete Span question
    span_question = any(kw in q for kw in [
        "span", "direct span", "complete span", "manager span",
        "largest span", "span of control",
    ])

    for measure in MEASURES:
        text = (measure.get("measure_name", "") + " " + measure.get("description", "")).lower()
        score = 0

        for alias, target in MEASURE_ALIASES.items():
            if alias in q and target == measure["measure_name"]:
                score += 10

        for token in re.findall(r"[a-z0-9%]+", q):
            if len(token) > 2 and token in text:
                score += 1

        # ── Span question boosting ────────────────────────────────────────
        # When the question is about manager span, boost (Span) measures and
        # penalise (Direct Reports) measures so the LLM gets the right family.
        if span_question:
            name = measure.get("measure_name", "")
            if "(Span)" in name:
                score += 8      # strong boost — correct table for span questions
            elif "(Direct Reports)" in name:
                score -= 5      # penalty — wrong table; different data entirely

        scored.append((score, measure))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

# ============================================================
# GENERALIZED DAX FILTER BUILDING
# ============================================================
# Single source of truth for turning a {column: value} dict into DAX FILTER
# clauses. Replaces v5's repeated, inconsistent per-branch "if key == X"
# blocks (build_filters, and three near-duplicates inside generate_step_dax)
# with one function that works for ANY column in ANY table, including the
# tables having no special-cased column at all.

def _format_filter_value(value, numeric: bool) -> str:
    # Boolean values (yes/true → 1, no/false → 0) must be checked FIRST
    # because in Python bool is a subclass of int, so True == 1 and
    # isinstance(True, int) is True — without this guard they'd pass the
    # int check and render as "1"/"0" only by accident.  Explicit is safer.
    if isinstance(value, bool):
        return "1" if value else "0"
    # String shortcuts so filters can pass "yes"/"no"/"true"/"false" as values
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
    Build a single FILTER(ALL(...), ...) clause for one column.
    Supports:
      - scalar equality:      Year = 2026
      - list -> IN {...}:     Rating IN {4, 5}
    Numeric vs string formatting is column-driven (NUMERIC_COLUMNS), not
    hand-special-cased per filter key.
    """
    col_ref = resolve_column(column, table)
    numeric = column in NUMERIC_COLUMNS or column == "Year"

    if isinstance(value, (list, tuple, set)):
        formatted = ", ".join(_format_filter_value(v, numeric) for v in value)
        return f"FILTER(ALL({col_ref}), {col_ref} IN {{{formatted}}})"

    formatted = _format_filter_value(value, numeric)
    return f"FILTER(ALL({col_ref}), {col_ref} = {formatted})"


def build_date_range_clause(start: str, end: str) -> str:
    sy, sm, sd = start.split("-")
    ey, em, ed = end.split("-")
    return (
        f"FILTER(\n    ALL(DimCal[Date]),\n"
        f"    DimCal[Date] >= DATE({int(sy)},{int(sm)},{int(sd)})\n"
        f"    &&\n"
        f"    DimCal[Date] <= DATE({int(ey)},{int(em)},{int(ed)})\n)"
    )


def build_filters(filters: Dict, table: Optional[str] = None) -> List[str]:
    """Turn a base filter dict into a list of DAX FILTER(...) clause strings.
    Date range (Start_Date/End_Date) is handled as one combined clause;
    every other key is resolved generically via build_filter_clause."""
    clauses = []
    skip = {"Start_Date", "End_Date"}

    for key, value in filters.items():
        if key in skip:
            continue
        if key == "Year":
            clauses.append(build_filter_clause("Year", value, table="DimCal"))
            continue
        clauses.append(build_filter_clause(key, value, table=table))

    if "Start_Date" in filters and "End_Date" in filters:
        clauses.append(build_date_range_clause(filters["Start_Date"], filters["End_Date"]))

    return clauses


def build_entity_in_clause(column: str, values: List, table: Optional[str] = None) -> str:
    """Build the 'restrict to these specific entities' clause used to thread
    a prior step's result (e.g. top-3 BU names) into a dependent step."""
    return build_filter_clause(column, list(values), table=table)


def build_sort(sort: Optional[Dict], fallback_alias: Optional[str] = None) -> str:
    if not sort:
        if fallback_alias:
            return f'[{fallback_alias}], DESC'
        return ""
    by = sort.get("by", "") or fallback_alias or ""
    if not by:
        return ""
    order = sort.get("order", "desc").upper()
    if by.startswith("[") and by.endswith("]"):
        return f"{by}, {order}"
    return f"[{by}], {order}"


# ============================================================
# GENERALIZED MEASURE-WRAP BUILDING
# ============================================================
# Your model's own measures (Female Attrition %, Voluntary Regretted Female
# Exits, Female Headcount, etc.) all follow the same idiom: take a base
# measure and wrap it in CALCULATE(...) adding one extra column predicate.
# This function generalizes that idiom so the agent can compose ANY measure
# with ANY combination of extra filters (Vol-Reg, Rating IN {4,5}, Gender,
# Business_Unit_Name, ...) instead of needing a pre-built measure for every
# combination (which mostly don't exist - e.g. there is no
# "High Performer Attrition % Female" measure in your model).

def build_filtered_measure_expression(
    measure_name: str,
    extra_filters: Optional[Dict] = None,
    table: Optional[str] = None,
) -> str:
    """
    Returns a DAX expression for `measure_name` wrapped with extra column
    predicates via CALCULATE, e.g.:

        build_filtered_measure_expression(
            "High Performer Annualized Attrition %",
            {"Gender": "Female"}
        )
        -> CALCULATE([High Performer Annualized Attrition %], Emp_Master[Gender] = "Female")

    extra_filters values can be scalars or lists (-> IN {...}).
    NOTE: this builds a row-context predicate (col = value / col IN {...}),
    NOT a FILTER(ALL(...),...) clause, because inside CALCULATE we generally
    want to narrow the existing context (e.g. narrow Attrition Count to
    Vol-Reg) rather than override an ALL(). This matches how your model's
    own measures (e.g. "Female Attrition %") are written.
    """
    base = get_measure_expression(measure_name)
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


# ============================================================
# UNIFIED METRIC-SPEC DAX RENDERER
# ============================================================
# This single function renders DAX for a "metric spec" - the one generic
# shape used by standard queries, comparison queries, and every workflow
# step. It replaces v5's THREE separate, partially-duplicated DAX generators
# (generate_standard_dax, generate_comparison_dax, generate_step_dax with its
# 4 internal branches) with one general implementation, so adding a new kind
# of question never again requires adding a new Python template.
#
# A metric spec looks like:
# {
#   "table": "Emp_Master",
#   "measures": ["Headcount", "Annualized Attrition %"],
#   "measure_filters": {                 # OPTIONAL per-measure extra wraps
#       "Annualized Attrition %": {"Exit_Type": "Vol-Reg"}
#   },
#   "measure_labels": {                  # OPTIONAL friendly output column names
#       "Annualized Attrition %": "Vol-Reg Annualized Attrition %"
#   },
#   "groupby": ["Business_Unit_Name"],
#   "filters": {"Year": 2026, "Employment_Type": "Full Time"},
#   "top_n": 3,
#   "sort": {"by": "Headcount", "order": "desc"}
# }

def render_metric_spec_dax(spec: Dict, entity_filter_clause: Optional[str] = None) -> str:
    table = spec.get("table") or "Emp_Master"
    measures = spec.get("measures", [])
    result_filters = spec.get("result_filters", []) or []

    if not measures:
        raise ValueError("metric spec requires at least one measure")

    measure_filters = spec.get("measure_filters", {}) or {}
    measure_labels = spec.get("measure_labels", {}) or {}
    groupby = spec.get("groupby", []) or []
    filters = spec.get("filters", {}) or {}
    top_n = spec.get("top_n")
    sort = spec.get("sort")

    dax_filters = build_filters(filters, table=table if table != "Emp_Master" else None)
    if entity_filter_clause:
        dax_filters.append(entity_filter_clause)

    dax_groupby = [resolve_column(c, table=table if table != "Emp_Master" else None) for c in groupby]

    # ---- Single scalar value: no groupby, no top_n ----
    if not groupby and not top_n:
        rows = []
        for m in measures:
            label = measure_labels.get(m, m)
            expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
            calc_body = expr if not dax_filters else f"CALCULATE(\n        {expr},\n        {',  '.join(dax_filters)}\n    )"
            rows.append(f'    "{label}",\n    {calc_body}')
        dax = "EVALUATE\nROW(\n" + ",\n".join(rows) + "\n)"
        return dax.strip()

    # ---- Grouped (and optionally TOPN) ----
    summarize_parts: List[str] = []
    summarize_parts.extend(dax_groupby)
    summarize_parts.extend(dax_filters)
    for m in measures:
        label = measure_labels.get(m, m)
        expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
        summarize_parts.append(f'"{label}"')
        summarize_parts.append(expr)

    summarize = "SUMMARIZECOLUMNS(\n    " + ",\n    ".join(summarize_parts) + "\n)"

    # Apply result filters (DAX equivalent of filtering aggregated rows)
    table_expression = summarize

    if result_filters:

        conditions = []

        for rf in result_filters:

            measure = rf["measure"]
            op = rf["operator"]
            value = rf["value"]

            if isinstance(value, str):
                value = f'"{value}"'

            conditions.append(
                f'[{measure}] {op} {value}'
            )

        condition_text = " && ".join(conditions)

        table_expression = f"""
    FILTER(
        {summarize},
        {condition_text}
    )
    """.strip()

    # if top_n:
    #     primary_measure = measure_labels.get(measures[0], measures[0])
    #     sort_expr = build_sort(sort, fallback_alias=primary_measure)
    #     dax = f"EVALUATE\nTOPN(\n    {top_n},\n    {summarize},\n    {sort_expr}\n)"
    #     return dax.strip()
    if top_n:
        primary_measure = measure_labels.get(measures[0], measures[0])
        sort_expr = build_sort(sort, fallback_alias=primary_measure)

        dax = f"""
        EVALUATE
        TOPN(
            {top_n},
            {table_expression},
            {sort_expr}
        )
        """
        return dax.strip()

    # return f"EVALUATE\n{summarize}".strip()
    return f"EVALUATE\n{table_expression}".strip()


def render_comparison_dax(spec: Dict) -> str:
    """
    Year-over-year / trend comparisons. Generalized over v5's version:
    works for any table/entity/measure combo via resolve_column +
    build_filtered_measure_expression instead of branching on
    "is this a manager-span query".
    """
    table = spec.get("table") or "Emp_Master"
    entity = spec.get("entity")
    measure = spec.get("measure")
    comparison = (spec.get("comparison") or "").lower()
    base_filters = spec.get("filters", {}) or {}

    if not entity or not measure:
        raise ValueError("Comparison queries require 'entity' and 'measure'")

    entity_col = resolve_column(entity, table=table if table != "Emp_Master" else None)
    dax_filters_no_year = build_filters(
        {k: v for k, v in base_filters.items() if k != "Year"},
        table=table if table != "Emp_Master" else None,
    )
    expr = get_measure_expression(measure)

    current_year = base_filters.get("Year", CURRENT_YEAR)
    previous_year = int(current_year[0]) - 1

    current_years = base_filters.get("Year", [CURRENT_YEAR])

    # Ensure it's always a list
    if not isinstance(current_years, list):
        current_years = [current_years]

    # Convert all years to integers and sort in descending order
    current_years = sorted(map(int, current_years), reverse=True)

    if len(current_years) >= 2:
        # Take the latest two years
        current_year = current_years[0]
        previous_year = current_years[1]
    else:
        # Only one year is present
        current_year = current_years[0]
        previous_year = current_year - 1

    print(f"Current Year: {current_year}")
    print(f"Previous Year: {previous_year}")

    extra = ("," + ",\n    ".join(dax_filters_no_year)) if dax_filters_no_year else ""

    if "yoy" in comparison or "year over year" in comparison:
        dax = f"""EVALUATE
                    FILTER(
                        ADDCOLUMNS(
                            SUMMARIZECOLUMNS(
                                {entity_col},
                                FILTER(ALL(DimCal[Year]), DimCal[Year] = {current_year}){extra},
                                "Current Value", {expr},
                                "Previous Value", CALCULATE(
                                    {expr},
                                    FILTER(ALL(DimCal[Year]), DimCal[Year] = {previous_year})
                                )
                            ),
                            "Change", [Current Value] - [Previous Value],
                            "% Change", DIVIDE([Current Value] - [Previous Value], [Previous Value], 0) * 100
                        ),
                        [Change] > 0
                    )"""
    else:
        time_dimension = spec.get("time_dimension", "Year")
        time_col = resolve_column(time_dimension, table="DimCal")
        dax = f"""EVALUATE
SUMMARIZECOLUMNS(
    {entity_col},
    {time_col}{extra},
    "{measure}", {expr}
)"""

    return dax.strip()


# ============================================================
# MASTER INTENT PROMPT  (classify + extract in ONE LLM call)
# ============================================================
# Combines the old v5 prompt's extraction richness (measure selection,
# group_by, filters, top_n, sort, generation mapping, worked examples) with
# the new classification logic (standard / comparison / workflow routing).
#
# For standard and comparison questions this is the ONLY prompt called —
# it returns the full spec ready for DAX rendering.
# For workflow questions it returns intent_type="workflow" and no spec;
# WORKFLOW_PLAN_PROMPT is then called separately to produce the step list.
# This keeps the master prompt's output small and reliable for the common
# case while still routing complex multi-step questions to the dedicated
# planner.

MASTER_INTENT_PROMPT = ChatPromptTemplate.from_template(
"""

You are an HR Analytics expert. Your PRIMARY task is to correctly classify the user's question into one of three intent types.

IMPORTANT:
Intent classification MUST be completed BEFORE considering schemas, measures, business rules, DAX limitations, filters, or extraction.

Follow this order:

    Step 1.
    Determine the intent.

    Step 2.
    If the intent is "workflow",
    STOP immediately and return only:

    {{
        "intent_type":"workflow",
        "reasoning":"..."
    }}

    Do NOT perform extraction.

    Step 3.
    Only if the intent is "standard" or "comparison",
    continue reading the remaining instructions and extract the query specification.

Intent classification has higher priority than every other instruction below.
      

═══════════════════════════════════════════════
USER QUESTION:
{question}
═══════════════════════════════════════════════

TODAY'S YEAR: {current_year}

RETRIEVED MEASURES (use EXACT names, never invent one):
{measures}

AVAILABLE TABLES AND COLUMNS (use EXACT names, never invent one):
{dimensions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Choose EXACTLY one intent.

STANDARD
---------
The request can be answered using ONE query.

Characteristics:
- One metric or multiple independent metrics.
- Any number of group-by columns.
- Filters are allowed.
- Top-N is allowed.
- Sorting is allowed.
- No later step depends on an earlier result.

Examples:
- Headcount by BU
- Attrition by Gender
- Headcount by BU and Gender
- Top 10 Departments by Headcount
- Attrition trend by Month

COMPARISON
----------
The request compares the SAME metric across time.

Characteristics:
- Year-over-Year
- Month-over-Month
- Trend
- Increase / Decrease
- Previous Year
- Previous Month

Examples:
- Attrition compared to last year
- Headcount trend
- Managers whose span increased YoY

WORKFLOW
--------
The request requires MULTIPLE dependent queries.

WORKFLOW TEST
    Ask yourself this question:
    Can the user's request be answered with a single SQL or DAX query?
        YES → STANDARD
        NO 

        Does the second part require the OUTPUT of the first query?

        YES → WORKFLOW
        Otherwise → STANDARD


Examples:
    ✓ Top 5 BUs, then show Gender split.
    ✓ Find departments with highest attrition, then list employees.
    ✓ Identify top managers, then analyze their teams.
    ✓ Top 3 locations, then attrition reasons for each.

Decision Rule:
    If one query is sufficient → STANDARD.
    If one query depends on another → WORKFLOW.
    Never classify a query as WORKFLOW simply because it contains multiple dimensions or multiple measures.

Examples that are NOT workflow:
    Question: Headcount by BU and Gender
    Intent: standard
    Reason: Both dimensions can be answered in one query.

    Question: Headcount and Attrition by BU
    Intent: standard
    Reason: Multiple measures do not require multiple queries.

    Question: Top 10 Departments by Attrition
    Intent: standard
    Reason: Top-N is still one query.

    Question: Attrition Trend by Month
    Intent: standard
    Reason: Trend is still one query.

    Question: Top 5 BUs and then show monthly attrition for those BUs
    Intent: workflow
    Reason: Second query depends on the Top 5 from the first query.

CLASSIFICATION LOCK
    Once the intent has been determined, do NOT change it after reading any schema, business rules, measures, examples, or DAX instructions.
    The remaining instructions are ONLY for extraction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — EXTRACT SPEC (standard / comparison only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following rules apply ONLY AFTER intent classification.
Do NOT use these rules when deciding the intent type.

ALWAYS apply these filters unless explicitly told otherwise:
	- FTE only: Employment_Category = "FTE"
	- Vol-Reg attrition only: Exit_Type = "Vol-Reg" AND Attrition = 1
	- IS_ACTIVE is integer (1/0), not boolean
	- Annualized attrition rate = (exits * 12) ÷ (avg FTE headcount * months elapsed)
	- Avg FTE headcount = SUM(IS_ACTIVE where FTE) ÷ DISTINCTCOUNT(Snap_Date)
	- YTD exit counts require cross-snapshot aggregation using MONTH(Snap_Date) = MonthCLWD pattern
	- Use CALCULATETABLE(ADDCOLUMNS(VALUES(...))) instead of SUMMARIZECOLUMNS with IN filters
	- Grade E8 does not exist
	- Always combine Exit_Type filter with Attrition = 1 to avoid inflated counts
    
SCHEMA_CONTEXT = 
	Table: Emp_Master (snapshot-based, one row per employee per Snap_Date)
		Key columns:
		- Snap_Date (date): Snapshot date
		- Employee_ID (string): Unique employee identifier
		- Employee_Name (string): Employee name
		- BU (string): Business Unit
		- Department (string): Department
		- Grade (string): Grade level (E1-E7, M1-M5, etc.)
		- Gender (string): Male/Female
		- Generation (string): Gen Z, Millennial, Gen X, Baby Boomer
		- Tenure_Bucket (string): Tenure grouping
		- Employment_Category (string): FTE, TPC, etc.
		- IS_ACTIVE (integer): 1 = active, 0 = inactive
		- Attrition (integer): 1 = exit in this snapshot, 0 = no
		- Exit_Type (string): Vol-Reg, Vol-Non-Reg, Invol, etc.
		- HRBP_Exit_Reason (string): Reason for exit
		- Rating (string): Performance rating
		- IS_ON_Resignation (integer): 1 = on notice
		- ConfirmedLastWorkingYear (integer): Year of last working date
		- MonthCLWD (integer): Month of confirmed last working date
		- Early_Inactive (integer): Flag for early inactivity
		- Designation (string): Job title
		- Location (string): Office location
		- Hiring_Source (string): Source of hire
		- Date_of_Joining (date): Join date
		- Band (string): Compensation band

	Built-in measures may return null against historical snap dates — always use base-column DAX.
	SUMMARIZECOLUMNS ignores Grade IN {{}} filters — use CALCULATETABLE(ADDCOLUMNS(VALUES(...))).

CRITICAL MODELING RULES (apply to every query):
    MEASURES
        • Use ONLY measure names that appear in RETRIEVED MEASURES. Never invent one.
        • "FTE" / "Full-Time" headcount → use measure "Headcount" with filter
        Employment_Type = "Full Time". There is no "FTE Headcount" measure.
        • "Annualized Attrition %" and "High Performer Annualized Attrition %" are
        real existing measures — select them directly. Never multiply by 12 yourself.
        • "Vol-Reg" / "voluntary attrition" → Exit_Type column = "Vol-Reg".
        No separate "Vol-Reg Annualized Attrition %" measure exists for Emp_Master.
        Express it by wrapping the base measure in measure_filters:
            {{"Annualized Attrition %": {{"Exit_Type": "Vol-Reg"}}}}
        Same idiom the model uses for Female Attrition % (wraps base + Gender filter).
        • NEVER put Exit_Type in step-wide "filters" when the step also returns a
        headcount-style measure — that would wrongly restrict headcount to only
        exited employees. Use measure_filters for Exit_Type scoping only.
        • For manager span-of-control questions use table "People Manager Complete Span"
        and its "_CS"-suffixed columns. Otherwise use "Emp_Master".

    COLUMNS
        • Use ONLY column names from AVAILABLE TABLES AND COLUMNS. Never invent one.
        • "High Performers" → Rating IN {{4, 5}} on Emp_Master.
        Column is "Rating", NOT "Overall_Rating" (that name only exists as
        "Overall_Rating_CS" on the People Manager Complete Span table).
        • Gender / dimension splits → put the column in "groupby", not in filters,
        and do not search for a gender-specific measure name.
        • Trend by year → groupby: ["Year"] from DimCal.
        • Trend by month → groupby: ["Month"] or ["MonthName"] from DimCal.

    GENERATION FILTER VALUES (use these exact strings):
    "Baby Boomers" | "Generation X" | "Generation Z or iGEN" | "Millennials or Generation Y"

    FILTERS
    • filters values are plain values (not DAX). Year is an integer, e.g. 2026.
    • Do not add Exit_Type to filters when a step has mixed attrition + headcount
    measures — use measure_filters instead (see MEASURES rules above).

    • "HRBP_Exit_Reason" values must come EXACTLY from this list:
    Absconding, Alpha, Behaviour Issues, BGV - Red, Career/Growth,
    Change of Career Track, Compensation, Contract Conversion - Regular,
    Contract Conversion - TPC, Culture Fitment, Death, Early Attrition,
    Early Retirement, End of Contract, End of Internship, Health Reasons,
    Higher Studies, Issues with Org Direction/Management (Skip),
    Lack of Role Clarity, Manager Issues, Misconduct, Non Performance,
    Other, Performance Reasons, Personal, Personal - Family Reasons,
    Relocation, Retirement, RIF, Role Redundancy, Termination - Other,
    Transferred Within Group, Vendor Rationalization, Work Life Balance.
    Never invent a value. The system will auto-detect from the question
    text so you generally do NOT need to set HRBP_Exit_Reason in filters
    yourself — only set it if the user names an exact reason explicitly.

    RESULT FILTERS

        If the user filters on an aggregated measure
        such as:

        Headcount > 100
        Repotee Count > 15
        Attrition % > 5
        Average Headcount < 50

        DO NOT place these inside "filters".

        Instead return

        "result_filters":[
            {{
                "measure":"Repotee Count",
                "operator":">",
                "value":15
            }}
        ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Always include "intent_type" and "reasoning" at the top level.

For "workflow" — stop here (planner handles the rest):
{{
    "intent_type": "workflow",
    "reasoning": "one sentence"
}}

For "standard":
{{
    "intent_type": "standard",
    "reasoning": "one sentence",
    "table": "Emp_Master",
    "measures": ["Measure Name"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["ColumnName"],
    "filters": {{"Year": {current_year}}},
    "result_filters": [],
    "top_n": null,
    "sort": null
}}

For "comparison":
{{
    "intent_type": "comparison",
    "reasoning": "one sentence",
    "entity": "Column name to group/compare",
    "table": "Emp_Master",
    "measure": "Measure Name",
    "comparison": "YoY",
    "time_dimension": "Year",
    "filters": {{}}
    "result_filters": [],
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Provide attrition count by business unit
A:
{{
    "intent_type": "standard",
    "reasoning": "Single metric grouped by one dimension, no dependencies.",
    "table": "Emp_Master",
    "measures": ["Attrition Count"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["Business_Unit_Name"],
    "filters": {{}},
    "top_n": null,
    "sort": null
}}

Q: Headcount trend for Gen Z
A:
{{
    "intent_type": "standard",
    "reasoning": "Headcount over time filtered to one generation.",
    "table": "Emp_Master",
    "measures": ["Headcount"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["Year"],
    "filters": {{"Generation": "Generation Z or iGEN"}},
    "top_n": null,
    "sort": {{"by": "Year", "order": "asc"}}
}}

Q: Top 5 managers with largest complete span
A:
{{
    "intent_type": "standard",
    "reasoning": "Top-N ranking of managers by span headcount.",
    "table": "People Manager Complete Span",
    "measures": ["Average Headcount (Span)"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["Manager_Name_CS"],
    "filters": {{}},
    "top_n": 5,
    "sort": {{"by": "Average Headcount (Span)", "order": "desc"}}
}}

Q: Managers whose complete span increased year over year
A:
{{
    "intent_type": "comparison",
    "reasoning": "Year-over-year change for one entity type (managers).",
    "entity": "Manager_Name_CS",
    "table": "People Manager Complete Span",
    "measure": "Average Headcount (Span)",
    "comparison": "YoY",
    "time_dimension": "Year",
    "filters": {{}}
}}

Q: Show attrition by gender and BU for Millennials
A:
{{
    "intent_type": "standard",
    "reasoning": "Single metric grouped by two dimensions with a generation filter.",
    "table": "Emp_Master",
    "measures": ["Attrition Count"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["Business_Unit_Name", "Gender"],
    "filters": {{"Generation": "Millennials or Generation Y"}},
    "top_n": null,
    "sort": null
}}

Q: What is the voluntary attrition % by BU?
A:
{{
    "intent_type": "standard",
    "reasoning": "Attrition % filtered to Vol-Reg exits, grouped by BU.",
    "table": "Emp_Master",
    "measures": ["Attrition %"],
    "measure_filters": {{"Attrition %": {{"Exit_Type": "Vol-Reg"}}}},
    "measure_labels": {{"Attrition %": "Vol-Reg Attrition %"}},
    "groupby": ["Business_Unit_Name"],
    "filters": {{}},
    "top_n": null,
    "sort": null
}}

Q: Top 3 BUs by FTE headcount with HP breakdown by gender for 2026
A:
{{
    "intent_type": "workflow",
    "reasoning": "Requires selecting top-3 BUs first, then dependent sub-breakdowns per BU."
}}

Q: List the top 5 managers with the largest direct span
A:
{{
    "intent_type": "standard",
    "reasoning": "Top-N ranking of managers by their direct span headcount.",
    "table": "People Manager Complete Span",
    "measures": ["Average Headcount (Span)"],
    "measure_filters": {{}},
    "measure_labels": {{}},
    "groupby": ["Manager_Name_CS"],
    "filters": {{}},
    "top_n": 5,
    "sort": {{"by": "Average Headcount (Span)", "order": "desc"}}
}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your response MUST be valid JSON.
Do not explain your reasoning outside the JSON.
Do not wrap JSON inside markdown.
Do not include any introductory text.
Return exactly one JSON object.
"""
)


def classify_and_extract_intent(question: str, measures: List[Dict], dimensions: str) -> Dict:
    """Single LLM call: classifies AND extracts full spec for standard/comparison.
    For workflow, returns only intent_type + reasoning (planner handles the rest)."""
    response = llm.invoke(
        MASTER_INTENT_PROMPT.format(
            question=question,
            current_year=CURRENT_YEAR,
            measures=json.dumps([m["measure_name"] for m in measures], indent=2),
            dimensions=dimensions,
        )
    )
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


# Backward-compat aliases so existing node code needs minimal changes
def classify_intent(question: str, measures: List[Dict], dimensions: str) -> Dict:
    return classify_and_extract_intent(question, measures, dimensions)


def extract_metric_spec(question: str, intent_type: str, measures: List[Dict], dimensions: str) -> Dict:
    """For standard/comparison: re-use the already-parsed result stored in state.
    This is called from standard_node / comparison_node which receive the full
    parsed dict from extract_intent_node — so this is now a no-op passthrough.
    Kept for interface compatibility; callers pass the cached intent dict."""
    # The actual extraction already happened in classify_and_extract_intent.
    # Nodes read from state["intent"] directly. This stub exists so any
    # external callers of v6.extract_metric_spec() still work unchanged.
    raise NotImplementedError(
        "extract_metric_spec() is no longer called internally — "
        "intent extraction now happens in extract_intent_node() via "
        "classify_and_extract_intent(). Read state['intent'] instead."
    )


# ============================================================
# WORKFLOW PLANNING PROMPT (the core generalization)
# ============================================================
# Instead of Python hardcoding a fixed set of step "types" (v5's
# top_bu_hp_analysis template), the LLM proposes the FULL ordered list of
# metric-spec steps needed to answer the question, including which step(s)
# the entity-dependent steps depend on. Python's job downstream is only to
# validate, resolve dependencies, and render/execute - never to template-match
# the question against a fixed catalog of shapes.

WORKFLOW_PLAN_PROMPT = ChatPromptTemplate.from_template(
"""
You are an HR Analytics workflow planner. The user's question requires
MULTIPLE dependent DAX queries. Break it down into an ordered list of steps.

User Question:
{question}

Retrieved Measures (use EXACT names from this list, never invent a measure):
{measures}

Available Tables and Columns (use EXACT column names, never invent one):
{dimensions}

CRITICAL MODELING NOTES (same rules apply to every step):
- "FTE" / "Full-Time" -> filters.Employment_Type = "Full Time". There is no
  separate FTE measure; use "Headcount" with that filter.
- "Annualized Attrition %" and "High Performer Annualized Attrition %" are
  real existing measures - select them directly, never compute annualization
  manually.
- "Vol-Reg" / "voluntary attrition" -> Exit_Type = "Vol-Reg". Express
  "Vol-Reg Annualized Attrition %" by wrapping "Annualized Attrition %" (or
  the High Performer version) with measure_filters: {{"Exit_Type": "Vol-Reg"}}.
  IMPORTANT: never put Exit_Type in a step's step-wide "filters" if that
  same step ALSO returns a non-attrition measure like "Headcount" or
  "High Performers Overall" - a step-wide Exit_Type filter would incorrectly
  restrict those headcount measures to only voluntarily-exited employees.
  Always scope Exit_Type to just the specific attrition measure via
  measure_filters instead.
- "High Performers" -> Rating IN {{4, 5}} on Emp_Master (column is "Rating",
  NOT "Overall_Rating"). Put Rating in the step's base "filters" (so it
  restricts the whole row context, including any groupby breakdown), and
  also reflect it in measure_filters only if a measure needs it explicitly
  recomputed under that restriction - don't apply the exact same predicate
  in both places on the same step.
- A dimension-wise split (e.g. "gender-wise headcount") is a "groupby" entry,
  not a search for a dimension-specific measure name.
- Prefer "Emp_Master" unless the question is about manager span-of-control
  (then use "People Manager Complete Span" / "_CS" columns).

DEPENDENCY RULES:
- The FIRST step is usually "find the top/bottom N <entity>".
- Any LATER step that should be restricted to those same entities must set:
    "depends_on": "<step_id of the entity-producing step>"
    "depends_on_column": "<output column name in that step holding the entity values, usually the groupby column name>"
    "entity_column": "<column in THIS step to filter by those entity values, usually the same column name>"
- Steps that don't need entity restriction (rare) can omit depends_on.
- Every step_id must be unique. Steps are executed in the given order.
- Maximum {max_steps} steps. Combine sub-asks for the SAME group of entities
  into as few steps as possible (e.g. one step can return several measures
  and several groupby columns at once - you do not need a separate step per
  measure or per dimension cut).

Respond with JSON only, matching exactly this schema:
{{
    "steps": [
        {{
            "step_id": "top_entities",
            "description": "short human description",
            "table": "Emp_Master",
            "measures": ["Headcount"],
            "measure_filters": {{}},
            "measure_labels": {{}},
            "groupby": ["Business_Unit_Name"],
            "filters": {{"Year": 2026, "Employment_Type": "Full Time"}},
            "top_n": 3,
            "sort": {{"by": "Headcount", "order": "desc"}},
            "depends_on": null,
            "depends_on_column": null,
            "entity_column": null
        }}
    ]
}}

Return JSON only, no markdown, no commentary.
"""
)


def plan_workflow(question: str, measures: List[Dict], dimensions: str) -> List[Dict]:
    response = llm.invoke(
        WORKFLOW_PLAN_PROMPT.format(
            question=question,
            measures=json.dumps([m["measure_name"] for m in measures], indent=2),
            dimensions=dimensions,
            max_steps=MAX_WORKFLOW_STEPS,
        )
    )
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "")
    parsed = json.loads(content)
    steps = parsed.get("steps", [])
    if not steps:
        raise ValueError("Workflow planner returned no steps")
    if len(steps) > MAX_WORKFLOW_STEPS:
        raise ValueError(f"Workflow planner returned {len(steps)} steps, exceeding cap of {MAX_WORKFLOW_STEPS}")
    return steps


# ============================================================
# VALIDATION
# ============================================================
# Validates against the REAL measures.json / dimensions.json, so a
# hallucinated measure or column is caught before any DAX is generated,
# rather than failing opaquely at Power BI execution time.

# Measures whose value would be WRONG if a step-wide Exit_Type filter (e.g.
# Vol-Reg) were applied to them, because they are headcount/population
# measures, not exit/attrition measures. Used to auto-fix a common planning
# mistake: putting Exit_Type in step-wide "filters" when the step also
# returns one of these.
NON_ATTRITION_MEASURE_HINTS = (
    "headcount", "high performers overall", "active high performers",
    "average high performer headcount", "people manager count",
    "female people managers", "new hire", "average tenure", "average age",
    "diversity",
)


def _looks_non_attrition(measure_name: str) -> bool:
    name = measure_name.lower()
    return any(hint in name for hint in NON_ATTRITION_MEASURE_HINTS) and "attrition" not in name and "exit" not in name


def autofix_exit_type_scope(spec: Dict) -> Dict:
    """
    If a step has Exit_Type in its step-wide 'filters' AND also includes a
    non-attrition measure (Headcount, High Performers Overall, ...), move
    Exit_Type out of the step-wide filters and into measure_filters for only
    the measures that are actually attrition-related. This self-heals the
    most common planning mistake instead of forcing a repair round-trip.
    """
    filters = spec.get("filters", {}) or {}
    if "Exit_Type" not in filters:
        return spec

    measures = spec.get("measures", []) or []
    non_attrition = [m for m in measures if _looks_non_attrition(m)]
    if not non_attrition:
        return spec  # every measure in this step is attrition-related; step-wide filter is fine

    spec = dict(spec)
    spec["filters"] = {k: v for k, v in filters.items() if k != "Exit_Type"}
    measure_filters = dict(spec.get("measure_filters") or {})
    for m in measures:
        if m in non_attrition:
            continue  # do not apply Exit_Type to headcount-style measures
        existing = dict(measure_filters.get(m, {}))
        existing.setdefault("Exit_Type", filters["Exit_Type"])
        measure_filters[m] = existing
    spec["measure_filters"] = measure_filters
    return spec


def validate_metric_spec(spec: Dict, retrieved_measures: List[Dict]) -> List[str]:
    errors = []
    table = spec.get("table", "Emp_Master")
    if table not in TABLE_NAMES:
        errors.append(f"Unknown table: '{table}'")

    measures = spec.get("measures", [])
    if not measures:
        errors.append("metric spec has no measures")
    for m in measures:
        if not validate_measure_exists(m):
            errors.append(f"Unknown measure: '{m}'")

    for m, wrap in (spec.get("measure_filters") or {}).items():
        if m not in measures:
            errors.append(f"measure_filters references '{m}' which is not in measures list")
        for col in (wrap or {}).keys():
            if col not in ALL_COLUMNS:
                errors.append(f"Unknown column in measure_filters: '{col}'")

    for col in spec.get("groupby", []) or []:
        if col not in ALL_COLUMNS:
            errors.append(f"Unknown groupby column: '{col}'")

    # Validate normal filters
    for col in (spec.get("filters") or {}).keys():
        if col in {"Year", "Start_Date", "End_Date"}:
            continue

        if col not in ALL_COLUMNS:
            errors.append(f"Unknown filter column: '{col}'")

    # Validate result filters (measure filters after aggregation)
    for rf in (spec.get("result_filters") or []):

        measure = rf.get("measure")

        if not measure:
            errors.append("result_filter missing measure")
            continue

        if not validate_measure_exists(measure):
            errors.append(
                f"Unknown measure in result_filters: '{measure}'"
            )

        if rf.get("operator") not in {
            ">",
            ">=",
            "<",
            "<=",
            "=",
            "!="
        }:
            errors.append(
                f"Invalid operator '{rf.get('operator')}' in result_filters"
            )

        if "value" not in rf:
            errors.append(
                f"result_filter for '{measure}' missing value"
            )
    return errors


def validate_workflow_plan(steps: List[Dict], retrieved_measures: List[Dict]) -> List[str]:
    errors = []
    seen_ids = set()
    for i, step in enumerate(steps):
        step_id = step.get("step_id")
        if not step_id:
            errors.append(f"Step {i} missing step_id")
            continue
        if step_id in seen_ids:
            errors.append(f"Duplicate step_id: '{step_id}'")
        seen_ids.add(step_id)

        errors.extend(f"[{step_id}] {e}" for e in validate_metric_spec(step, retrieved_measures))

        depends_on = step.get("depends_on")
        if depends_on and depends_on not in seen_ids:
            errors.append(f"[{step_id}] depends_on '{depends_on}' is not an earlier step")
        if depends_on:
            entity_col = step.get("entity_column")
            if not entity_col or entity_col not in ALL_COLUMNS:
                errors.append(f"[{step_id}] depends_on set but entity_column is missing/invalid")

    return errors


# ============================================================
# WORKFLOW EXECUTION HELPERS
# ============================================================

def extract_column_values(result: Dict, column_label: str) -> List:
    """Pull distinct values of `column_label` out of a Power BI executeQueries
    result. Power BI prefixes returned column names with the table/section
    alias (e.g. "[Business_Unit_Name]" or "Business_Unit_Name"), so we match
    flexibly on suffix rather than requiring an exact key match."""
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
        # fall back to the first column if no exact/suffix match found
        matching_key = list(rows[0].keys())[0]

    values = []
    for row in rows:
        v = row.get(matching_key)
        if v is not None and v not in values:
            values.append(v)
    return values


# ============================================================
# POWER BI EXECUTION
# ============================================================

def execute_dax_query(dax_query: str):
    """Execute DAX query against Power BI. Unchanged from v5 apart from
    reading WORKSPACE_ID/DATASET_ID from env config defined at the top of
    this file."""
    credential = InteractiveBrowserCredential()
    token = credential.get_token("https://analysis.windows.net/powerbi/api/.default")
    access_token = token.token

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "queries": [{"query": dax_query}],
        "serializerSettings": {"includeNulls": True},
    }

    response = requests.post(
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/datasets/{DATASET_ID}/executeQueries",
        headers=headers,
        json=payload,
        verify=False,
    )

    try:
        body = response.json()
    except Exception:
        body = {"error": response.text}

    return response.status_code, body


# ============================================================
# REPAIR LOGIC
# ============================================================
# Kept conceptually the same as v5, but now reusable for EVERY workflow step
# too (v5 only ever repaired the single standard/comparison DAX query; a
# failing workflow step just aborted the entire question with no retry).

REPAIR_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert Power BI DAX developer.

The following DAX failed.

ORIGINAL DAX:
{dax}

POWER BI ERROR:
{error}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

RULES:
1. Fix ONLY the DAX.
2. Do NOT invent measures.
3. Do NOT invent dimensions or columns.
4. Preserve the original intent (same measures, same filters, same grouping).
5. Return DAX ONLY.
6. No markdown.
7. No explanation.
"""
)


def repair_dax(dax: str, error: Dict, retrieved_measures: List[Dict], dimensions: str) -> str:
    response = llm.invoke(
        REPAIR_PROMPT.format(
            dax=dax,
            error=json.dumps(error, indent=2),
            measures=json.dumps([m["measure_name"] for m in retrieved_measures], indent=2),
            dimensions=dimensions,
        )
    )
    fixed = response.content.strip()
    fixed = fixed.replace("```DAX", "").replace("```dax", "").replace("```", "")
    return fixed.strip()


def execute_with_repair(dax_query: str, retrieved_measures: List[Dict], dimensions: str,
                         max_attempts: int = MAX_REPAIR_ATTEMPTS):
    """Execute a DAX query, repairing and retrying on failure up to
    max_attempts times. Returns (status, result, final_dax, attempts_used)."""
    current_dax = dax_query
    last_status, last_result = None, None

    for attempt in range(max_attempts + 1):
        status, result = execute_dax_query(current_dax)
        last_status, last_result = status, result
        if status == 200:
            return status, result, current_dax, attempt
        if attempt < max_attempts:
            current_dax = repair_dax(current_dax, result, retrieved_measures, dimensions)

    return last_status, last_result, current_dax, max_attempts


# ============================================================
# RESULT PROCESSING
# ============================================================

def is_empty_result(result: Dict) -> bool:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        return len(rows) == 0
    except Exception:
        return False


def convert_blank_to_zero(result: Dict) -> Dict:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        for row in rows:
            for key, value in row.items():
                if value is None:
                    row[key] = 0
    except Exception:
        pass
    return result


def format_percentages(result: Dict) -> Dict:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        for row in rows:
            for key, value in row.items():
                if value is None:
                    continue
                if "%" in key and isinstance(value, (int, float)):
                    row[key] = f"{value * 100:.2f}%"
    except Exception:
        pass
    return result


def result_to_text(result: Dict) -> str:
    try:
        rows = result["results"][0]["tables"][0]["rows"]
        if len(rows) == 0:
            return "No records returned."
        lines = []
        for row in rows:
            items = [f"{k}: {v}" for k, v in row.items()]
            lines.append(" | ".join(items))
        return "\n".join(lines)
    except Exception:
        return json.dumps(result, indent=2)


# ============================================================
# ANSWER GENERATION
# ============================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
"""
You are an expert HR Data Analyst.

User Question:
{question}

Analysis Results:
{result}

RULES:
1. Answer using ONLY the provided results.
2. Never invent numbers.
3. If a section has no records, say so plainly.
4. Percentages are already formatted as XX.XX% - use them as given.
5. Organize the answer to mirror the structure of the question (e.g. one
   section per Business Unit if the question asked for a per-BU breakdown).
6. Use concise business language and tables/bullets where that improves
   clarity.
7. Do not mention DAX, Power BI, or any internal step names.

Return the final business answer.
"""
)


def generate_final_answer(question: str, results) -> str:
    if isinstance(results, dict) and "workflow_steps" in results:
        text_results = []
        for step_id, payload in results["workflow_steps"].items():
            result = payload["result"] if isinstance(payload, dict) and "result" in payload else payload
            description = payload.get("description", step_id) if isinstance(payload, dict) else step_id
            processed = convert_blank_to_zero(result)
            processed = format_percentages(processed)
            step_text = result_to_text(processed)
            text_results.append(f"STEP — {description}:\n{step_text}")
        combined_results = "\n\n".join(text_results)
    else:
        processed = convert_blank_to_zero(results)
        processed = format_percentages(processed)
        combined_results = result_to_text(processed)

    response = llm.invoke(
        ANSWER_PROMPT.format(question=question, result=combined_results)
    )
    return response.content


def handle_empty_result(result: Dict) -> Optional[str]:
    if is_empty_result(result):
        return "No records returned for the applied filters."
    return None


# ============================================================
# LANGGRAPH NODES
# ============================================================

def retrieve_measures_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 1: RETRIEVING MEASURES =====")
    retrieved = retrieve_measures(state["question"], top_k=25)
    print("Retrieved measures:", [m["measure_name"] for m in retrieved])
    state["retrieved_measures"] = retrieved
    state["dimensions"] = get_dimension_context()
    return state


def extract_intent_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 2: CLASSIFYING INTENT + EXTRACTING SPEC =====")
    try:
        # Single LLM call: classifies AND returns the full spec for
        # standard/comparison. For workflow it returns only intent_type +
        # reasoning; WORKFLOW_PLAN_PROMPT handles the step planning.
        parsed = classify_and_extract_intent(
            state["question"], state["retrieved_measures"], state["dimensions"]
        )
    except Exception as e:
        state["error_messages"] = [f"Intent extraction failed: {e}"]
        return state

    intent_type = parsed.get("intent_type", "standard")
    if intent_type not in ("standard", "comparison", "workflow"):
        intent_type = "standard"

    print("Classified as:", intent_type, "-", parsed.get("reasoning", ""))
    state["intent_type"] = intent_type
    state["intent"] = parsed          # full spec already embedded for std/comparison
    return state


def standard_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 3A: BUILDING STANDARD QUERY =====")

    try:
        # Spec was already extracted in extract_intent_node — no second LLM call.
        spec = dict(state["intent"])
        spec.pop("intent_type", None)
        spec.pop("reasoning", None)
        spec["filters"] = apply_defaults(
            state["question"], spec.get("filters") or {},
            table=spec.get("table", "Emp_Master"),
        )
        spec = autofix_exit_type_scope(spec)

        errors = validate_metric_spec(spec, state["retrieved_measures"])
        if errors:
            state["error_messages"] = state.get("error_messages", []) + errors
            return state

        dax_query = render_metric_spec_dax(spec)
        print("Generated DAX:\n", dax_query)

        state["enriched_intent"] = spec
        state["dax_query"] = dax_query
        state["final_dax"] = dax_query
        state["repair_attempts"] = 0
    except Exception as e:
        state["error_messages"] = state.get("error_messages", []) + [f"Standard query build failed: {e}"]
    return state


def comparison_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 3B: BUILDING COMPARISON QUERY =====")
    try:
        # Spec was already extracted in extract_intent_node — no second LLM call.
        spec = dict(state["intent"])
        spec.pop("intent_type", None)
        spec.pop("reasoning", None)
        spec["filters"] = apply_defaults(
            state["question"], spec.get("filters") or {},
            table=spec.get("table", "Emp_Master"),
        )

        entity = spec.get("entity")
        measure = spec.get("measure")
        if entity and entity not in ALL_COLUMNS:
            state["error_messages"] = state.get("error_messages", []) + [f"Invalid entity column: '{entity}'"]
            return state
        if measure and not validate_measure_exists(measure):
            state["error_messages"] = state.get("error_messages", []) + [f"Invalid measure: '{measure}'"]
            return state

        dax_query = render_comparison_dax(spec)
        print("Generated Comparison DAX:\n", dax_query)

        state["enriched_intent"] = spec
        state["dax_query"] = dax_query
        state["final_dax"] = dax_query
        state["repair_attempts"] = 0
    except Exception as e:
        state["error_messages"] = state.get("error_messages", []) + [f"Comparison query build failed: {e}"]
    return state


def workflow_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 3C: PLANNING WORKFLOW =====")
    try:
        steps = plan_workflow(state["question"], state["retrieved_measures"], state["dimensions"])

        fixed_steps = []
        for step in steps:
            step["filters"] = apply_defaults(
                state["question"], step.get("filters", {}),
                allow_exit_type_default=False,
                table=step.get("table", "Emp_Master"),
            )
            step = autofix_exit_type_scope(step)
            fixed_steps.append(step)
        steps = fixed_steps

        errors = validate_workflow_plan(steps, state["retrieved_measures"])
        if errors:
            state["error_messages"] = state.get("error_messages", []) + errors
            return state

        print(f"Planned {len(steps)} steps:")
        for s in steps:
            print(f"  - [{s['step_id']}] {s.get('description', '')}"
                  f"{' (depends_on=' + s['depends_on'] + ')' if s.get('depends_on') else ''}")

        state["workflow_plan"] = steps
        state["repair_attempts"] = 0
    except Exception as e:
        state["error_messages"] = state.get("error_messages", []) + [f"Workflow planning failed: {e}"]
    return state


def execute_workflow_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 4C: EXECUTING WORKFLOW =====")

    # ── Resume context ────────────────────────────────────────────────────
    # On the first call these are None / empty (normal entry).
    # On a re-entry after repair_workflow_step_node ran, workflow_failed_step_id
    # and workflow_step_repair_dax are set: we retry just that step with the
    # fixed DAX, then continue with the remaining steps as normal.
    resume_step_id  = state.get("workflow_failed_step_id")
    resume_dax      = state.get("workflow_step_repair_dax")
    is_resume       = bool(resume_step_id and resume_dax)

    # Carry over already-completed results and captured entities from previous
    # runs so we never re-execute steps that already succeeded.
    # On a resume after repair, workflow_results lives in state["execution_result"]
    # (that's what was saved when the node exited on failure).
    prior = state.get("workflow_results") or state.get("execution_result") or {}
    workflow_results = dict(prior)
    if "workflow_steps" not in workflow_results:
        workflow_results["workflow_steps"] = {}
    workflow_dax = dict(state.get("workflow_dax") or {})

    # Rebuild step_entities from whatever steps already have results,
    # so dependent steps that come AFTER the failed+repaired step still
    # get the correct entity lists from step 0 / any earlier steps.
    step_entities: Dict[str, List] = {}
    for step in state["workflow_plan"]:
        sid = step["step_id"]
        if sid in workflow_results["workflow_steps"]:
            groupby_cols = step.get("groupby", [])
            if groupby_cols:
                step_entities[sid] = extract_column_values(
                    workflow_results["workflow_steps"][sid]["result"], groupby_cols[0]
                )

    # ── Main execution loop ───────────────────────────────────────────────
    for idx, step in enumerate(state["workflow_plan"]):
        step_id = step["step_id"]

        # Skip steps that already completed successfully in a previous pass.
        if step_id in workflow_results["workflow_steps"]:
            print(f"--- Skipping already-completed step: {step_id} ---")
            continue

        print(f"\n--- Executing step: {step_id} ({step.get('description', '')}) ---")

        # ── Resolve entity dependency ─────────────────────────────────────
        entity_clause = None
        depends_on = step.get("depends_on")
        if depends_on:
            entity_values = step_entities.get(depends_on)
            if not entity_values:
                state["error_messages"] = state.get("error_messages", []) + [
                    f"Step '{step_id}' depends on '{depends_on}' but no entities were captured from it"
                ]
                state["execution_status"] = 500
                state["execution_result"] = workflow_results
                state["workflow_dax"] = workflow_dax
                # This is a planning/dependency error, not a DAX syntax error —
                # repair cannot fix it, so go straight to error_handler.
                state["workflow_failed_step_id"] = None
                state["workflow_step_repair_dax"] = None
                return state
            entity_clause = build_entity_in_clause(
                step["entity_column"], entity_values,
                table=step.get("table") if step.get("table") != "Emp_Master" else None,
            )

        # ── Determine DAX to try ──────────────────────────────────────────
        if is_resume and step_id == resume_step_id:
            # We are resuming at the exact step that previously failed.
            # Use the repaired DAX instead of re-rendering from scratch.
            dax_query = resume_dax
            print(f"[RESUME] Using repaired DAX for '{step_id}'")
            is_resume = False          # clear so subsequent steps render normally
        else:
            try:
                dax_query = render_metric_spec_dax(step, entity_filter_clause=entity_clause)
            except Exception as e:
                state["error_messages"] = state.get("error_messages", []) + [
                    f"Step '{step_id}' DAX build failed: {e}"
                ]
                state["execution_status"] = 500
                state["execution_result"] = workflow_results
                state["workflow_dax"] = workflow_dax
                state["workflow_failed_step_id"] = None
                state["workflow_step_repair_dax"] = None
                return state

        print(f"DAX:\n{dax_query}")
        workflow_dax[step_id] = dax_query

        # ── Execute ───────────────────────────────────────────────────────
        status, result = execute_dax_query(dax_query)
        print(f"Status: {status}")

        if status != 200:
            print(f"Step '{step_id}' failed — routing to repair node")
            state["execution_status"] = 500
            state["execution_result"] = workflow_results   # preserve successes so far
            state["workflow_results"] = workflow_results   # also keep in dedicated key for resume
            state["workflow_dax"] = workflow_dax
            state["workflow_failed_step_id"] = step_id
            state["workflow_failed_step_index"] = idx
            state["workflow_step_error"] = result
            state["workflow_step_repair_dax"] = None
            return state

        # ── Step succeeded ────────────────────────────────────────────────
        workflow_results["workflow_steps"][step_id] = {
            "description": step.get("description", step_id),
            "result": result,
        }
        # Capture entities for downstream dependent steps.
        groupby_cols = step.get("groupby", [])
        if groupby_cols:
            primary_col = groupby_cols[0]
            step_entities[step_id] = extract_column_values(result, primary_col)
            print(f"Captured entities from '{step_id}': {step_entities[step_id]}")

        # Clear any leftover repair state now that this step succeeded.
        state["workflow_failed_step_id"] = None
        state["workflow_step_repair_dax"] = None
        state["workflow_step_error"] = None
        state["workflow_step_repair_attempts"] = 0

    # ── All steps done ────────────────────────────────────────────────────
    state["execution_status"] = 200
    state["execution_result"] = workflow_results
    state["workflow_dax"] = workflow_dax
    state["workflow_failed_step_id"] = None
    state["workflow_step_repair_dax"] = None
    return state


def execute_dax_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print(f"\n===== STEP 4: EXECUTING DAX (Attempt {state['repair_attempts'] + 1}) =====")
    status, result = execute_dax_query(state["final_dax"])
    print(f"Status: {status}")
    state["execution_status"] = status
    state["execution_result"] = result
    return state


def repair_dax_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print(f"\n===== STEP 5: REPAIRING DAX (Attempt {state['repair_attempts'] + 1}) =====")
    repaired = repair_dax(
        state["final_dax"], state["execution_result"], state["retrieved_measures"], state["dimensions"]
    )
    print("Repaired DAX:\n", repaired)
    state["final_dax"] = repaired
    state["repair_attempts"] += 1
    return state


def repair_workflow_step_node(state: HRAnalyticsState) -> HRAnalyticsState:
    """
    Repairs a single failed workflow step's DAX and stores the fixed version
    back in state so execute_workflow_node can resume at exactly that step.

    Mirrors repair_dax_node but operates on workflow step context:
      - Reads the failing DAX from state["workflow_dax"][failed_step_id]
      - Reads the raw Power BI error from state["workflow_step_error"]
      - Writes the repaired DAX to state["workflow_step_repair_dax"]
      - Increments state["workflow_step_repair_attempts"]

    After this node, the graph routes back to execute_workflow_node which
    picks up at the failed step with the fixed DAX (skipping already-
    completed steps so they are never re-executed).

    If repair attempts are exhausted, sets execution_status=500 so the
    graph routes to error_handler instead.
    """
    failed_step_id = state.get("workflow_failed_step_id")
    current_attempts = state.get("workflow_step_repair_attempts") or 0

    print(f"\n===== REPAIRING WORKFLOW STEP: '{failed_step_id}' "
          f"(Attempt {current_attempts + 1}/{MAX_REPAIR_ATTEMPTS}) =====")

    if current_attempts >= MAX_REPAIR_ATTEMPTS:
        print(f"Step '{failed_step_id}' exhausted {MAX_REPAIR_ATTEMPTS} repair attempts — giving up")
        state["error_messages"] = state.get("error_messages", []) + [
            f"Step '{failed_step_id}' could not be repaired after {MAX_REPAIR_ATTEMPTS} attempt(s). "
            f"Last error: {json.dumps(state.get('workflow_step_error', {}))[:400]}"
        ]
        state["execution_status"] = 500
        return state

    # The DAX we need to repair is what the step last tried.
    failing_dax = (state.get("workflow_dax") or {}).get(failed_step_id, "")
    error_body  = state.get("workflow_step_error") or {}

    if not failing_dax:
        state["error_messages"] = state.get("error_messages", []) + [
            f"Repair node: no DAX found for step '{failed_step_id}' in workflow_dax"
        ]
        state["execution_status"] = 500
        return state

    print(f"Failing DAX:\n{failing_dax}")
    print(f"PBI Error:\n{json.dumps(error_body, indent=2)[:600]}")

    repaired = repair_dax(
        failing_dax, error_body, state["retrieved_measures"], state["dimensions"]
    )
    print(f"Repaired DAX:\n{repaired}")

    state["workflow_step_repair_dax"] = repaired
    state["workflow_step_repair_attempts"] = current_attempts + 1
    # Keep execution_status at 500 so the routing function can distinguish
    # "needs retry" (has repair_dax set, not exhausted) from "give up".
    return state


def generate_answer_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== STEP 6: GENERATING ANSWER =====")

    result = state["execution_result"]
    if isinstance(result, dict) and "workflow_steps" not in result:
        empty_message = handle_empty_result(result)
        if empty_message:
            state["answer"] = empty_message
            return state

    answer = generate_final_answer(state["question"], result)
    print("Final answer:\n", answer)
    state["answer"] = answer
    return state


def error_handler_node(state: HRAnalyticsState) -> HRAnalyticsState:
    print("\n===== ERROR HANDLING =====")
    error_msg = "Unable to process the request due to the following issues:\n"
    error_msg += "\n".join(f"- {err}" for err in state.get("error_messages", []))
    if state.get("execution_result"):
        error_msg += f"\n\nLast execution detail:\n{json.dumps(state['execution_result'], indent=2)[:1500]}"
    state["answer"] = error_msg
    return state


# ============================================================
# ROUTING LOGIC
# ============================================================

def route_after_intent(state: HRAnalyticsState) -> str:
    if state.get("error_messages"):
        return "error_handler"
    intent_type = state.get("intent_type", "standard")
    if intent_type in ("standard", "comparison", "workflow"):
        return intent_type
    return "standard"


def route_after_build(state: HRAnalyticsState) -> str:
    """After standard/comparison/workflow node runs, route to the right
    execution path - or straight to error_handler if building failed."""
    if state.get("error_messages"):
        return "error_handler"
    intent_type = state.get("intent_type", "standard")
    if intent_type == "workflow":
        return "execute_workflow"
    return "execute_dax"


def should_continue_after_execution(state: HRAnalyticsState) -> str:
    if state["execution_status"] == 200:
        return "generate_answer"
    elif state["repair_attempts"] < MAX_REPAIR_ATTEMPTS:
        return "repair_dax"
    else:
        state["error_messages"] = state.get("error_messages", []) + [
            f"Failed to execute DAX after {MAX_REPAIR_ATTEMPTS} repair attempts"
        ]
        return "error_handler"


def should_continue_after_workflow(state: HRAnalyticsState) -> str:
    """
    Called after execute_workflow_node returns.

    Three outcomes:
      "generate_answer"        — all steps completed (status=200)
      "repair_workflow_step"   — a step failed AND we still have repair
                                  attempts left for it → route to repair node
      "error_handler"          — step failed AND attempts exhausted (or it was
                                  a dependency/build error that repair can't fix,
                                  i.e. workflow_failed_step_id is None)
    """
    if state["execution_status"] == 200:
        return "generate_answer"

    failed_step_id = state.get("workflow_failed_step_id")
    if not failed_step_id:
        # No step ID means a dependency or DAX-build error — not repairable
        return "error_handler"

    current_attempts = state.get("workflow_step_repair_attempts") or 0
    if current_attempts < MAX_REPAIR_ATTEMPTS:
        print(f"[ROUTER] Workflow step '{failed_step_id}' failed — routing to repair "
              f"(attempt {current_attempts + 1}/{MAX_REPAIR_ATTEMPTS})")
        return "repair_workflow_step"

    return "error_handler"


def should_continue_after_workflow_repair(state: HRAnalyticsState) -> str:
    """
    Called after repair_workflow_step_node returns.

    Two outcomes:
      "execute_workflow"  — repair produced a fixed DAX → retry
      "error_handler"     — repair exhausted attempts (node set status=500
                            and cleared workflow_step_repair_dax)
    """
    if state.get("workflow_step_repair_dax") and state["execution_status"] == 500:
        # Repaired DAX is ready; go back to execute_workflow to retry the step
        return "execute_workflow"
    return "error_handler"


def should_continue_after_repair(state: HRAnalyticsState) -> str:
    return "execute_dax"


# ============================================================
# BUILD LANGGRAPH
# ============================================================
#
# Full graph topology (standard / comparison path unchanged):
#
#   retrieve_measures
#         │
#   extract_intent ──────────────────────────────┐
#     │          │          │                    │
#  standard  comparison  workflow          error_handler ──► END
#     │          │          │
#  execute_dax  execute_dax  execute_workflow ◄──────────────────────────┐
#     │                          │          │                            │
#  (200) ──► generate_answer     │          └──► repair_workflow_step ───┘
#     │                          │                     │
#  (fail) ──► repair_dax         │                  (exhausted)
#     │          │               │                     │
#     └──────────┘        error_handler            error_handler
#
# The key addition: execute_workflow now has a proper retry edge
# (execute_workflow → repair_workflow_step → execute_workflow)
# matching what the standard path already had.

def build_hr_analytics_graph():
    workflow = StateGraph(HRAnalyticsState)

    workflow.add_node("retrieve_measures",      retrieve_measures_node)
    workflow.add_node("extract_intent",         extract_intent_node)
    workflow.add_node("standard",               standard_node)
    workflow.add_node("comparison",             comparison_node)
    workflow.add_node("workflow",               workflow_node)
    workflow.add_node("execute_dax",            execute_dax_node)
    workflow.add_node("execute_workflow",        execute_workflow_node)
    workflow.add_node("repair_dax",             repair_dax_node)
    workflow.add_node("repair_workflow_step",   repair_workflow_step_node)
    workflow.add_node("generate_answer",        generate_answer_node)
    workflow.add_node("error_handler",          error_handler_node)

    workflow.set_entry_point("retrieve_measures")
    workflow.add_edge("retrieve_measures", "extract_intent")

    workflow.add_conditional_edges(
        "extract_intent",
        route_after_intent,
        {
            "standard":     "standard",
            "comparison":   "comparison",
            "workflow":     "workflow",
            "error_handler":"error_handler",
        },
    )

    workflow.add_conditional_edges(
        "standard", route_after_build,
        {"execute_dax": "execute_dax", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "comparison", route_after_build,
        {"execute_dax": "execute_dax", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "workflow", route_after_build,
        {"execute_workflow": "execute_workflow", "error_handler": "error_handler"},
    )

    # Standard / comparison execution + repair loop (unchanged)
    workflow.add_conditional_edges(
        "execute_dax",
        should_continue_after_execution,
        {
            "generate_answer": "generate_answer",
            "repair_dax":      "repair_dax",
            "error_handler":   "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "repair_dax",
        should_continue_after_repair,
        {"execute_dax": "execute_dax"},
    )

    # Workflow execution + per-step repair loop (new)
    workflow.add_conditional_edges(
        "execute_workflow",
        should_continue_after_workflow,
        {
            "generate_answer":      "generate_answer",
            "repair_workflow_step": "repair_workflow_step",
            "error_handler":        "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "repair_workflow_step",
        should_continue_after_workflow_repair,
        {
            "execute_workflow": "execute_workflow",
            "error_handler":    "error_handler",
        },
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("error_handler",   END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# ============================================================
# MAIN PIPELINE
# ============================================================

def ask(question: str, session_id: str) -> str:
    """
    Main entry point for HR Analytics agent.
    
    Args:
        question: User question
        session_id: Session identifier for tracking conversation context
    
    Returns:
        Business-readable answer string
    """
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)
    print("=" * 80)

    app = build_hr_analytics_graph()

    initial_state: HRAnalyticsState = {
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
        # Workflow step-level repair fields (all start empty)
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
# TEST FUNCTIONS
# ============================================================

def test_pipeline():
    import uuid
    
    session_id = str(uuid.uuid4())
    
    test_questions = [
        "What is the headcount?",
        "Provide attrition count by business unit",
        "Managers whose complete span increased year over year",
        "Headcount trend for Gen Z",
        ("Show me, for the year 2026, the top 3 Business Units (BUs) by highest "
         "Full-Time Employee (FTE) headcount. For each of these BUs, provide: "
         "The overall FTE headcount. The overall annualized voluntary attrition "
         "percentage (Vol-Reg). For High Performers (where Overall_Rating is 4 "
         "or 5) in each BU: The overall headcount of High Performers. The "
         "annualized voluntary attrition percentage (Vol-Reg) for High "
         "Performers. The gender-wise headcount and Vol-Reg attrition % for "
         "High Performers (Male/Female) within each BU."),
    ]

    for question in test_questions:
        print(f"\n{'=' * 80}\nTESTING: {question}\n{'=' * 80}")
        try:
            answer = ask(question, session_id)
            print(f"\n{'=' * 80}\nFINAL RESULT:\n{answer}\n{'=' * 80}")
        except Exception as e:
            print(f"\nERROR processing question '{question}':\n{e}\n{'-' * 80}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import uuid
    
    print("=" * 80)
    print("HR Analytics Agent - Interactive Mode")
    print("=" * 80)

    session_id = str(uuid.uuid4())
    print(f"📋 Session ID: {session_id}\n")

    test_question = ("""What is the attrition rate specifically among M7+ grades in 2025 vs 2024?""")
    print(f"\nTesting pipeline with:\n{test_question}")

    try:
        answer = ask(test_question, session_id)
        print(f"\n{'=' * 80}\nSUCCESS - RESULT:\n{answer}\n{'=' * 80}")
    except Exception as e:
        print(f"Pipeline error: {e}")