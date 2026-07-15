"""
Diversity & Inclusion Dashboard AI Agent — diversity_v7.py
===========================================================

WHY diversity_v7.py EXISTS
--------------------------
The original dv2.py had several limitations that prevented production deployment:

1. No validation — hallucinated measure/column names went straight to Power BI
2. Basic error handling — single try-catch, no repair loop per workflow step
3. No workflow engine — couldn't handle "Top 3 BUs by diversity %, then attrition for each"
4. Manual filter building — repetitive code, no generic column resolution
5. Scattered defaults — employment type/exit type applied inconsistently
6. No dependency resolution — all sub-queries were independent
7. Minimal documentation and no test suite

WHAT CHANGED IN diversity_v7.py
--------------------------------
- Full v12.py architecture: workflow engine with dependency resolution
- Validation before DAX generation (measures + columns)
- Per-step repair loops exposed as LangGraph edges
- Generic filter/DAX builders using dimensions.json auto-resolution
- Centralized BUSINESS_RULES for all defaults
- Complete error recovery with max retry limits
- Comprehensive documentation and test cases

Architecture:
  question
    → retrieve_diversity_measures
    → extract_diversity_intent  (classify: standard / comparison / topn / workflow)
         │
         ├─ standard / topn / comparison
         │     → build_diversity_dax
         │     → execute_diversity_dax ──(fail)──► repair_diversity_dax ──► execute
         │          │ (200)
         │     → generate_diversity_answer → END
         │
         └─ workflow
               → plan_diversity_workflow   (LLM plans ordered step list)
               → execute_diversity_workflow  ←──────────────────────────────┐
                    │ step succeeded                                         │
                    │ step failed ──► repair_diversity_workflow_step ────────┘
                    │ all steps done
               → generate_diversity_answer → END
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

AZURE_OPENAI_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
API_VERSION             = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

WORKSPACE_ID = os.environ.get("POWERBI_WORKSPACE_ID")
DATASET_ID   = os.environ.get("POWERBI_DATASET_ID")

DIVERSITY_MEASURES_FILE    = os.environ.get("DIVERSITY_MEASURES_FILE", "diversity_measures.json")
DIVERSITY_DIMENSIONS_FILE  = os.environ.get("DIVERSITY_DIMENSIONS_FILE", "diversity_dimension.json")

MAX_REPAIR_ATTEMPTS = 2
MAX_WORKFLOW_STEPS = 8
CURRENT_YEAR = datetime.today().year

POWERBI_RESOURCE = "https://analysis.windows.net/powerbi/api"
POWERBI_EXECUTE_URL = (
    f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
    f"/datasets/{DATASET_ID}/executeQueries"
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(DIVERSITY_MEASURES_FILE, "r", encoding="utf-8") as f:
    DIVERSITY_MEASURES = json.load(f)

with open(DIVERSITY_DIMENSIONS_FILE, "r", encoding="utf-8") as f:
    DIVERSITY_DIMENSIONS_DATA = json.load(f)
    DIVERSITY_DIMENSIONS = DIVERSITY_DIMENSIONS_DATA["tables"]

DIVERSITY_MEASURES_BY_NAME = {m["measure_name"]: m for m in DIVERSITY_MEASURES}
ALL_DIVERSITY_MEASURE_NAMES = set(DIVERSITY_MEASURES_BY_NAME.keys())

# ============================================================
# TABLE / COLUMN RESOLUTION
# ============================================================

TABLE_NAMES = list(DIVERSITY_DIMENSIONS.keys())

# Build column -> table mapping
COLUMN_TO_TABLE: Dict[str, str] = {}
for _table, _cols in DIVERSITY_DIMENSIONS.items():
    for _col in _cols:
        COLUMN_TO_TABLE[_col] = _table

ALL_COLUMNS = set(COLUMN_TO_TABLE.keys())

# Numeric columns (don't quote in DAX filters)
NUMERIC_COLUMNS = {"Year", "Month_Number"}


def qualify_table(table_name: str) -> str:
    """Wrap table name in quotes if it contains spaces."""
    return f"'{table_name}'" if " " in table_name else table_name


def resolve_column(col: str, table: Optional[str] = None) -> str:
    """
    Resolve column to fully-qualified 'Table'[Column] DAX reference.
    Raises ValueError on unknown columns.
    """
    if table:
        if col not in DIVERSITY_DIMENSIONS.get(table, []):
            raise ValueError(f"Column '{col}' not in table '{table}'")
        return f"{qualify_table(table)}[{col}]"
    
    resolved_table = COLUMN_TO_TABLE.get(col)
    if not resolved_table:
        raise ValueError(f"Unknown column '{col}'")
    return f"{qualify_table(resolved_table)}[{col}]"


def get_dimension_context() -> str:
    """Return formatted string of all tables and columns."""
    lines = []
    for table, cols in DIVERSITY_DIMENSIONS.items():
        lines.append(f"\nTABLE: {table}")
        lines.extend(f"  {c}" for c in cols)
    return "\n".join(lines)


# ============================================================
# BUSINESS RULES
# ============================================================

BUSINESS_RULES = {
    "default_year": CURRENT_YEAR,
    "default_employment_type": "Full Time",
    "default_exit_type": "Vol Reg",
}

MEASURE_ALIASES = {
    "headcount": "Head Count",
    "employee count": "Head Count",
    "diversity": "Diversity %",
    "attrition": "Attrition (%)",
    "attrition rate": "Attrition (%)",
    "attrition count": "Attrition",
    "exits": "Attrition",
    "turnover": "Attrition (%)",
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
        self._latest_year: Optional[int] = None

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

    def get_latest_year(self) -> int:
        if self._latest_year:
            return self._latest_year
        dax = 'EVALUATE ROW("LatestYear", CALCULATE(MAX(DimCal[Year]), ALL(DimCal)))'
        try:
            result = self.execute_dax(dax)
            rows = result["results"][0]["tables"][0]["rows"]
            self._latest_year = int(rows[0]["[LatestYear]"])
        except Exception:
            self._latest_year = CURRENT_YEAR
        return self._latest_year


pbi = PowerBIClient()


# ============================================================
# AGENT STATE
# ============================================================

class WorkflowStep(TypedDict, total=False):
    step_id: str
    description: str
    table: str
    measures: List[str]
    measure_filters: Dict[str, Dict]  # {measure_name: {col: value}}
    measure_labels: Dict[str, str]    # {measure_name: friendly_label}
    groupby: List[str]
    filters: Dict
    top_n: Optional[int]
    sort: Optional[Dict]
    depends_on: Optional[str]         # step_id dependency
    depends_on_column: Optional[str]  # column with entity values
    entity_column: Optional[str]      # column to filter in this step


class DiversityState(TypedDict):
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
    # Workflow step-level repair
    workflow_failed_step_id: Optional[str]
    workflow_failed_step_index: Optional[int]
    workflow_step_repair_dax: Optional[str]
    workflow_step_error: Optional[Dict]
    workflow_step_repair_attempts: int


# ============================================================
# DEFAULTS & FILTER APPLICATION
# ============================================================

def apply_defaults(question: str, filters: Dict, allow_exit_type_default: bool = True,
                   table: Optional[str] = None) -> Dict:
    """
    Apply business-rule defaults to filter dict.
    
    allow_exit_type_default is False for workflow steps to avoid
    incorrectly filtering non-attrition measures.
    """
    q = question.lower()
    filters = dict(filters)
    table_cols = set(DIVERSITY_DIMENSIONS.get(table, [])) if table else None
    
    def col_ok(col_name: str) -> bool:
        if table_cols is None:
            return True
        return col_name in table_cols
    
    # Year default
    if "Year" not in filters:
        year_match = re.search(r"\b(20\d{2})\b", q)
        filters["Year"] = int(year_match.group(1)) if year_match else BUSINESS_RULES["default_year"]
    
    # Employment Type default
    employment_terms = ["intern", "contract", "temporary", "part time", "part-time"]
    if "Employment_Type_DEIB" not in filters and col_ok("Employment_Type_DEIB"):
        if "full time" in q or "full-time" in q or "fte" in q:
            filters["Employment_Type_DEIB"] = "Full Time"
        elif not any(x in q for x in employment_terms):
            filters["Employment_Type_DEIB"] = BUSINESS_RULES["default_employment_type"]
    
    # Exit Type default (only for attrition queries)
    if allow_exit_type_default and "Exit_Type_DEIB" not in filters and col_ok("Exit_Type_DEIB"):
        if "attrition" in q and "vol-reg" not in q and "vol reg" not in q and "exit type" not in q:
            filters["Exit_Type_DEIB"] = BUSINESS_RULES["default_exit_type"]
        elif "vol-reg" in q or "vol reg" in q or "voluntary" in q:
            filters["Exit_Type_DEIB"] = "Vol-Reg"
    
    return filters


# ============================================================
# VALIDATION
# ============================================================

def validate_diversity_measures(measures: List[str]) -> List[str]:
    """Validate measure names against diversity_measures.json."""
    errors = []
    for m in measures:
        if m not in ALL_DIVERSITY_MEASURE_NAMES:
            errors.append(f"Unknown Diversity measure: '{m}'")
    return errors


def validate_diversity_columns(cols: List[str]) -> List[str]:
    """Validate column names against diversity_dimension.json."""
    return [f"Unknown Diversity column: '{c}'" for c in cols if c not in ALL_COLUMNS]


# ============================================================
# GENERIC DAX FILTER BUILDING
# ============================================================

def _format_filter_value(value, numeric: bool) -> str:
    """Format a single filter value for DAX."""
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
    Build a single FILTER(ALL(...), ...) clause for one column.
    Supports scalar equality and list -> IN {...}
    """
    col_ref = resolve_column(column, table)
    numeric = column in NUMERIC_COLUMNS or column == "Year"
    
    if isinstance(value, (list, tuple, set)):
        formatted = ", ".join(_format_filter_value(v, numeric) for v in value)
        return f"FILTER(ALL({col_ref}), {col_ref} IN {{{formatted}}})"
    
    formatted = _format_filter_value(value, numeric)
    return f"FILTER(ALL({col_ref}), {col_ref} = {formatted})"


def build_filters(filters: Dict, table: Optional[str] = None) -> List[str]:
    """Turn a filter dict into list of DAX FILTER(...) clauses."""
    clauses = []
    for key, value in filters.items():
        if key == "Year":
            clauses.append(build_filter_clause("Year", value, table="DimCal"))
        else:
            clauses.append(build_filter_clause(key, value, table=table))
    return clauses


def build_entity_in_clause(column: str, values: List, table: Optional[str] = None) -> str:
    """Build 'restrict to these entities' clause for workflow dependencies."""
    return build_filter_clause(column, list(values), table=table)


def build_sort(sort: Optional[Dict], fallback_alias: Optional[str] = None) -> str:
    """Build ORDER BY / sort expression for TOPN."""
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
# MEASURE-LEVEL FILTER WRAPPING
# ============================================================

def build_filtered_measure_expression(
    measure_name: str,
    extra_filters: Optional[Dict] = None,
    table: Optional[str] = None,
) -> str:
    """
    Wrap measure with extra column predicates via CALCULATE.
    Example: build_filtered_measure_expression(
        "Attrition (%)", {"Exit_Type_DEIB": "Vol Reg"}
    ) -> CALCULATE([Attrition (%)], Diversity[Exit_Type_DEIB] = "Vol Reg")
    """
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


# ============================================================
# UNIFIED METRIC-SPEC DAX RENDERER
# ============================================================

def render_metric_spec_dax(spec: Dict, entity_filter_clause: Optional[str] = None) -> str:
    """
    Render a metric spec to DAX. Works for standard, comparison, and workflow steps.
    
    Spec shape:
    {
      "table": "Diversity",
      "measures": ["Head Count", "Attrition (%)"],
      "measure_filters": {"Attrition (%)": {"Exit_Type_DEIB": "Vol Reg"}},
      "measure_labels": {"Attrition (%)": "Vol-Reg Attrition %"},
      "groupby": ["Business_Unit_Name_DEIB"],
      "filters": {"Year": 2026, "Employment_Type_DEIB": "Full Time"},
      "top_n": 3,
      "sort": {"by": "Head Count", "order": "desc"}
    }
    """
    table = spec.get("table") or "Diversity"
    measures = spec.get("measures", [])
    
    if not measures:
        raise ValueError("metric spec requires at least one measure")
    
    measure_filters = spec.get("measure_filters", {}) or {}
    measure_labels = spec.get("measure_labels", {}) or {}
    groupby = spec.get("groupby", []) or []
    filters = spec.get("filters", {}) or {}
    top_n = spec.get("top_n")
    sort = spec.get("sort")
    
    dax_filters = build_filters(filters, table=table if table != "Diversity" else None)
    if entity_filter_clause:
        dax_filters.append(entity_filter_clause)
    
    dax_groupby = [resolve_column(c, table=table if table != "Diversity" else None) for c in groupby]
    
    # Scalar (no groupby, no top_n)
    if not groupby and not top_n:
        rows = []
        for m in measures:
            label = measure_labels.get(m, m)
            expr = build_filtered_measure_expression(m, measure_filters.get(m), table=table)
            calc_body = expr if not dax_filters else f"CALCULATE(\n        {expr},\n        {', '.join(dax_filters)}\n    )"
            rows.append(f'    "{label}",\n    {calc_body}')
        return "EVALUATE\nROW(\n" + ",\n".join(rows) + "\n)"
    
    # Grouped (SUMMARIZECOLUMNS)
    summarize_parts: List[str] = []
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
        return f"EVALUATE\nTOPN(\n    {top_n},\n    {summarize},\n    {sort_expr}\n)"
    
    if sort and sort.get("by"):
        sort_by = sort["by"]
        sort_order = (sort.get("order") or "desc").upper()
        return f"EVALUATE\nCALCULATETABLE(\n    {summarize}\n)\nORDER BY [{sort_by}] {sort_order}"
    
    return f"EVALUATE\n{summarize}"


# ============================================================
# MEASURE RETRIEVAL
# ============================================================

def retrieve_diversity_measures(question: str, top_k: int = 20) -> List[Dict]:
    """Retrieve relevant measures using lexical scoring."""
    q = question.lower()
    scored = []
    
    for measure in DIVERSITY_MEASURES:
        text = (measure.get("measure_name", "") + " " + 
                measure.get("description", "")).lower()
        score = 0
        
        # Alias matching
        for alias, target in MEASURE_ALIASES.items():
            if alias in q and target == measure["measure_name"]:
                score += 10
        
        # Token matching
        for token in re.findall(r"[a-z0-9%]+", q):
            if len(token) > 2 and token in text:
                score += 1
        
        scored.append((score, measure))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


# ============================================================
# NODE 1: RETRIEVE MEASURES
# ============================================================

def retrieve_diversity_measures_node(state: DiversityState) -> DiversityState:
    """Retrieve relevant measures for the question."""
    measures = retrieve_diversity_measures(state["question"])
    dims = get_dimension_context()
    return {**state, "retrieved_measures": measures, "dimensions": dims}


# ============================================================
# NODE 2: EXTRACT INTENT
# ============================================================

DIVERSITY_MEASURES_REFERENCE = "\n\n".join(
    f"Measure: [{m['measure_name']}]\n"
    f"Description: {m['description']}\n"
    f"Formula:\n{m['formula']}"
    for m in DIVERSITY_MEASURES
)

DIVERSITY_MASTER_INTENT_PROMPT = ChatPromptTemplate.from_template(
"""You are a Diversity & Inclusion analytics expert.

USER QUESTION:
{question}

TODAY'S YEAR: {current_year}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

CLASSIFY the question as ONE of:
- "standard" — one or more metrics, optionally grouped/filtered
- "topn" — ranking (Top/Bottom N)
- "comparison" — compare metrics side-by-side
- "workflow" — complex multi-step with dependencies

THEN extract the full metric spec as JSON.

For STANDARD/TOPN/COMPARISON: Return ONE spec.
For WORKFLOW: Return ordered list of step specs with dependencies.

OUTPUT JSON ONLY. No explanation.

STANDARD/TOPN/COMPARISON spec:
{{
  "intent_type": "standard|topn|comparison",
  "table": "Diversity",
  "measures": ["Head Count", "Attrition (%)"],
  "measure_filters": {{"Attrition (%)": {{"Exit_Type_DEIB": "Vol Reg"}}}},
  "groupby": ["Business_Unit_Name_DEIB"],
  "filters": {{"Year": 2026}},
  "top_n": 3,
  "sort": {{"by": "Head Count", "order": "desc"}}
}}

WORKFLOW spec:
{{
  "intent_type": "workflow",
  "steps": [
    {{
      "step_id": "step_1",
      "description": "Find top 3 BUs by headcount",
      "table": "Diversity",
      "measures": ["Head Count"],
      "groupby": ["Business_Unit_Name_DEIB"],
      "filters": {{"Year": 2026}},
      "top_n": 3,
      "sort": {{"by": "Head Count", "order": "desc"}}
    }},
    {{
      "step_id": "step_2",
      "description": "For those BUs, show diversity by gender",
      "table": "Diversity",
      "measures": ["Diversity %"],
      "groupby": ["Business_Unit_Name_DEIB", "Gender_DEIB"],
      "filters": {{"Year": 2026}},
      "depends_on": "step_1",
      "depends_on_column": "Business_Unit_Name_DEIB",
      "entity_column": "Business_Unit_Name_DEIB"
    }}
  ]
}}
""")


def extract_diversity_intent_node(state: DiversityState) -> DiversityState:
    """Classify question and extract metric spec."""
    measures_ref = "\n\n".join([
        f"{m['measure_name']}: {m['description']}" 
        for m in state["retrieved_measures"][:15]
    ])
    
    prompt = DIVERSITY_MASTER_INTENT_PROMPT.format(
        question=state["question"],
        current_year=CURRENT_YEAR,
        measures=measures_ref,
        dimensions=state["dimensions"],
    )
    
    response = llm.invoke(prompt)
    raw = response.content.strip().strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    
    try:
        intent = json.loads(raw)
    except Exception as e:
        print(f"⚠️ Intent parse failed: {e}")
        # Fallback to standard
        intent = {
            "intent_type": "standard",
            "table": "Diversity",
            "measures": ["Head Count"],
            "filters": {},
        }
    
    intent_type = intent.get("intent_type", "standard")
    
    return {**state, "intent": intent, "intent_type": intent_type}


# ============================================================
# NODE 3: ENRICH INTENT (Apply Defaults & Validate)
# ============================================================

def enrich_diversity_intent_node(state: DiversityState) -> DiversityState:
    """Apply defaults and validate before DAX generation."""
    intent = dict(state["intent"])
    intent_type = state["intent_type"]
    question = state["question"]
    
    if intent_type in ("standard", "topn", "comparison"):
        # Apply defaults
        filters = apply_defaults(question, intent.get("filters", {}), table=intent.get("table"))
        intent["filters"] = filters
        
        # Validate
        measures = intent.get("measures", [])
        groupby = intent.get("groupby", [])
        
        measure_errors = validate_diversity_measures(measures)
        column_errors = validate_diversity_columns(groupby)
        
        if measure_errors or column_errors:
            error_msg = "\n".join(measure_errors + column_errors)
            return {
                **state,
                "enriched_intent": intent,
                "error_messages": [error_msg],
                "answer": f"Validation failed:\n{error_msg}"
            }
    
    return {**state, "enriched_intent": intent}


# ============================================================
# NODE 4: BUILD DAX (Standard/TopN/Comparison)
# ============================================================

def build_diversity_dax_node(state: DiversityState) -> DiversityState:
    """Build DAX for standard/topn/comparison queries."""
    try:
        spec = state["enriched_intent"]
        dax = render_metric_spec_dax(spec)
        return {**state, "dax_query": dax, "final_dax": dax}
    except Exception as e:
        error_msg = f"DAX generation failed: {str(e)}"
        return {
            **state,
            "error_messages": [error_msg],
            "answer": error_msg
        }


# ============================================================
# NODE 5: EXECUTE DAX
# ============================================================

def execute_diversity_dax_node(state: DiversityState) -> DiversityState:
    """Execute DAX query against Power BI."""
    dax = state.get("final_dax") or state.get("dax_query")
    
    try:
        result = pbi.execute_dax(dax)
        return {
            **state,
            "execution_status": 200,
            "execution_result": result,
            "final_dax": dax,
        }
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Execution failed: {error_msg}")
        
        # Check if we can repair
        if state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
            return {
                **state,
                "execution_status": 500,
                "error_messages": [error_msg],
            }
        else:
            # Max retries reached
            return {
                **state,
                "execution_status": 500,
                "error_messages": [f"Max repair attempts reached. Last error: {error_msg}"],
                "answer": f"Unable to execute query after {MAX_REPAIR_ATTEMPTS} attempts.\nLast error: {error_msg}"
            }


# ============================================================
# NODE 6: REPAIR DAX
# ============================================================

REPAIR_DAX_PROMPT = ChatPromptTemplate.from_template(
"""You are a DAX expert. The query below failed.

ORIGINAL QUESTION: {question}

FAILED DAX:
{dax}

ERROR:
{error}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

Fix the DAX. Common issues:
- Wrong measure/column name → use exact name from lists above
- Missing EVALUATE
- Wrong filter syntax
- Ambiguous column → qualify with table name

Return ONLY the corrected DAX. No explanation.
""")


def repair_diversity_dax_node(state: DiversityState) -> DiversityState:
    """Attempt to repair failed DAX."""
    failed_dax = state.get("final_dax") or state.get("dax_query")
    error_msg = state.get("error_messages", [])[-1] if state.get("error_messages") else "Unknown error"
    
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:20]])
    
    prompt = REPAIR_DAX_PROMPT.format(
        question=state["question"],
        dax=failed_dax,
        error=error_msg,
        measures=measures_ref,
        dimensions=state["dimensions"],
    )
    
    response = llm.invoke(prompt)
    repaired_dax = response.content.strip().strip("`").strip()
    if repaired_dax.startswith("dax"):
        repaired_dax = repaired_dax[3:].strip()
    
    print(f"🔧 Repair attempt {state.get('repair_attempts', 0) + 1}")
    
    return {
        **state,
        "final_dax": repaired_dax,
        "repair_attempts": state.get("repair_attempts", 0) + 1,
    }


# ============================================================
# NODE 7: PLAN WORKFLOW
# ============================================================

WORKFLOW_PLANNER_PROMPT = ChatPromptTemplate.from_template(
"""You are a Diversity & Inclusion analytics workflow planner.

USER QUESTION:
{question}

AVAILABLE MEASURES:
{measures}

AVAILABLE DIMENSIONS:
{dimensions}

Plan a workflow as ordered steps. Each step can depend on prior steps.

Return JSON array of steps:
[
  {{
    "step_id": "step_1",
    "description": "Find top 3 BUs by diversity %",
    "table": "Diversity",
    "measures": ["Diversity %"],
    "groupby": ["Business_Unit_Name_DEIB"],
    "filters": {{"Year": 2026}},
    "top_n": 3,
    "sort": {{"by": "Diversity %", "order": "desc"}}
  }},
  {{
    "step_id": "step_2",
    "description": "For those BUs, show attrition by gender",
    "table": "Diversity",
    "measures": ["Attrition (%)"],
    "measure_filters": {{"Attrition (%)": {{"Exit_Type_DEIB": "Vol Reg"}}}},
    "groupby": ["Business_Unit_Name_DEIB", "Gender_DEIB"],
    "filters": {{"Year": 2026}},
    "depends_on": "step_1",
    "depends_on_column": "Business_Unit_Name_DEIB",
    "entity_column": "Business_Unit_Name_DEIB"
  }}
]

Rules:
- Max {max_steps} steps
- Use depends_on for entity threading
- Apply measure_filters for per-measure wrapping (e.g., Exit_Type for attrition)
- Return JSON ONLY
""")


def plan_diversity_workflow_node(state: DiversityState) -> DiversityState:
    """Plan multi-step workflow."""
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:15]])
    
    prompt = WORKFLOW_PLANNER_PROMPT.format(
        question=state["question"],
        measures=measures_ref,
        dimensions=state["dimensions"],
        max_steps=MAX_WORKFLOW_STEPS,
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
        print(f"⚠️ Workflow parse failed: {e}")
        workflow_plan = []
    
    # Apply defaults per step
    for step in workflow_plan:
        step["filters"] = apply_defaults(
            state["question"],
            step.get("filters", {}),
            allow_exit_type_default=False,  # Use measure_filters instead
            table=step.get("table")
        )
    
    # Validate steps
    for step in workflow_plan:
        measures = step.get("measures", [])
        groupby = step.get("groupby", [])
        
        measure_errors = validate_diversity_measures(measures)
        column_errors = validate_diversity_columns(groupby)
        
        if measure_errors or column_errors:
            error_msg = f"Step {step['step_id']} validation failed:\n" + "\n".join(measure_errors + column_errors)
            return {
                **state,
                "workflow_plan": [],
                "error_messages": [error_msg],
                "answer": f"Workflow validation failed:\n{error_msg}"
            }
    
    return {**state, "workflow_plan": workflow_plan}


# ============================================================
# NODE 8: EXECUTE WORKFLOW
# ============================================================

def extract_diversity_column_values(result: Dict, column_label: str) -> List:
    """Extract distinct values of a column from Power BI result."""
    try:
        rows = result["results"][0]["tables"][0]["rows"]
    except Exception:
        return []
    if not rows:
        return []
    
    # Find matching key
    matching_key = None
    for key in rows[0].keys():
        if (key == column_label or 
            key.endswith(f"[{column_label}]") or 
            key.endswith(f".{column_label}")):
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


def execute_diversity_workflow_node(state: DiversityState) -> DiversityState:
    """Execute workflow steps with dependency resolution."""
    workflow_plan = state["workflow_plan"]
    workflow_results = dict(state.get("workflow_results", {}))
    workflow_dax = dict(state.get("workflow_dax", {}))
    
    # Check if resuming after repair
    if state.get("workflow_step_repair_dax"):
        failed_idx = state["workflow_failed_step_index"]
        step = workflow_plan[failed_idx]
        dax = state["workflow_step_repair_dax"]
        
        print(f"🔁 Retrying step {step['step_id']} with repaired DAX")
        
        try:
            result = pbi.execute_dax(dax)
            workflow_results[step["step_id"]] = result
            workflow_dax[step["step_id"]] = dax
            
            # Clear repair state and continue
            state = {
                **state,
                "workflow_results": workflow_results,
                "workflow_dax": workflow_dax,
                "workflow_step_repair_dax": None,
                "workflow_failed_step_id": None,
                "workflow_failed_step_index": None,
                "workflow_step_error": None,
            }
            
            # Continue from next step
            start_idx = failed_idx + 1
        except Exception as e:
            # Repair failed
            error_msg = str(e)
            return {
                **state,
                "workflow_step_error": {"message": error_msg},
                "error_messages": [f"Step {step['step_id']} failed after repair: {error_msg}"],
                "answer": f"Workflow failed at step {step['step_id']} even after repair."
            }
    else:
        start_idx = len(workflow_results)
    
    # Execute remaining steps
    for idx in range(start_idx, len(workflow_plan)):
        step = workflow_plan[idx]
        step_id = step["step_id"]
        
        print(f"📊 Executing workflow step: {step_id}")
        
        # Resolve dependency
        entity_filter_clause = None
        if step.get("depends_on"):
            dep_step_id = step["depends_on"]
            dep_column = step["depends_on_column"]
            entity_column = step["entity_column"]
            
            if dep_step_id not in workflow_results:
                error_msg = f"Step {step_id} depends on {dep_step_id} which hasn't run"
                return {
                    **state,
                    "error_messages": [error_msg],
                    "answer": error_msg
                }
            
            entity_values = extract_diversity_column_values(
                workflow_results[dep_step_id],
                dep_column
            )
            
            if not entity_values:
                error_msg = f"No entities found in {dep_step_id} for column {dep_column}"
                return {
                    **state,
                    "workflow_results": workflow_results,
                    "workflow_dax": workflow_dax,
                    "error_messages": [error_msg],
                    "answer": f"Workflow stopped: {error_msg}"
                }
            
            entity_filter_clause = build_entity_in_clause(entity_column, entity_values, table=step.get("table"))
        
        # Build and execute DAX
        try:
            dax = render_metric_spec_dax(step, entity_filter_clause)
            result = pbi.execute_dax(dax)
            
            workflow_results[step_id] = result
            workflow_dax[step_id] = dax
            
            print(f"✅ Step {step_id} succeeded")
        
        except Exception as e:
            # Step failed - route to repair if retries left
            error_msg = str(e)
            print(f"❌ Step {step_id} failed: {error_msg}")
            
            if state.get("workflow_step_repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
                return {
                    **state,
                    "workflow_results": workflow_results,
                    "workflow_dax": workflow_dax,
                    "workflow_failed_step_id": step_id,
                    "workflow_failed_step_index": idx,
                    "workflow_step_error": {"message": error_msg, "dax": dax},
                }
            else:
                return {
                    **state,
                    "workflow_results": workflow_results,
                    "workflow_dax": workflow_dax,
                    "error_messages": [f"Step {step_id} failed after max retries: {error_msg}"],
                    "answer": f"Workflow failed at step {step_id}"
                }
    
    # All steps succeeded
    return {
        **state,
        "workflow_results": workflow_results,
        "workflow_dax": workflow_dax,
        "execution_status": 200,
    }


# ============================================================
# NODE 9: REPAIR WORKFLOW STEP
# ============================================================

def repair_diversity_workflow_step_node(state: DiversityState) -> DiversityState:
    """Repair a failed workflow step."""
    step_id = state["workflow_failed_step_id"]
    step_idx = state["workflow_failed_step_index"]
    step = state["workflow_plan"][step_idx]
    error_info = state["workflow_step_error"]
    
    failed_dax = error_info.get("dax", "")
    error_msg = error_info.get("message", "Unknown error")
    
    measures_ref = "\n".join([m["measure_name"] for m in state["retrieved_measures"][:20]])
    
    prompt = REPAIR_DAX_PROMPT.format(
        question=f"Step {step_id}: {step.get('description', '')}",
        dax=failed_dax,
        error=error_msg,
        measures=measures_ref,
        dimensions=state["dimensions"],
    )
    
    response = llm.invoke(prompt)
    repaired_dax = response.content.strip().strip("`").strip()
    if repaired_dax.startswith("dax"):
        repaired_dax = repaired_dax[3:].strip()
    
    attempts = state.get("workflow_step_repair_attempts", 0) + 1
    print(f"🔧 Workflow step repair attempt {attempts}")
    
    return {
        **state,
        "workflow_step_repair_dax": repaired_dax,
        "workflow_step_repair_attempts": attempts,
    }


# ============================================================
# NODE 10: GENERATE ANSWER
# ============================================================

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_template(
"""You are a Diversity & Inclusion business analyst.

USER QUESTION:
{question}

QUERY TYPE: {intent_type}

POWER BI RESULT:
{result}

Write a clear, business-readable answer.

Rules:
- Start with active filters (Year, Employment Type, Exit Type)
- Use ASCII tables for multi-row results
- Round percentages to 2 decimals and show as %
- Highlight insights (trends, outliers, notable patterns)
- Keep it concise but complete
""")


def generate_diversity_answer_node(state: DiversityState) -> DiversityState:
    """Generate business-readable answer."""
    intent_type = state["intent_type"]
    
    if state.get("answer"):  # Already set by error handler
        return state
    
    if intent_type == "workflow":
        # Combine workflow results
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
        # Single result
        result_str = json.dumps(state.get("execution_result", {}), indent=2)
    
    prompt = ANSWER_GENERATION_PROMPT.format(
        question=state["question"],
        intent_type=intent_type,
        result=result_str,
    )
    
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    return {**state, "answer": answer}


# ============================================================
# ROUTING LOGIC
# ============================================================

def route_by_intent(state: DiversityState) -> str:
    """Route to appropriate path based on intent type."""
    intent_type = state.get("intent_type", "standard")
    
    if state.get("error_messages"):
        return "generate_answer"
    
    if intent_type == "workflow":
        return "plan_workflow"
    else:
        return "build_dax"


def route_after_execute(state: DiversityState) -> str:
    """Route after execution - to repair or answer generation."""
    status = state.get("execution_status", 0)
    
    if status == 200:
        return "generate_answer"
    elif state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        return "repair_dax"
    else:
        return "generate_answer"


def route_after_workflow_execute(state: DiversityState) -> str:
    """Route after workflow execution."""
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

def build_diversity_graph():
    """Build the Diversity agent LangGraph."""
    graph = StateGraph(DiversityState)
    
    # Add nodes
    graph.add_node("retrieve_measures", retrieve_diversity_measures_node)
    graph.add_node("extract_intent", extract_diversity_intent_node)
    graph.add_node("enrich_intent", enrich_diversity_intent_node)
    graph.add_node("build_dax", build_diversity_dax_node)
    graph.add_node("execute_dax", execute_diversity_dax_node)
    graph.add_node("repair_dax", repair_diversity_dax_node)
    graph.add_node("plan_workflow", plan_diversity_workflow_node)
    graph.add_node("execute_workflow", execute_diversity_workflow_node)
    graph.add_node("repair_workflow_step", repair_diversity_workflow_step_node)
    graph.add_node("generate_answer", generate_diversity_answer_node)
    
    # Entry point
    graph.set_entry_point("retrieve_measures")
    
    # Linear flow to intent classification
    graph.add_edge("retrieve_measures", "extract_intent")
    graph.add_edge("extract_intent", "enrich_intent")
    
    # Route by intent type
    graph.add_conditional_edges(
        "enrich_intent",
        route_by_intent,
        {
            "build_dax": "build_dax",
            "plan_workflow": "plan_workflow",
            "generate_answer": "generate_answer",
        }
    )
    
    # Standard/TopN/Comparison path
    graph.add_edge("build_dax", "execute_dax")
    graph.add_conditional_edges(
        "execute_dax",
        route_after_execute,
        {
            "generate_answer": "generate_answer",
            "repair_dax": "repair_dax",
        }
    )
    graph.add_edge("repair_dax", "execute_dax")  # Retry after repair
    
    # Workflow path
    graph.add_edge("plan_workflow", "execute_workflow")
    graph.add_conditional_edges(
        "execute_workflow",
        route_after_workflow_execute,
        {
            "generate_answer": "generate_answer",
            "repair_workflow_step": "repair_workflow_step",
        }
    )
    graph.add_edge("repair_workflow_step", "execute_workflow")  # Retry step
    
    # End
    graph.add_edge("generate_answer", END)
    
    return graph.compile(checkpointer=MemorySaver())


# ============================================================
# PUBLIC API
# ============================================================

def ask(question: str, session_id: str) -> str:
    """
    Main entry point for Diversity agent.
    
    Args:
        question: User question
        session_id: Session identifier for tracking context
    
    Returns:
        Business-readable answer string
    """
    print("\n" + "=" * 80)
    print("DIVERSITY QUESTION:")
    print(question)
    print("=" * 80)
    
    app = build_diversity_graph()
    
    initial_state: DiversityState = {
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

def test_diversity_pipeline():
    """Test the Diversity agent with sample questions."""
    import uuid
    
    session_id = str(uuid.uuid4())
    
    test_questions = [
        "What is the diversity % by business unit for 2025?",
        "Show attrition count and attrition % by exit type",
        "Top 3 business units by headcount, then show gender diversity for each",
        "What is the voluntary attrition rate for full-time employees?",
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
    print("Diversity & Inclusion Agent v7 (Production Grade)")
    print("=" * 80)
    
    session_id = str(uuid.uuid4())
    print(f"📋 Session ID: {session_id}\n")
    
    # Quick smoke test
    test_question = "What is the diversity % by business unit for 2025?"
    print(f"Testing with: {test_question}\n")
    
    try:
        answer = ask(test_question, session_id)
        print(f"\n{'=' * 80}\nRESULT:\n{answer}\n{'=' * 80}")
    except Exception as e:
        print(f"Error: {e}")
