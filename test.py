common test

def render_comparison_dax(spec: Dict) -> str:
    """
    Year-over-year / trend comparisons. `entity` is now OPTIONAL — when
    absent, this renders a single overall trend line with no groupby
    dimension (e.g. "attrition rate this year vs last year" with no
    per-BU/per-manager breakdown).
    """
    table = spec.get("table") or "Emp_Master"
    entity = spec.get("entity")
    measure = spec.get("measure")
    comparison = (spec.get("comparison") or "").lower()
    base_filters = spec.get("filters", {}) or {}

    if not measure:
        raise ValueError("Comparison queries require 'measure'")

    entity_col = None
    if entity:
        entity_col = resolve_column(entity, table=table if table != "Emp_Master" else None)

    dax_filters_no_year = build_filters(
        {k: v for k, v in base_filters.items() if k != "Year"},
        table=table if table != "Emp_Master" else None,
    )
    expr = get_measure_expression(measure)

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
        group_line = f"{entity_col},\n                                " if entity_col else ""
        dax = f"""EVALUATE
                    FILTER(
                        ADDCOLUMNS(
                            SUMMARIZECOLUMNS(
                                {group_line}FILTER(ALL(DimCal[Year]), DimCal[Year] = {current_year}){extra},
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
        group_line = f"{entity_col},\n    " if entity_col else ""
        dax = f"""EVALUATE
SUMMARIZECOLUMNS(
    {group_line}{time_col}{extra},
    "{measure}", {expr}
)"""

    return dax.strip()



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

        errors = []
        if not measure:
            errors.append("LLM did not return a 'measure' for this comparison query")
        elif not validate_measure_exists(measure):
            errors.append(f"Invalid measure: '{measure}'")
        if entity and entity not in ALL_COLUMNS:
            errors.append(f"Invalid entity column: '{entity}'")

        if errors:
            state["error_messages"] = state.get("error_messages", []) + errors
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



    For "comparison":
{{
    "intent_type": "comparison",
    "reasoning": "one sentence",
    "entity": "Column name to group/compare, or null if this is a single overall trend with no dimension breakdown",
    "table": "Emp_Master",
    "measure": "Measure Name",
    "comparison": "YoY",
    "time_dimension": "Year",
    "filters": {{}},
    "result_filters": []
}}


    
