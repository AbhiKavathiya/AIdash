def _extract_rows(result: Dict) -> List[Dict]:
    """
    Safely pull rows from a Power BI executeQueries response.
    Power BI often OMITS the 'rows' key entirely (rather than returning
    'rows': []) when a query matches zero records. This helper treats a
    missing key, missing table, or missing results list all as "no rows"
    instead of raising, so downstream code can rely on a simple empty-list
    check.
    """
    try:
        results = result.get("results") or []
        if not results:
            return []
        tables = results[0].get("tables") or []
        if not tables:
            return []
        return tables[0].get("rows") or []
    except (AttributeError, TypeError, IndexError, KeyError):
        return []


def generate_soc_answer(question: str, result: Dict) -> str:
    rows = _extract_rows(result)
    if not rows:
        result_text = "No records returned."
    else:
        lines = []
        for row in rows:
            lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        result_text = "\n".join(lines)

    response = llm.invoke(
        SOC_ANSWER_PROMPT.format(question=question, result=result_text)
    )
    return response.content



def generate_soc_answer_node(state: SOCState) -> SOCState:
    print("\n===== SOC STEP 6: GENERATING ANSWER =====")

    result = state["execution_result"]

    # ── Workflow: combine all step results into one text block ────────────
    if isinstance(result, dict) and "workflow_steps" in result:
        text_parts = []
        any_rows_found = False

        for step_id, payload in result["workflow_steps"].items():
            desc = payload.get("description", step_id)
            step_result = payload.get("result", {})
            rows = _extract_rows(step_result)

            if not rows:
                rows_text = "No records returned."
            else:
                any_rows_found = True
                rows_text = "\n".join(
                    " | ".join(f"{k}: {v}" for k, v in row.items())
                    for row in rows
                )
            text_parts.append(f"SECTION — {desc}:\n{rows_text}")

        combined = "\n\n".join(text_parts)

        if not any_rows_found:
            state["answer"] = (
                "No records were returned for any part of this request "
                "with the applied filters."
            )
            return state

        response = llm.invoke(
            SOC_ANSWER_PROMPT.format(question=state["question"], result=combined)
        )
        state["answer"] = response.content
        return state

    # ── Single query ──────────────────────────────────────────────────────
    rows = _extract_rows(result)
    if not rows:
        state["answer"] = "No records returned for the applied filters."
        return state

    answer = generate_soc_answer(state["question"], result)
    print("Answer:\n", answer)
    state["answer"] = answer
    return state



def extract_soc_column_values(result: Dict, column_label: str) -> List:
    """
    Pull distinct non-null values of `column_label` from a Power BI
    executeQueries result.  Matches flexibly on column key suffix because
    Power BI prefixes returned column names with the table alias, e.g.
    "'Direct Reportee'[Function Label]" or "[Function Label]".
    Falls back to the first column if no suffix match is found.
    """
    rows = _extract_rows(result)
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

