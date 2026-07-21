import json
from typing import List, Dict, Optional, TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Import agents
import diversity_v7
import ijp_v5
import soc_v6
import v12


# ============================================================
# STATE
# ============================================================

class SubQuery(TypedDict):
    query_text: str
    target_agent: str  # "diversity" | "ijp" | "soc" | "hr"
    response: str
    error: Optional[str]


class OrchestratorState(TypedDict):
    user_query: str
    session_id: str
    sub_queries: List[SubQuery]
    final_answer: str


# ============================================================
# LLM SETUP
# ============================================================

llm = AzureChatOpenAI(
    azure_endpoint=diversity_v7.AZURE_OPENAI_ENDPOINT,
    azure_deployment=diversity_v7.AZURE_OPENAI_DEPLOYMENT,
    api_version=diversity_v7.API_VERSION,
    api_key=diversity_v7.AZURE_OPENAI_API_KEY,
)


# ============================================================
# ROUTING PROMPT (IMPROVED)
# ============================================================

ROUTING_SYSTEM = """You are a query router for an HR analytics platform with 4 specialized dashboards.

AVAILABLE DASHBOARDS:
─────────────────────

1. DIVERSITY & INCLUSION (diversity)
   - Diversity %, gender breakdowns
   - Attrition, attrition %, exit analysis
   - Employment type analysis
   - Voluntary vs involuntary exits

2. INTERNAL JOB POSTING (ijp)
   - IJP eligibility (same-level, one-level-up, promotion)
   - Career mobility metrics
   - Mid-year promotion eligibility

3. SPAN OF CONTROL (soc)
   - Manager span of control
   - Direct reports, layers
   - Management hierarchy

4. HR ANALYTICS (hr)
   - General headcount, FTE
   - High performers, ratings
   - Business unit, function, grade analysis

ROUTING RULES:
──────────────

**CRITICAL**: For multi-metric queries, split into sub-queries BUT preserve ALL context:

1. If query asks for MULTIPLE metrics from DIFFERENT dashboards:
   - Split into separate sub-queries (one per dashboard)
   - Copy ALL filters to EVERY sub-query
   - Copy ALL grouping dimensions to EVERY sub-query  
   - Copy ALL time periods to EVERY sub-query
   
2. If query asks for multiple metrics from SAME dashboard:
   - Keep as ONE sub-query with all metrics
   
3. Each sub-query must be FULLY STANDALONE - someone reading only that sub-query should understand exactly what data to get

Return JSON array:
[
  {{
    "query_text": "complete question with all filters and groupings",
    "target_agent": "diversity|ijp|soc|hr"
  }}
]

EXAMPLES:
─────────

Input: "Show headcount and diversity % by business unit for 2025"
Output: [
  {{"query_text": "Show headcount by business unit for 2025", "target_agent": "hr"}},
  {{"query_text": "Show diversity % by business unit for 2025", "target_agent": "diversity"}}
]

Input: "Show diversity % and attrition % by business unit for 2025"
Output: [
  {{"query_text": "Show diversity % and attrition % by business unit for 2025", "target_agent": "diversity"}}
]

Input: "Show IJP eligible count and attrition rate for full-time employees by grade"
Output: [
  {{"query_text": "Show IJP eligible count for full-time employees by grade", "target_agent": "ijp"}},
  {{"query_text": "Show attrition rate for full-time employees by grade", "target_agent": "diversity"}}
]

Input: "Top 5 business units by headcount and show their diversity %"
Output: [
  {{"query_text": "Top 5 business units by headcount", "target_agent": "hr"}},
  {{"query_text": "Show diversity % for the top 5 business units by headcount", "target_agent": "diversity"}}
]

**REMEMBER**: Each sub-query must have ALL filters, ALL groupings, ALL time periods from original query!

NO EXPLANATION. ONLY JSON.
"""


# ============================================================
# AGENT MAPPING (Direct module references)
# ============================================================

AGENT_MODULES = {
    "diversity": diversity_v7,
    "ijp": ijp_v5,
    "soc": soc_v6,
    "hr": v12,
}

AGENT_NAMES = {
    "diversity": "Diversity & Inclusion Dashboard",
    "ijp": "Internal Job Posting (IJP) Dashboard",
    "soc": "Span of Control (SOC) Dashboard",
    "hr": "HR Analytics Dashboard",
}




# ============================================================
# NODE 1: ROUTE QUERIES
# ============================================================

def route_queries_node(state: OrchestratorState) -> OrchestratorState:
    """Route query to appropriate agent(s) using LLM."""
    query = state["user_query"]
    
    print(f"\n📍 Routing query...")
    
    resp = llm.invoke([
        SystemMessage(content=ROUTING_SYSTEM),
        HumanMessage(content=query),
    ])
    
    raw = resp.content.strip().strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()
    
    print(f"🔍 LLM Routing Response:\n{raw[:400]}...\n")
    
    try:
        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError("Invalid routing response")
    except Exception as e:
        # Fallback: route to hr
        print(f"⚠️ Routing parse failed ({e}), defaulting to HR agent")
        sub_queries = [{
            "query_text": query,
            "target_agent": "hr",
            "response": "",
            "error": None
        }]
    
    # Add response/error fields
    for sq in sub_queries:
        sq["response"] = ""
        sq["error"] = None
    
    print(f"🔀 ROUTED to {len(sub_queries)} sub-query(ies):")
    for i, sq in enumerate(sub_queries, 1):
        print(f"   {i}. [{sq['target_agent'].upper()}] {sq['query_text']}")
    
    return {**state, "sub_queries": sub_queries}


# ============================================================
# NODE 2: EXECUTE QUERIES
# ============================================================

def execute_queries_node(state: OrchestratorState) -> OrchestratorState:
    """Execute each sub-query against its target agent."""
    sub_queries = state["sub_queries"]
    session_id = state.get("session_id", "default-session")
    
    for i, sq in enumerate(sub_queries, 1):
        agent_key = sq["target_agent"]
        query_text = sq["query_text"]
        
        if agent_key not in AGENT_MODULES:
            sq["error"] = f"Unknown agent: {agent_key}"
            sq["response"] = f"Error: Dashboard '{agent_key}' not found"
            print(f"\n❌ Sub-query {i}: Unknown agent '{agent_key}'")
            continue
        
        agent_module = AGENT_MODULES[agent_key]
        agent_name = AGENT_NAMES[agent_key]
        
        print(f"\n{'─' * 70}")
        print(f"📊 Sub-query {i}/{len(sub_queries)}: {agent_name}")
        print(f"   Query: {query_text}")
        print(f"{'─' * 70}")
        
        try:
            # Call agent's ask function
            response = agent_module.ask(query_text, session_id)
            sq["response"] = response
            print(f"\n✅ Success!")
            print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
        except Exception as e:
            error_msg = str(e)
            sq["error"] = error_msg
            sq["response"] = f"Error from {agent_name}: {error_msg}"
            print(f"\n❌ Failed: {error_msg[:200]}...")
            
            # Print stack trace for debugging
            import traceback
            print(f"\n🔍 Stack trace:")
            traceback.print_exc()
    
    return {**state, "sub_queries": sub_queries}


# ============================================================
# NODE 3: MERGE RESPONSES
# ============================================================

MERGE_SYSTEM = """You are an HR analytics assistant.

The user asked a question. The system split it into sub-queries and got responses from specialized dashboards.

Your job:
1. Combine responses into ONE coherent answer
2. If multiple sub-queries, present results clearly (use sections/tables)
3. If only one sub-query, present its answer directly
4. If any errors, acknowledge them but present what worked
5. Be concise and business-readable

DO NOT explain the routing or which dashboards were used - just present the final answer.
"""


def merge_responses_node(state: OrchestratorState) -> OrchestratorState:
    """Merge all sub-query responses into final answer."""
    sub_queries = state["sub_queries"]
    
    print(f"\n💬 Merging {len(sub_queries)} response(s)...")
    
    # Single query - return directly
    if len(sub_queries) == 1:
        sq = sub_queries[0]
        final = sq["response"]
        return {**state, "final_answer": final}
    
    # Multiple queries - merge intelligently
    context_parts = []
    for i, sq in enumerate(sub_queries, 1):
        agent_name = AGENT_NAMES.get(sq["target_agent"], sq["target_agent"])
        context_parts.append(
            f"[Sub-Query {i}]\n"
            f"Question: {sq['query_text']}\n"
            f"Dashboard: {agent_name}\n"
            f"Response:\n{sq['response']}\n"
        )
    
    context = "\n".join(context_parts)
    
    resp = llm.invoke([
        SystemMessage(content=MERGE_SYSTEM),
        HumanMessage(content=(
            f"Original user question: {state['user_query']}\n\n"
            f"Sub-query responses:\n\n{context}"
        )),
    ])
    
    final = resp.content.strip()
    print(f"✅ Merged response generated")
    
    return {**state, "final_answer": final}


# ============================================================
# BUILD GRAPH
# ============================================================

def build_orchestrator_graph():
    """Build the orchestration graph with routing."""
    graph = StateGraph(OrchestratorState)
    
    # Add nodes
    graph.add_node("route_queries", route_queries_node)
    graph.add_node("execute_queries", execute_queries_node)
    graph.add_node("merge_responses", merge_responses_node)
    
    # Define edges
    graph.set_entry_point("route_queries")
    graph.add_edge("route_queries", "execute_queries")
    graph.add_edge("execute_queries", "merge_responses")
    graph.add_edge("merge_responses", END)
    
    return graph.compile()


# ============================================================
# PUBLIC API
# ============================================================

def ask(query: str, session_id: str) -> str:
    """
    Ask a question about HR dashboards. Routes and merges intelligently.
    
    Args:
        query: User question (may target one or multiple dashboards)
        session_id: Session identifier for tracking conversation context
    
    Returns:
        Unified answer combining responses from all relevant dashboards
    """
    print("\n" + "=" * 80)
    print(f"QUESTION: {query}")
    print("=" * 80)
    
    orchestrator = build_orchestrator_graph()
    
    initial_state: OrchestratorState = {
        "user_query": query,
        "session_id": session_id,
        "sub_queries": [],
        "final_answer": "",
    }
    
    result = orchestrator.invoke(initial_state)
    return result["final_answer"]


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import uuid
    
    WIDTH = 80
    print("\n" + "═" * WIDTH)
    print("  🤖 Multi-Agent Dashboard Orchestrator")
    print("  Routes via prompts, executes, and merges")
    print("  Type 'exit' to quit")
    print("═" * WIDTH + "\n")
    
    session_id = str(uuid.uuid4())
    print(f"📋 Session ID: {session_id}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        
        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        
        answer = ask(user_input, session_id)
        
        print("\n" + "─" * WIDTH)
        print(f"🤖 Agent:\n{answer}")
        print("─" * WIDTH + "\n")
