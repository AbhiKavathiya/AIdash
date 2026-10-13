"""
Streamlit App - Self-Contained Module
Interactive web interface for the RAG system.
"""

import streamlit as st
from typing import List, Dict, Optional
import json
from datetime import datetime

# Import local modules
from retrieval import AzureSearchRetriever
from rag_agent import RAGAgent


# ========== CONFIGURATION - UPDATE THESE VALUES ==========
CONFIG = {
    "search_service_endpoint": "https://your-search-service.search.windows.net",
    "search_api_key": "your-search-api-key",
    "index_name": "policy-index",
    "openai_api_key": "your-openai-api-key",
    "openai_endpoint": "https://your-openai-resource.openai.azure.com/",
    "chat_deployment": "gpt-4",
    "embedding_model": "text-embedding-3-large"
}

APP_CONFIG = {
    "title": "Policy Document Assistant",
    "subtitle": "Ask questions about policies, FAQs, matrices, and guidelines",
    "icon": "📚"
}
# ========================================================


# ========== Page Configuration ==========
st.set_page_config(
    page_title="Policy Document Q&A",
    page_icon=APP_CONFIG["icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== Custom CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
    }
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        margin-right: 15%;
        border-left: 4px solid #667eea;
    }
    .source-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: white;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .source-header {
        font-weight: 600;
        color: #667eea;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .source-content {
        color: #555;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    .filter-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: #667eea;
        color: white;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ========== Session State Initialization ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "current_filters" not in st.session_state:
    st.session_state.current_filters = {
        "document_types": [],
        "document_names": []
    }

if "stats" not in st.session_state:
    st.session_state.stats = None


# ========== Initialize Agent ==========
@st.cache_resource
def initialize_agent():
    """Initialize the RAG agent"""
    try:
        # Check if configuration is updated
        if "your-" in CONFIG["search_service_endpoint"]:
            st.error("⚠️ Please update the configuration values in streamlit_app.py")
            st.stop()
        
        # Initialize retriever
        retriever = AzureSearchRetriever(
            search_endpoint=CONFIG["search_service_endpoint"],
            search_api_key=CONFIG["search_api_key"],
            openai_api_key=CONFIG["openai_api_key"],
            openai_endpoint=CONFIG["openai_endpoint"],
            embedding_model=CONFIG["embedding_model"]
        )
        
        # Initialize RAG agent
        agent = RAGAgent(
            retriever=retriever,
            openai_api_key=CONFIG["openai_api_key"],
            openai_endpoint=CONFIG["openai_endpoint"],
            chat_deployment=CONFIG["chat_deployment"],
            index_name=CONFIG["index_name"]
        )
        
        return agent
    
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


# ========== Helper Functions ==========
def get_conversation_history() -> List[Dict]:
    """Get conversation history in the required format"""
    history = []
    for msg in st.session_state.chat_history:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return history


def display_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None):
    """Display a chat message with optional sources"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div style="margin-bottom: 0.5rem; font-weight: 600;">{icon} {role.title()}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources if available
    if sources and role == "assistant":
        with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
            for idx, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">{idx}. {source['document_name']}</div>
                    <div style="color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;">
                        Type: {source['document_type']} | Chunks: {len(source['chunks'])}
                    </div>
                    <div class="source-content">
                        {source['chunks'][0]['content'][:300]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)


def display_filters():
    """Display active filters"""
    filters = st.session_state.current_filters
    
    if filters["document_types"] or filters["document_names"]:
        st.markdown("**Active Filters:**")
        
        filter_html = ""
        for doc_type in filters["document_types"]:
            filter_html += f'<span class="filter-chip">📑 {doc_type}</span>'
        
        for doc_name in filters["document_names"]:
            filter_html += f'<span class="filter-chip">📄 {doc_name}</span>'
        
        st.markdown(filter_html, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Filters"):
            st.session_state.current_filters = {
                "document_types": [],
                "document_names": []
            }
            st.rerun()


# ========== Main App ==========
def main():
    """Main application"""
    
    # Header
    st.markdown(f'<h1 class="main-header">{APP_CONFIG["icon"]} {APP_CONFIG["title"]}</h1>', 
                unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{APP_CONFIG["subtitle"]}</p>', 
                unsafe_allow_html=True)
    
    # Initialize agent
    if st.session_state.agent is None:
        with st.spinner("🔄 Initializing AI agent..."):
            st.session_state.agent = initialize_agent()
            
            if st.session_state.agent:
                st.session_state.stats = st.session_state.agent.get_index_stats()
                st.success("✅ Agent initialized successfully!")
            else:
                st.error("❌ Failed to initialize agent. Please check configuration.")
                st.stop()
    
    agent = st.session_state.agent
    
    # ========== Sidebar ==========
    with st.sidebar:
        st.header("📊 System Info")
        
        if st.session_state.stats:
            stats = st.session_state.stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('unique_documents', 0)}</div>
                    <div class="metric-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('total_chunks', 0)}</div>
                    <div class="metric-label">Chunks</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== Filters Section ==========
        st.header("🔍 Filters")
        
        # Document Type Filter
        st.subheader("Document Type")
        doc_types = agent.get_document_types()
        
        selected_types = st.multiselect(
            "Filter by document type:",
            options=doc_types,
            default=st.session_state.current_filters["document_types"]
        )
        st.session_state.current_filters["document_types"] = selected_types
        
        # Document Name Filter
        if st.checkbox("Filter by specific documents"):
            documents = agent.get_available_documents()
            doc_names = [doc["document_name"] for doc in documents]
            
            selected_docs = st.multiselect(
                "Choose documents:",
                doc_names,
                default=st.session_state.current_filters["document_names"]
            )
            st.session_state.current_filters["document_names"] = selected_docs
        else:
            st.session_state.current_filters["document_names"] = []
        
        st.markdown("---")
        
        # ========== Available Documents ==========
        st.header("📄 Available Documents")
        
        if st.session_state.stats and st.session_state.stats.get('documents'):
            for doc in st.session_state.stats['documents'][:10]:
                st.markdown(f"**{doc['document_name']}**")
                st.caption(f"_{doc['document_type']}_")
        
        st.markdown("---")
        
        # ========== Quick Actions ==========
        st.header("⚡ Quick Actions")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset Filters"):
            st.session_state.current_filters = {
                "document_types": [],
                "document_names": []
            }
            st.rerun()
    
    # ========== Main Content ==========
    
    # Display active filters
    display_filters()
    
    # Example Questions
    with st.expander("💡 Example Questions", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are the eligibility criteria?", key="ex1", use_container_width=True):
                st.session_state.example_question = "What are the eligibility criteria?"
            
            if st.button("What is the approval process?", key="ex2", use_container_width=True):
                st.session_state.example_question = "What is the approval process?"
        
        with col2:
            if st.button("Tell me about deviation matrix", key="ex3", use_container_width=True):
                st.session_state.example_question = "Tell me about the deviation matrix"
            
            if st.button("What are the documentation requirements?", key="ex4", use_container_width=True):
                st.session_state.example_question = "What are the documentation requirements?"
    
    # Chat History Display
    st.markdown("### 💬 Conversation")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(
                role=message["role"],
                content=message["content"],
                sources=message.get("sources")
            )
    
    # User Input
    st.markdown("---")
    
    # Check for example question
    user_input = st.session_state.get("example_question", "")
    if user_input:
        del st.session_state.example_question
    else:
        user_input = st.chat_input("Ask a question about policies, FAQs, or guidelines...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with chat_container:
            display_chat_message("user", user_input)
        
        # Get response from agent
        with st.spinner("🤔 Thinking..."):
            try:
                filters = st.session_state.current_filters
                
                response = agent.ask(
                    question=user_input,
                    document_types=filters["document_types"] if filters["document_types"] else None,
                    document_names=filters["document_names"] if filters["document_names"] else None,
                    chat_history=get_conversation_history()[:-1]  # Exclude current question
                )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "metadata": response.get("metadata", {})
                })
                
                # Display assistant message
                with chat_container:
                    display_chat_message(
                        role="assistant",
                        content=response["answer"],
                        sources=response.get("sources", [])
                    )
                
                # Show metadata
                if response.get("metadata"):
                    with st.expander("📈 Query Metadata", expanded=False):
                        metadata = response["metadata"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Retrieved Chunks", metadata.get("retrieved_count", 0))
                        with col2:
                            st.metric("Unique Sources", metadata.get("unique_sources", 0))
                        with col3:
                            st.metric("Answer Length", f"{metadata.get('answer_length', 0)} chars")
                        
                        if metadata.get("document_types_found"):
                            st.write("**Document Types Used:**")
                            st.write(", ".join(metadata["document_types_found"]))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try again.",
                    "sources": []
                })
        
        st.rerun()


if __name__ == "__main__":
    main()
