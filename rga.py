from typing import List, Dict, Optional
from openai import AzureOpenAI
import json
from datetime import datetime


class RAGAgent:
    """RAG agent that combines retrieval and generation"""
    
    def __init__(
        self,
        retriever,
        openai_api_key: str,
        openai_endpoint: str,
        chat_deployment: str = "gpt-4",
        index_name: str = "policy-index"
    ):
        """
        Initialize RAG agent
        
        Args:
            retriever: AzureSearchRetriever instance
            openai_api_key: Azure OpenAI API key
            openai_endpoint: Azure OpenAI endpoint
            chat_deployment: Chat model deployment name
            index_name: Azure Search index name
        """
        self.retriever = retriever
        self.index_name = index_name
        
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key,
            api_version="2024-02-01",
            azure_endpoint=openai_endpoint
        )
        self.chat_deployment = chat_deployment
        
        print(f"✓ RAG Agent initialized with {chat_deployment}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent"""
        return """You are a helpful AI assistant that answers questions based on policy documents, FAQs, and guidelines.

Your responsibilities:
1. Answer questions accurately based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
3. Cite specific documents and sections when providing answers
4. Be concise but thorough
5. If multiple documents provide relevant information, synthesize them coherently

Format your answers clearly with:
- Direct answer first
- Supporting details from the documents
- Document references (Document name, Section)
"""
    
    def _build_context(self, search_results: List) -> str:
        """Build context from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"""
Document {i}: {result.document_name}
Type: {result.document_type}
Section: {result.section}
Content:
{result.chunk_text}
---
""")
        
        return "\n".join(context_parts)
    
    def _group_results_by_document(self, search_results: List) -> List[Dict]:
        """Group search results by document"""
        documents = {}
        
        for result in search_results:
            doc_name = result.document_name
            if doc_name not in documents:
                documents[doc_name] = {
                    "document_name": doc_name,
                    "document_type": result.document_type,
                    "chunks": []
                }
            
            documents[doc_name]["chunks"].append({
                "section": result.section,
                "content": result.chunk_text,
                "score": result.score,
                "reranker_score": result.reranker_score
            })
        
        return list(documents.values())
    
    def ask(
        self,
        question: str,
        document_types: Optional[List[str]] = None,
        document_names: Optional[List[str]] = None,
        top_k: int = 5,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: User question
            document_types: Optional filter by document types
            document_names: Optional filter by document names
            top_k: Number of chunks to retrieve
            chat_history: Optional conversation history
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n💬 Question: {question}")
        
        # Build filter expression
        filter_parts = []
        if document_types:
            type_filters = " or ".join([f"document_type eq '{dt}'" for dt in document_types])
            filter_parts.append(f"({type_filters})")
        
        if document_names:
            name_filters = " or ".join([f"document_name eq '{dn}'" for dn in document_names])
            filter_parts.append(f"({name_filters})")
        
        filter_expression = " and ".join(filter_parts) if filter_parts else None
        
        # Retrieve relevant chunks
        search_results = self.retriever.hybrid_search(
            index_name=self.index_name,
            query=question,
            top_k=top_k,
            use_semantic=True,
            filter_expression=filter_expression
        )
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {
                    "retrieved_count": 0,
                    "unique_sources": 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Build context
        context = self._build_context(search_results)
        
        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history[-6:])  # Last 3 exchanges
        
        # Add current question with context
        user_message = f"""Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate answer
        print("🤔 Generating answer...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Prepare sources
            sources = self._group_results_by_document(search_results)
            
            # Metadata
            metadata = {
                "retrieved_count": len(search_results),
                "unique_sources": len(sources),
                "document_types_found": list(set([r.document_type for r in search_results])),
                "answer_length": len(answer),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✓ Answer generated ({len(answer)} chars)")
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": metadata
            }
        
        except Exception as e:
            print(f"✗ Error generating answer: {str(e)}")
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "sources": self._group_results_by_document(search_results),
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def multi_turn_conversation(
        self,
        question: str,
        chat_history: List[Dict],
        document_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Handle multi-turn conversation with context awareness
        
        Args:
            question: Current question
            chat_history: Previous conversation history
            document_types: Optional filter by document types
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Check if this is a follow-up question
        if chat_history and len(chat_history) >= 2:
            # Reformulate question with context
            reformulation_messages = [
                {
                    "role": "system",
                    "content": "You reformulate follow-up questions to be standalone by incorporating context from the conversation history."
                },
                {
                    "role": "user",
                    "content": f"""Conversation history:
{json.dumps(chat_history[-4:], indent=2)}

Current follow-up question: {question}

Reformulate this question to be standalone and clear, incorporating any necessary context from the conversation history."""
                }
            ]
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.chat_deployment,
                    messages=reformulation_messages,
                    temperature=0.2,
                    max_tokens=150
                )
                
                reformulated_question = response.choices[0].message.content
                print(f"🔄 Reformulated: {reformulated_question}")
                
                # Use reformulated question for retrieval
                return self.ask(
                    question=reformulated_question,
                    document_types=document_types,
                    top_k=top_k,
                    chat_history=chat_history
                )
            
            except Exception as e:
                print(f"⚠️ Reformulation failed, using original question: {str(e)}")
                return self.ask(
                    question=question,
                    document_types=document_types,
                    top_k=top_k,
                    chat_history=chat_history
                )
        else:
            # First question or not enough history
            return self.ask(
                question=question,
                document_types=document_types,
                top_k=top_k,
                chat_history=chat_history
            )
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        return self.retriever.get_index_stats(self.index_name)
    
    def get_document_types(self) -> List[str]:
        """Get available document types"""
        return self.retriever.get_document_types(self.index_name)
    
    def get_available_documents(self) -> List[Dict]:
        """Get list of available documents"""
        return self.retriever.get_available_documents(self.index_name)


def main():
    """Example usage"""
    from retrieval import AzureSearchRetriever
    
    # Configuration
    SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
    SEARCH_API_KEY = "your-search-api-key"
    OPENAI_API_KEY = "your-openai-api-key"
    OPENAI_ENDPOINT = "https://your-openai-resource.openai.azure.com/"
    CHAT_DEPLOYMENT = "gpt-4"
    INDEX_NAME = "policy-index"
    
    # Initialize retriever
    retriever = AzureSearchRetriever(
        search_endpoint=SEARCH_ENDPOINT,
        search_api_key=SEARCH_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        openai_endpoint=OPENAI_ENDPOINT
    )
    
    # Initialize RAG agent
    agent = RAGAgent(
        retriever=retriever,
        openai_api_key=OPENAI_API_KEY,
        openai_endpoint=OPENAI_ENDPOINT,
        chat_deployment=CHAT_DEPLOYMENT,
        index_name=INDEX_NAME
    )
    
    # Example 1: Simple question
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Question")
    print("="*80)
    
    response = agent.ask("What are the eligibility criteria for the prime product?")
    print(f"\n🤖 Answer:\n{response['answer']}")
    print(f"\n📚 Sources: {response['metadata']['unique_sources']} documents")
    for source in response['sources']:
        print(f"  - {source['document_name']} ({len(source['chunks'])} chunks)")
    
    # Example 2: Filtered question
    print("\n" + "="*80)
    print("EXAMPLE 2: Filtered by Document Type")
    print("="*80)
    
    response = agent.ask(
        question="What is the approval process?",
        document_types=["policy", "guideline"]
    )
    print(f"\n🤖 Answer:\n{response['answer']}")
    
    # Example 3: Multi-turn conversation
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-turn Conversation")
    print("="*80)
    
    chat_history = [
        {"role": "user", "content": "What are the credit approval requirements?"},
        {"role": "assistant", "content": "The credit approval requirements include..."}
    ]
    
    response = agent.multi_turn_conversation(
        question="What about the deviation matrix?",
        chat_history=chat_history
    )
    print(f"\n🤖 Answer:\n{response['answer']}")
    
    # Example 4: Get stats
    print("\n" + "="*80)
    print("EXAMPLE 4: Index Statistics")
    print("="*80)
    
    stats = agent.get_index_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique documents: {stats['unique_documents']}")
    print(f"Document types: {stats['document_types']}")


if __name__ == "__main__":
    main()
