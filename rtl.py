"""
Azure AI Search Retrieval - Self-Contained Module
Handles document retrieval with hybrid search capabilities.
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from typing import List, Dict, Optional
import json
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Structured search result with metadata"""
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    section: str
    chunk_text: str
    token_count: int
    page_number: int
    score: float
    reranker_score: Optional[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "section": self.section,
            "chunk_text": self.chunk_text,
            "token_count": self.token_count,
            "page_number": self.page_number,
            "score": self.score,
            "reranker_score": self.reranker_score
        }


class AzureSearchRetriever:
    """Handles document retrieval from Azure AI Search"""
    
    def __init__(
        self,
        search_endpoint: str,
        search_api_key: str,
        openai_api_key: str,
        openai_endpoint: str,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072
    ):
        """Initialize the retriever with Azure credentials"""
        self.search_endpoint = search_endpoint
        self.search_credential = AzureKeyCredential(search_api_key)
        
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key,
            api_version="2024-02-01",
            azure_endpoint=openai_endpoint
        )
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        
        print(f"✓ Retriever initialized with {embedding_model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"✗ Error generating embedding: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        use_semantic: bool = True,
        filter_expression: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search (vector + keyword) with optional semantic ranking
        
        Args:
            index_name: Name of the search index
            query: Search query
            top_k: Number of results to return
            use_semantic: Whether to use semantic ranking
            filter_expression: Optional OData filter expression
            
        Returns:
            List of SearchResult objects
        """
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        print(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_vector = self.generate_embedding(query)
        
        # Build search parameters
        search_params = {
            "search_text": query,
            "vector_queries": [VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k * 2,
                fields="content_vector"
            )],
            "select": [
                "id", "document_id", "document_name", "document_type",
                "chunk_text", "section", "token_count", "page_number"
            ],
            "top": top_k
        }
        
        if filter_expression:
            search_params["filter"] = filter_expression
        
        if use_semantic:
            search_params["query_type"] = "semantic"
            search_params["semantic_configuration_name"] = "semantic-config"
        
        # Execute search
        results = search_client.search(**search_params)
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.get("id"),
                document_id=result.get("document_id", ""),
                document_name=result.get("document_name"),
                document_type=result.get("document_type"),
                section=result.get("section", ""),
                chunk_text=result.get("chunk_text"),
                token_count=result.get("token_count", 0),
                page_number=result.get("page_number", 0),
                score=result.get("@search.score", 0.0),
                reranker_score=result.get("@search.reranker_score")
            ))
        
        print(f"✓ Found {len(search_results)} results")
        return search_results
    
    def vector_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform pure vector search (semantic similarity only)"""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        print(f"🔍 Vector search for: '{query}'")
        query_vector = self.generate_embedding(query)
        
        search_params = {
            "search_text": None,
            "vector_queries": [VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )],
            "select": [
                "id", "document_id", "document_name", "document_type",
                "chunk_text", "section", "token_count", "page_number"
            ],
            "top": top_k
        }
        
        if filter_expression:
            search_params["filter"] = filter_expression
        
        results = search_client.search(**search_params)
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.get("id"),
                document_id=result.get("document_id", ""),
                document_name=result.get("document_name"),
                document_type=result.get("document_type"),
                section=result.get("section", ""),
                chunk_text=result.get("chunk_text"),
                token_count=result.get("token_count", 0),
                page_number=result.get("page_number", 0),
                score=result.get("@search.score", 0.0),
                reranker_score=None
            ))
        
        print(f"✓ Found {len(search_results)} results")
        return search_results
    
    def keyword_search(
        self,
        index_name: str,
        query: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform pure keyword search (BM25 full-text search)"""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        print(f"🔍 Keyword search for: '{query}'")
        
        search_params = {
            "search_text": query,
            "select": [
                "id", "document_id", "document_name", "document_type",
                "chunk_text", "section", "token_count", "page_number"
            ],
            "top": top_k
        }
        
        if filter_expression:
            search_params["filter"] = filter_expression
        
        results = search_client.search(**search_params)
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.get("id"),
                document_id=result.get("document_id", ""),
                document_name=result.get("document_name"),
                document_type=result.get("document_type"),
                section=result.get("section", ""),
                chunk_text=result.get("chunk_text"),
                token_count=result.get("token_count", 0),
                page_number=result.get("page_number", 0),
                score=result.get("@search.score", 0.0),
                reranker_score=None
            ))
        
        print(f"✓ Found {len(search_results)} results")
        return search_results
    
    def get_index_stats(self, index_name: str) -> Dict:
        """Get statistics about the index"""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        # Get total count
        result = search_client.search(search_text="*", include_total_count=True, top=0)
        total_chunks = result.get_count()
        
        # Get document types
        results = search_client.search(
            search_text="*",
            select=["document_id", "document_type", "document_name"],
            top=10000
        )
        
        document_ids = set()
        document_types = {}
        documents = []
        
        for r in results:
            doc_id = r.get("document_id")
            doc_type = r.get("document_type")
            doc_name = r.get("document_name")
            
            if doc_id:
                document_ids.add(doc_id)
                if doc_name and doc_name not in [d["document_name"] for d in documents]:
                    documents.append({
                        "document_name": doc_name,
                        "document_type": doc_type
                    })
            if doc_type:
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "unique_documents": len(document_ids),
            "document_types": document_types,
            "documents": documents[:20]  # First 20 documents
        }
    
    def get_document_types(self, index_name: str) -> List[str]:
        """Get list of unique document types in the index"""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        results = search_client.search(
            search_text="*",
            select=["document_type"],
            top=10000
        )
        
        doc_types = set()
        for r in results:
            doc_type = r.get("document_type")
            if doc_type:
                doc_types.add(doc_type)
        
        return sorted(list(doc_types))
    
    def get_available_documents(self, index_name: str) -> List[Dict]:
        """Get list of all documents in the index"""
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        results = search_client.search(
            search_text="*",
            select=["document_id", "document_name", "document_type"],
            top=10000
        )
        
        seen = set()
        documents = []
        
        for r in results:
            doc_name = r.get("document_name")
            if doc_name and doc_name not in seen:
                documents.append({
                    "document_id": r.get("document_id"),
                    "document_name": doc_name,
                    "document_type": r.get("document_type")
                })
                seen.add(doc_name)
        
        return documents


def print_search_results(results: List[SearchResult], show_full_text: bool = False):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS ({len(results)} found)")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 📄 {result.document_name} ({result.document_type})")
        print(f"   📂 Section: {result.section}")
        print(f"   📏 Tokens: {result.token_count} | Page: {result.page_number}")
        print(f"   ⭐ Score: {result.score:.4f}", end="")
        if result.reranker_score:
            print(f" | 🎯 Semantic: {result.reranker_score:.4f}")
        else:
            print()
        
        print(f"   📝 Content:")
        if show_full_text:
            print(f"      {result.chunk_text}")
        else:
            preview = result.chunk_text[:300] + "..." if len(result.chunk_text) > 300 else result.chunk_text
            print(f"      {preview}")


def main():
    """Example usage"""
    
    # Configuration
    SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
    SEARCH_API_KEY = "your-search-api-key"
    OPENAI_API_KEY = "your-openai-api-key"
    OPENAI_ENDPOINT = "https://your-openai-resource.openai.azure.com/"
    INDEX_NAME = "policy-index"
    
    # Initialize retriever
    retriever = AzureSearchRetriever(
        search_endpoint=SEARCH_ENDPOINT,
        search_api_key=SEARCH_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        openai_endpoint=OPENAI_ENDPOINT
    )
    
    # Example 1: Hybrid search
    print("\n" + "="*80)
    print("HYBRID SEARCH")
    print("="*80)
    results = retriever.hybrid_search(
        index_name=INDEX_NAME,
        query="What are the eligibility criteria?",
        top_k=5,
        use_semantic=True
    )
    print_search_results(results)
    
    # Example 2: Filtered search
    print("\n" + "="*80)
    print("FILTERED SEARCH")
    print("="*80)
    results = retriever.hybrid_search(
        index_name=INDEX_NAME,
        query="approval process",
        top_k=3,
        use_semantic=True,
        filter_expression="document_type eq 'policy'"
    )
    print_search_results(results)
    
    # Example 3: Get index stats
    print("\n" + "="*80)
    print("INDEX STATISTICS")
    print("="*80)
    stats = retriever.get_index_stats(INDEX_NAME)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique documents: {stats['unique_documents']}")
    print(f"Document types: {stats['document_types']}")


if __name__ == "__main__":
    main()
