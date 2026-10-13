"""
Azure AI Search Indexing - Self-Contained Module
Handles document processing, chunking, and indexing with hybrid search capabilities.
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from openai import AzureOpenAI
import tiktoken
from typing import List, Dict, Optional
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Try importing document processing libraries
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentType(Enum):
    """Document types with their specific characteristics"""
    POLICY = "policy"
    FAQ = "faq"
    MATRIX = "matrix"
    GUIDELINE = "guideline"
    GENERAL = "general"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""
    target_tokens: int
    max_tokens: int
    overlap_tokens: int


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    section: str
    chunk_text: str
    page_number: int
    token_count: int
    metadata: Dict


class DocumentChunker:
    """Handles document chunking with smart strategies"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        # Reduced max_tokens to stay well under 8192 embedding limit
        self.configs = {
            DocumentType.POLICY: ChunkingConfig(800, 1500, 100),
            DocumentType.FAQ: ChunkingConfig(600, 1000, 50),
            DocumentType.MATRIX: ChunkingConfig(700, 1200, 80),
            DocumentType.GUIDELINE: ChunkingConfig(800, 1500, 100),
            DocumentType.GENERAL: ChunkingConfig(800, 1500, 100),
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def classify_document(self, content: str, filename: str) -> DocumentType:
        """Classify document based on content and filename"""
        content_lower = content[:2000].lower()
        filename_lower = filename.lower()
        
        if 'faq' in filename_lower or 'question' in content_lower:
            return DocumentType.FAQ
        elif 'matrix' in filename_lower or 'approval' in content_lower:
            return DocumentType.MATRIX
        elif 'policy' in filename_lower or 'policy' in content_lower:
            return DocumentType.POLICY
        elif 'guideline' in filename_lower or 'procedure' in content_lower:
            return DocumentType.GUIDELINE
        else:
            return DocumentType.GENERAL
    
    def chunk_document(
        self,
        content: str,
        document_name: str,
        document_id: str,
        doc_type: DocumentType
    ) -> List[DocumentChunk]:
        """Chunk document into smaller pieces"""
        config = self.configs[doc_type]
        chunks = []
        
        # Safety check: ensure max_tokens doesn't exceed embedding limit
        EMBEDDING_MAX_TOKENS = 8000  # Safe limit below 8192
        if config.max_tokens > EMBEDDING_MAX_TOKENS:
            config.max_tokens = EMBEDDING_MAX_TOKENS
            print(f"  ⚠️ Adjusted max_tokens to {EMBEDDING_MAX_TOKENS} for embedding compatibility")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_section = "Introduction"
        current_text = []
        current_tokens = 0
        chunk_counter = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # Check if paragraph is a section header
            if len(para) < 100 and (para.isupper() or re.match(r'^\d+\.', para)):
                if current_text:
                    chunk_text = '\n\n'.join(current_text).strip()
                    if current_tokens > 50:
                        chunks.append(self._create_chunk(
                            chunk_counter, document_id, document_name,
                            doc_type.value, current_section, chunk_text,
                            current_tokens
                        ))
                        chunk_counter += 1
                
                current_section = para
                current_text = []
                current_tokens = 0
            else:
                if current_tokens + para_tokens > config.max_tokens and current_text:
                    chunk_text = '\n\n'.join(current_text).strip()
                    chunks.append(self._create_chunk(
                        chunk_counter, document_id, document_name,
                        doc_type.value, current_section, chunk_text,
                        current_tokens
                    ))
                    chunk_counter += 1
                    
                    # Add overlap
                    if config.overlap_tokens > 0 and current_text:
                        overlap_text = current_text[-1]
                        current_text = [overlap_text, para]
                        current_tokens = self.count_tokens(overlap_text) + para_tokens
                    else:
                        current_text = [para]
                        current_tokens = para_tokens
                else:
                    current_text.append(para)
                    current_tokens += para_tokens
        
        # Final chunk
        if current_text and current_tokens > 50:
            chunk_text = '\n\n'.join(current_text).strip()
            chunks.append(self._create_chunk(
                chunk_counter, document_id, document_name,
                doc_type.value, current_section, chunk_text,
                current_tokens
            ))
        
        return chunks
    
    def _create_chunk(
        self, chunk_num: int, doc_id: str, doc_name: str,
        doc_type: str, section: str, text: str, token_count: int
    ) -> DocumentChunk:
        """Create a DocumentChunk object"""
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_num:04d}",
            document_id=doc_id,
            document_name=doc_name,
            document_type=doc_type,
            section=section,
            chunk_text=text,
            page_number=1,
            token_count=token_count,
            metadata={"created_at": datetime.now().isoformat()}
        )


class AzureSearchIndexer:
    """Handles Azure AI Search index creation and document indexing"""
    
    def __init__(
        self,
        search_endpoint: str,
        search_api_key: str,
        openai_api_key: str,
        openai_endpoint: str,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072
    ):
        """Initialize the indexer with Azure credentials"""
        self.search_endpoint = search_endpoint
        self.search_credential = AzureKeyCredential(search_api_key)
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.search_credential
        )
        
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key,
            api_version="2024-02-01",
            azure_endpoint=openai_endpoint
        )
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chunker = DocumentChunker()
        
        print(f"✓ Indexer initialized with {embedding_model}")
    
    def create_index(self, index_name: str) -> bool:
        """Create search index with hybrid search configuration"""
        print(f"\n🔧 Creating index: {index_name}")
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="document_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="chunk_text", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_profile_name="vector-profile"
            ),
            SimpleField(name="metadata", type=SearchFieldDataType.String)
        ]
        
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-algorithm")
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-algorithm"
                )
            ]
        )
        
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="document_name"),
                content_fields=[
                    SemanticField(field_name="chunk_text"),
                    SemanticField(field_name="section")
                ],
                keywords_fields=[
                    SemanticField(field_name="document_type")
                ]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            self.index_client.create_or_update_index(index)
            print(f"✓ Index created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating index: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        try:
            # Truncate text if too long (max 8191 tokens for text-embedding-3-large)
            tokens = self.chunker.encoder.encode(text)
            if len(tokens) > 8191:
                print(f"  ⚠️ Truncating text from {len(tokens)} to 8191 tokens")
                tokens = tokens[:8191]
                text = self.chunker.encoder.decode(tokens)
            
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"✗ Error generating embedding: {str(e)}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.txt' or ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif ext == '.pdf' and PdfReader:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                return text
            
            elif ext in ['.docx', '.doc'] and DocxDocument:
                doc = DocxDocument(file_path)
                return '\n\n'.join([para.text for para in doc.paragraphs])
            
            else:
                print(f"⚠️ Unsupported file format: {ext}")
                return ""
        
        except Exception as e:
            print(f"✗ Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def index_document(self, index_name: str, file_path: str) -> bool:
        """Index a single document"""
        print(f"\n📄 Processing: {os.path.basename(file_path)}")
        
        # Extract text
        content = self.extract_text_from_file(file_path)
        if not content:
            print(f"✗ No content extracted from {file_path}")
            return False
        
        # Generate document ID
        document_name = os.path.basename(file_path)
        document_id = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(document_name)[0])
        
        # Classify document
        doc_type = self.chunker.classify_document(content, document_name)
        print(f"  Type: {doc_type.value}")
        
        # Chunk document
        chunks = self.chunker.chunk_document(content, document_name, document_id, doc_type)
        print(f"  Created {len(chunks)} chunks")
        
        # Prepare documents for indexing
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=self.search_credential
        )
        
        documents = []
        for chunk in chunks:
            # Generate embedding
            embedding = self.generate_embedding(chunk.chunk_text)
            
            doc = {
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "document_type": chunk.document_type,
                "section": chunk.section,
                "chunk_text": chunk.chunk_text,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
                "content_vector": embedding,
                "metadata": json.dumps(chunk.metadata)
            }
            documents.append(doc)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                print(f"  ✓ Uploaded batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            except Exception as e:
                print(f"  ✗ Error uploading batch: {str(e)}")
                return False
        
        print(f"✓ Successfully indexed {document_name}")
        return True
    
    def index_directory(self, index_name: str, directory_path: str) -> Dict:
        """Index all documents in a directory"""
        print(f"\n📁 Indexing directory: {directory_path}")
        
        supported_formats = ['.txt', '.md', '.pdf', '.docx', '.doc']
        files = []
        
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_formats):
                    files.append(os.path.join(root, filename))
        
        print(f"Found {len(files)} documents to process")
        
        results = {
            "total": len(files),
            "successful": 0,
            "failed": 0
        }
        
        for file_path in files:
            if self.index_document(index_name, file_path):
                results["successful"] += 1
            else:
                results["failed"] += 1
        
        print(f"\n{'='*80}")
        print(f"INDEXING COMPLETE")
        print(f"{'='*80}")
        print(f"Total: {results['total']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        
        return results


def main():
    """Example usage"""
    
    # Configuration
    SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
    SEARCH_API_KEY = "your-search-api-key"
    OPENAI_API_KEY = "your-openai-api-key"
    OPENAI_ENDPOINT = "https://your-openai-resource.openai.azure.com/"
    INDEX_NAME = "policy-index"
    DOCUMENTS_PATH = "./documents"
    
    # Initialize indexer
    indexer = AzureSearchIndexer(
        search_endpoint=SEARCH_ENDPOINT,
        search_api_key=SEARCH_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        openai_endpoint=OPENAI_ENDPOINT
    )
    
    # Create index
    indexer.create_index(INDEX_NAME)
    
    # Index documents
    indexer.index_directory(INDEX_NAME, DOCUMENTS_PATH)


if __name__ == "__main__":
    main()
