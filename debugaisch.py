from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import json

# Configuration - UPDATE THESE
CONFIG = {
    "search_service_endpoint": "https://YOUR_SEARCH_SERVICE.search.windows.net",
    "search_api_key": "YOUR_SEARCH_API_KEY",
    "index_name": "policy-index"
}

def check_index():
    """Check index configuration and content"""
    
    if "YOUR_" in CONFIG["search_service_endpoint"]:
        print("❌ Please update CONFIG with your credentials")
        return
    
    print("\n" + "="*70)
    print("INDEX DIAGNOSTICS")
    print("="*70)
    
    try:
        # Connect to index
        index_client = SearchIndexClient(
            endpoint=CONFIG["search_service_endpoint"],
            credential=AzureKeyCredential(CONFIG["search_api_key"])
        )
        
        search_client = SearchClient(
            endpoint=CONFIG["search_service_endpoint"],
            index_name=CONFIG["index_name"],
            credential=AzureKeyCredential(CONFIG["search_api_key"])
        )
        
        # Step 1: Check if index exists
        print("\n1. Checking if index exists...")
        try:
            index = index_client.get_index(CONFIG["index_name"])
            print(f"   ✓ Index '{CONFIG['index_name']}' exists")
        except Exception as e:
            print(f"   ✗ Index not found: {str(e)}")
            print("\n   ⚠️  You need to create the index first!")
            print("   Run: python azure_ai_search_index.py")
            return
        
        # Step 2: Check index fields
        print("\n2. Index Fields:")
        print("   " + "-"*60)
        for field in index.fields:
            searchable = "🔍" if field.searchable else "  "
            filterable = "🔖" if field.filterable else "  "
            retrievable = "📥" if field.retrievable else "  "
            vector = "🎯" if field.type == "Collection(Edm.Single)" else "  "
            print(f"   {searchable}{filterable}{retrievable}{vector} {field.name:30s} ({field.type})")
        
        print("\n   Legend: 🔍=Searchable 🔖=Filterable 📥=Retrievable 🎯=Vector")
        
        # Step 3: Count documents
        print("\n3. Document Count:")
        try:
            results = search_client.search(
                search_text="*",
                include_total_count=True,
                top=0
            )
            count = results.get_count() if hasattr(results, 'get_count') else 0
            print(f"   Total documents in index: {count}")
            
            if count == 0:
                print("\n   ⚠️  INDEX IS EMPTY!")
                print("   You need to upload and index documents first:")
                print("   1. python extract_all_documents.py")
                print("   2. python ingest_all_documents.py")
                return
            
        except Exception as e:
            print(f"   Error counting documents: {str(e)}")
        
        # Step 4: Get sample documents
        print("\n4. Sample Documents:")
        try:
            results = search_client.search(
                search_text="*",
                top=3
            )
            
            sample_count = 0
            for result in results:
                sample_count += 1
                print(f"\n   Document {sample_count}:")
                
                # Show all fields
                for key, value in result.items():
                    if key.startswith("@"):
                        continue
                    if key == "content_vector":
                        print(f"      {key}: [vector with {len(value) if value else 0} dimensions]")
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"      {key}: {value[:100]}...")
                    else:
                        print(f"      {key}: {value}")
            
            if sample_count == 0:
                print("   No documents found!")
                
        except Exception as e:
            print(f"   Error retrieving samples: {str(e)}")
        
        # Step 5: Test simple search
        print("\n5. Test Simple Search:")
        try:
            test_query = "document"
            results = search_client.search(
                search_text=test_query,
                top=3
            )
            
            result_count = 0
            for result in results:
                result_count += 1
                doc_name = result.get("document_name", "N/A")
                content_preview = result.get("content", "")[:100]
                print(f"   {result_count}. {doc_name}")
                print(f"      Preview: {content_preview}...")
            
            if result_count == 0:
                print(f"   No results for query: '{test_query}'")
            else:
                print(f"\n   ✓ Found {result_count} results")
                
        except Exception as e:
            print(f"   Error in test search: {str(e)}")
        
        # Step 6: Check vector field
        print("\n6. Vector Search Configuration:")
        vector_fields = [f for f in index.fields if f.type == "Collection(Edm.Single)"]
        
        if vector_fields:
            print(f"   ✓ Found {len(vector_fields)} vector field(s):")
            for vf in vector_fields:
                print(f"      - {vf.name}")
        else:
            print("   ✗ No vector fields found!")
            print("   Hybrid search will not work without vector fields")
        
        # Step 7: Check required fields for unified_retrieval.py
        print("\n7. Required Fields Check:")
        required_fields = [
            "chunk_id",
            "document_name", 
            "document_type",
            "content",
            "content_vector"
        ]
        
        optional_fields = ["page_number", "section"]
        
        existing_field_names = [f.name for f in index.fields]
        
        print("   Required fields:")
        for field in required_fields:
            if field in existing_field_names:
                print(f"      ✓ {field}")
            else:
                print(f"      ✗ {field} - MISSING!")
        
        print("\n   Optional fields:")
        for field in optional_fields:
            if field in existing_field_names:
                print(f"      ✓ {field}")
            else:
                print(f"      ⚠️  {field} - not found (optional)")
        
        # Step 8: Test vector search
        print("\n8. Test Vector Search:")
        if vector_fields:
            try:
                # Create a dummy vector (all zeros)
                vector_field_name = vector_fields[0].name
                vector_size = 3072  # text-embedding-3-large size
                dummy_vector = [0.0] * vector_size
                
                results = search_client.search(
                    search_text=None,
                    vector_queries=[{
                        "kind": "vector",
                        "vector": dummy_vector,
                        "fields": vector_field_name,
                        "k": 3
                    }],
                    top=3
                )
                
                result_count = sum(1 for _ in results)
                print(f"   ✓ Vector search works - found {result_count} results")
                
            except Exception as e:
                print(f"   ✗ Vector search error: {str(e)}")
        else:
            print("   ⚠️  Skipped - no vector fields")
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_index()
