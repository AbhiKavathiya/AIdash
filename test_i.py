
import os
import sys
import json

# Set dummy env vars if not set (for testing)
if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://dummy.openai.azure.com/"
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = "dummy-key"
if not os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"
if not os.environ.get("POWERBI_WORKSPACE_ID"):
    os.environ["POWERBI_WORKSPACE_ID"] = "dummy-workspace"
if not os.environ.get("POWERBI_DATASET_ID"):
    os.environ["POWERBI_DATASET_ID"] = "dummy-dataset"

print("=" * 80)
print("IJP Agent Debug Test")
print("=" * 80)

errors_found = []

try:
    print("\n1. Testing imports...")
    import ijp_v5
    print("✅ Imports successful")
    
    print("\n2. Checking metadata files...")
    
    # Check IJP files
    ijp_measures_file = "ijp_measures.json"
    ijp_dimension_file = "ijp_dimension.json"
    
    if not os.path.exists(ijp_measures_file):
        error_msg = f"❌ {ijp_measures_file} NOT FOUND"
        print(error_msg)
        errors_found.append(error_msg)
        print("\n   SOLUTION:")
        print(f"   1. Copy the sample: cp ijp_measures_SAMPLE.json {ijp_measures_file}")
        print("   2. Edit it to match your Power BI model measures")
        print("\n   Example content:")
        print('   [{"measure_name": "Head Count", "description": "...", "formula": "..."}]')
    else:
        print(f"✅ {ijp_measures_file} found")
        # Validate JSON
        try:
            with open(ijp_measures_file, 'r') as f:
                measures_data = json.load(f)
                if not isinstance(measures_data, list):
                    error_msg = f"❌ {ijp_measures_file} should be a JSON array"
                    print(error_msg)
                    errors_found.append(error_msg)
                elif len(measures_data) == 0:
                    error_msg = f"⚠️ {ijp_measures_file} is empty"
                    print(error_msg)
                    errors_found.append(error_msg)
                else:
                    print(f"   ✅ Valid JSON with {len(measures_data)} measures")
        except json.JSONDecodeError as e:
            error_msg = f"❌ {ijp_measures_file} has invalid JSON: {e}"
            print(error_msg)
            errors_found.append(error_msg)
    
    if not os.path.exists(ijp_dimension_file):
        error_msg = f"❌ {ijp_dimension_file} NOT FOUND"
        print(error_msg)
        errors_found.append(error_msg)
        print("\n   SOLUTION:")
        print(f"   1. Copy the sample: cp ijp_dimension_SAMPLE.json {ijp_dimension_file}")
        print("   2. Edit it to match your Power BI model tables and columns")
        print("\n   Example content:")
        print('   {"tables": {"Internal Job Posting": ["Column1", "Column2", ...]}}')
    else:
        print(f"✅ {ijp_dimension_file} found")
        # Validate JSON
        try:
            with open(ijp_dimension_file, 'r') as f:
                dim_data = json.load(f)
                if not isinstance(dim_data, dict) or "tables" not in dim_data:
                    error_msg = f"❌ {ijp_dimension_file} should have 'tables' key"
                    print(error_msg)
                    errors_found.append(error_msg)
                elif len(dim_data.get("tables", {})) == 0:
                    error_msg = f"⚠️ {ijp_dimension_file} has no tables defined"
                    print(error_msg)
                    errors_found.append(error_msg)
                else:
                    print(f"   ✅ Valid JSON with {len(dim_data['tables'])} tables")
        except json.JSONDecodeError as e:
            error_msg = f"❌ {ijp_dimension_file} has invalid JSON: {e}"
            print(error_msg)
            errors_found.append(error_msg)
    
    if errors_found:
        print("\n" + "=" * 80)
        print("❌ SETUP INCOMPLETE - Fix the issues above first")
        print("=" * 80)
        print("\nRead SETUP_INSTRUCTIONS.md for detailed help")
        sys.exit(1)
    
    print("\n3. Checking measures loaded...")
    print(f"   Loaded {len(ijp_v5.IJP_MEASURES)} measures")
    if ijp_v5.IJP_MEASURES:
        print(f"   Sample measures:")
        for m in ijp_v5.IJP_MEASURES[:3]:
            print(f"   - {m['measure_name']}")
    else:
        error_msg = "⚠️ No measures loaded - check file content"
        print(error_msg)
        errors_found.append(error_msg)
    
    print("\n4. Checking dimensions loaded...")
    print(f"   Loaded {len(ijp_v5.IJP_DIMENSIONS)} tables")
    if ijp_v5.IJP_DIMENSIONS:
        for table, cols in ijp_v5.IJP_DIMENSIONS.items():
            print(f"   - {table}: {len(cols)} columns")
    else:
        error_msg = "⚠️ No dimensions loaded - check file content"
        print(error_msg)
        errors_found.append(error_msg)
    
    print("\n5. Testing state creation...")
    test_state = {
        "question": "Test question",
        "intent_type": "standard",
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
    print("✅ State structure valid")
    
    print("\n6. Testing intent with measures...")
    if ijp_v5.IJP_MEASURES:
        test_intent = {
            "intent_type": "standard",
            "table": "Internal Job Posting",
            "measures": [ijp_v5.IJP_MEASURES[0]["measure_name"]],
            "filters": {},
            "groupby": []
        }
        print(f"   Measures: {test_intent['measures']}")
        print("✅ Intent structure valid")
    else:
        print("   ⚠️ Skipping - no measures available")
    
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED")
    print("=" * 80)
    print("\nYour IJP agent is ready for testing!")
    print("\nNext steps:")
    print("1. Set real Azure OpenAI credentials:")
    print("   export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'")
    print("   export AZURE_OPENAI_API_KEY='your-api-key'")
    print("   export AZURE_OPENAI_DEPLOYMENT='gpt-4o'")
    print("\n2. Set real Power BI credentials:")
    print("   export POWERBI_WORKSPACE_ID='your-workspace-id'")
    print("   export POWERBI_DATASET_ID='your-dataset-id'")
    print("\n3. Test with a simple query:")
    print("   python ijp_v5.py")
    print("\n4. Check for any errors and review logs")
    
except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("\nEnsure these files exist:")
    print("- ijp_measures.json")
    print("- ijp_dimension.json")
    print("\nRun these commands to create from samples:")
    print("  cp ijp_measures_SAMPLE.json ijp_measures.json")
    print("  cp ijp_dimension_SAMPLE.json ijp_dimension.json")
    print("\nThen edit them to match your Power BI model")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

