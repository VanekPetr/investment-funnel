import importlib

# Test importing just the specific module that was causing the original error
try:
    importlib.import_module('funnel.pages.models.ai_feature')
    print("Successfully imported funnel.pages.models.ai_feature")
except Exception as e:
    print(f"Error importing funnel.pages.models.ai_feature: {e}")