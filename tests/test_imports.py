import importlib

# Test importing the pages
try:
    importlib.import_module('funnel.pages.ai_feature_selection')
    print("Successfully imported ai_feature_selection")
except Exception as e:
    print(f"Error importing ai_feature_selection: {e}")

try:
    importlib.import_module('funnel.pages.overview')
    print("Successfully imported overview")
except Exception as e:
    print(f"Error importing overview: {e}")

try:
    importlib.import_module('funnel.pages.lifecycle')
    print("Successfully imported lifecycle")
except Exception as e:
    print(f"Error importing lifecycle: {e}")

try:
    importlib.import_module('funnel.pages.backtest')
    print("Successfully imported backtest")
except Exception as e:
    print(f"Error importing backtest: {e}")