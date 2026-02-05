
import os
import sys

# Add the source directory to path so we can import modules
sys.path.append(os.getcwd())

try:
    from psp.serialization import load_model
    import quartz_solar_forecast.forecasts.v1 as v1
    
    print(f"Python Version: {sys.version}")
    
    MODELS = ["model-0.3.0.pkl", "model-0.4.0.pkl"]
    
    for model_file in MODELS:
        print(f"\nProcessing {model_file}...")
        model_path = os.path.join(os.path.dirname(v1.__file__), f"../models/{model_file}")
        model_path = os.path.normpath(model_path)
        
        print(f"Path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Skipping {model_file} - Not found")
            continue

        model = load_model(model_path)
        print(f"SUCCESS: Loaded {model_file}")
        
        # Re-save
        print(f"Re-saving {model_file}...")
        with open(model_path, "wb") as f:
            import pickle
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SUCCESS: Re-saved {model_file}")

except Exception as e:
    print("\nFAILURE: Could not load model.")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    import traceback
    traceback.print_exc()
