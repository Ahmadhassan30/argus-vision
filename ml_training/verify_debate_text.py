import json
import os
import sys
from unittest.mock import MagicMock

# Mock all heavy ML/CV libraries so this script runs with zero dependencies locally
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["pytorch_grad_cam"] = MagicMock()
sys.modules["pytorch_grad_cam.utils"] = MagicMock()
sys.modules["pytorch_grad_cam.utils.model_targets"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.spatial"] = MagicMock()
sys.modules["scipy.spatial.distance"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

# Test input data for verification
test_stats = {"mean": 0.35467, "std": 0.12345, "max": 0.87654}
test_cases = [
    ("MEL", 0.85, test_stats, "NV"),
    ("AK", 0.42, test_stats, "BCC"),
    ("DF", 0.67, test_stats, "VASC"),
]

def load_notebook_defs(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # We will search for all cells and extract lines defining the relevant functions/constants
    combined_code = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "ISIC_CLASS_DESCRIPTIONS" in source or "_fallback_argument" in source:
                combined_code.append(source)
                
    full_source = "\n".join(combined_code)
    
    sandbox = {}
    sandbox["Dict"] = dict
    sandbox["Any"] = lambda x: x
    sandbox["List"] = list
    sandbox["Optional"] = lambda x: x
    sandbox["Tuple"] = tuple
    sandbox["re"] = __import__("re")
    sandbox["np"] = __import__("numpy")
    sandbox["os"] = __import__("os")
    sandbox["torch"] = sys.modules["torch"]
    sandbox["cv2"] = sys.modules["cv2"]
    sandbox["nn"] = sys.modules["torch.nn"]
    sandbox["F"] = sys.modules["torch.nn.functional"]
    sandbox["find_file"] = lambda x: None
    sandbox["FULL_NAMES"] = {
        "MEL": "Melanoma", "NV": "Melanocytic Nevus", "BCC": "Basal Cell Carcinoma",
        "AK": "Actinic Keratosis", "BKL": "Benign Keratosis", "DF": "Dermatofibroma",
        "VASC": "Vascular Lesion", "SCC": "Squamous Cell Carcinoma",
    }
    
    try:
        exec(full_source, sandbox)
    except Exception as exc:
        print(f"Error executing notebook source from {nb_path}: {exc}")
        # Print source lines for debugging
        for idx, line in enumerate(full_source.splitlines()):
            print(f"{idx+1}: {line}")
        raise exc
        
    return sandbox

def load_backend_defs():
    # Execute argument_gen.py in a sandbox
    path = "backend/ml/debate/argument_gen.py"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    
    sandbox = {}
    sandbox["dict"] = dict
    sandbox["Any"] = lambda x: x
    sandbox["Optional"] = lambda x: x
    sandbox["re"] = __import__("re")
    sandbox["logging"] = __import__("logging")
    
    exec(source, sandbox)
    return sandbox

def main():
    print("=== Verification of debate argument wording alignment ===")
    
    # Load debate_text_utils
    import debate_text_utils as ref
    
    # Load notebooks
    nb4 = load_notebook_defs("ml_training/04_train_consensus.ipynb")
    nb5 = load_notebook_defs("ml_training/05_evaluation.ipynb")
    backend = load_backend_defs()
    
    sources = {
        "debate_text_utils": (ref.CLASS_FULL_NAMES, ref.ISIC_CLASS_DESCRIPTIONS, ref._region_summary, ref._fallback_argument),
        "04_train_consensus": (
            nb4.get("CLASS_FULL_NAMES", nb4.get("CLASS_NAMES", nb4.get("ISIC_CLASSES"))), 
            nb4["ISIC_CLASS_DESCRIPTIONS"], 
            nb4["_region_summary"], 
            nb4["_fallback_argument"]
        ),
        "05_evaluation": (
            nb5.get("FULL_NAMES", nb5.get("CLASS_FULL_NAMES")), 
            nb5["ISIC_CLASS_DESCRIPTIONS"], 
            nb5["_region_summary"], 
            nb5["_fallback_argument"]
        ),
        "backend/argument_gen": (
            backend.get("CLASS_FULL_NAMES"), 
            backend["ISIC_CLASS_DESCRIPTIONS"], 
            backend["_region_summary"], 
            backend["_fallback_argument"]
        )
    }
    
    errors = []
    
    # Verify outputs match exactly
    for name, (full_names, descriptions, region_summary, fallback_argument) in sources.items():
        print(f"Checking {name}...")
        
        # Verify full name dict mapping is consistent (ISIC_CLASSES)
        classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
        
        # Handle dict or list for full_names
        for c in classes:
            # Full names
            if isinstance(full_names, dict):
                ref_val = ref.CLASS_FULL_NAMES[c]
                val = full_names.get(c)
                if val != ref_val:
                    errors.append(f"{name}: FULL_NAME of {c} is '{val}', expected '{ref_val}'")
            
            # Descriptions
            ref_desc = ref.ISIC_CLASS_DESCRIPTIONS[c]
            desc = descriptions.get(c)
            if desc != ref_desc:
                errors.append(f"{name}: Description of {c} is '{desc}', expected '{ref_desc}'")
                
        # Verify region_summary output
        try:
            ref_reg = ref._region_summary(test_stats)
            val_reg = region_summary(test_stats)
            if val_reg != ref_reg:
                errors.append(f"{name}: _region_summary output is '{val_reg}', expected '{ref_reg}'")
        except Exception as exc:
            errors.append(f"{name}: _region_summary failed with {exc}")
            
        # Verify fallback_argument output
        for pred, conf, stats, opponent in test_cases:
            try:
                ref_fallback = ref._fallback_argument(pred, conf, stats, opponent)
                
                # Check signature of fallback_argument (backend takes bbox/region_stats keyword)
                import inspect
                sig = inspect.signature(fallback_argument)
                kwargs = {}
                if "pred_class" in sig.parameters:
                    kwargs["pred_class"] = pred
                else:
                    kwargs["pred"] = pred
                    
                if "confidence" in sig.parameters:
                    kwargs["confidence"] = conf
                else:
                    kwargs["conf"] = conf
                    
                if "region_stats" in sig.parameters:
                    kwargs["region_stats"] = stats
                else:
                    kwargs["stats"] = stats
                    
                if "opponent_pred" in sig.parameters:
                    kwargs["opponent_pred"] = opponent
                elif "opponent" in sig.parameters:
                    kwargs["opponent"] = opponent
                elif "opp" in sig.parameters:
                    kwargs["opp"] = opponent
                    
                val_fallback = fallback_argument(**kwargs)
                if val_fallback != ref_fallback:
                    errors.append(f"{name}: _fallback_argument({pred}, {conf}, ...) output is:\n  '{val_fallback}'\nExpected:\n  '{ref_fallback}'")
            except Exception as exc:
                errors.append(f"{name}: _fallback_argument({pred}, {conf}, ...) failed with {exc}")
                
    if errors:
        print("\n!!! ALIGNMENT VERIFICATION FAILED !!!")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)
    else:
        print("\n=== SUCCESS: All debate text utils and fallback argument definitions are perfectly aligned! ===")
        sys.exit(0)

if __name__ == "__main__":
    main()
