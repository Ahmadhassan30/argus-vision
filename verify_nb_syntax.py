import json
import traceback

def check_notebook(path):
    print(f"Verifying {path}...")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    errors = 0
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            lines = []
            for line in cell.get("source", []):
                trimmed = line.strip()
                if trimmed.startswith("%") or trimmed.startswith("!"):
                    lines.append("# " + line)
                else:
                    lines.append(line)
            cleaned_source = "".join(lines)
            compile(cleaned_source, f"{path}_cell_{idx}", "exec")
        except SyntaxError as e:
            print(f"  [SyntaxError] Cell {idx} at line {e.lineno}: {e.msg}")
            print(f"  Code snippet:")
            snippet = cell.get("source", [])
            start = max(0, e.lineno - 3)
            end = min(len(snippet), e.lineno + 3)
            for lno in range(start, end):
                prefix = "-> " if lno + 1 == e.lineno else "   "
                print(f"    {prefix}{lno+1}: {snippet[lno].rstrip()}")
            errors += 1
        except Exception as e:
            print(f"  [Error] Cell {idx}: {e}")
            traceback.print_exc()
            errors += 1
            
    if errors == 0:
        print(f"  No syntax errors found in {path}.")
    else:
        print(f"  Found {errors} errors in {path}.")
    return errors

errs = check_notebook("ml_training/04_train_consensus.ipynb")
errs += check_notebook("ml_training/05_evaluation.ipynb")
if errs > 0:
    exit(1)
else:
    print("All syntax checks passed successfully!")
