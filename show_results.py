import json
from pathlib import Path

print("\nMODEL COMPARISON RESULTS\n")

file = Path("models/model_comparison_metrics.json")

if not file.exists():
    print("Metrics file not found. Run training first.")
else:
    with open(file) as f:
        results = json.load(f)

    for model, score in results.items():
        print(f"{model:<20} : {score:.4f}")
