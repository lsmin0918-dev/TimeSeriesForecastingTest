import json
import sys

# Force utf-8 output for console
sys.stdout.reconfigure(encoding='utf-8')

nb_path = "c:\\AI\\FinalProject\\TimeSeriesForecastingTest\\assignment_notebook.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])[:100].replace("\n", " ")
        print(f"Cell {i}: {source}")
