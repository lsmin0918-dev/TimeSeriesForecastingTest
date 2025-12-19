import json

nb_path = "c:\\AI\\FinalProject\\TimeSeriesForecastingTest\\assignment_notebook.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        source_str = "".join(cell["source"])
        if "## ✅ 과제 체크리스트" in source_str:
            new_source = []
            for line in cell["source"]:
                if "- [x]" in line:
                    # Replace '- [x]' with '- ✅'
                    new_line = line.replace("- [x]", "- ✅")
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            
            if new_source != cell["source"]:
                cell["source"] = new_source
                updated = True
                print("Checklist updated successfully.")
            break

if updated:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook saved.")
else:
    print("No changes needed or checklist not found.")
