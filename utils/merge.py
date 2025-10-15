import os
import json

DATASET_DIRECTORY = "data/instruction/moe_peft"  

merged_data = []

for filename in os.listdir(DATASET_DIRECTORY):
    if filename.endswith(".json"):
        file_path = os.path.join(DATASET_DIRECTORY, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Skipping {filename}: not a list of records")
            except json.JSONDecodeError:
                print(f"Skipping {filename}: invalid JSON")

# Save the merged dataset
output_path = os.path.join(DATASET_DIRECTORY, "merged_dataset.json")
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(merged_data, fout, indent=4, ensure_ascii=False)

print(f"Merged {len(merged_data)} entries into {output_path}")
