import pandas as pd
import json
import os

def merge_json_to_csv(json_folder, output_csv):
    all_data = []
    
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            intent_name = filename.split(".")[0].capitalize()
            
            file_path = os.path.join(json_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Grab the queries list from the JSON
                queries = data.get("queries", [])
                
                for q in queries:
                    all_data.append({"Query": q, "Intent": intent_name})

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f" Successfully merged {len(df)} samples into {output_csv}")

if __name__ == "__main__":
    # Point this to wherever Omi saved the JSON files
    merge_json_to_csv("training_jsons", "queries_dataset.csv")