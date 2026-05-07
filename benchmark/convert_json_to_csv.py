import json
import csv
import os

INPUT_JSON = "benchmark.json"
OUTPUT_CSV = "benchmark.csv"

def flatten_score_entry(entry):
    return {
        "evaluated_model": entry.get("evaluated_model"),
        "ground_truth_label": entry.get("ground_truth_label"),
        "predicted_label": entry.get("predicted_label"),
        "pair_id": entry.get("pair_id"),
        "program_id": entry.get("program_id"),
        "total_score": entry.get("total_score"),
        "verdict": entry.get("verdict"),

        "explanation_rating":
            entry.get("subscores", {}).get("explanation_rating"),

        "hard_label_accuracy":
            entry.get("subscores", {}).get("hard_label_accuracy"),

        "result_accuracy":
            entry.get("subscores", {}).get("result_accuracy"),

        "weighted_explanation_rating":
            entry.get("weighted_scores", {}).get("explanation_rating"),

        "weighted_hard_label":
            entry.get("weighted_scores", {}).get("hard_label"),

        "weighted_result_accuracy":
            entry.get("weighted_scores", {}).get("result_accuracy"),
    }

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: {INPUT_JSON} not found")
        return

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    scores = data.get("scores", [])

    if not scores:
        print("No scores found.")
        return

    rows = [flatten_score_entry(s) for s in scores]

    fieldnames = rows[0].keys()

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"CSV generated: {OUTPUT_CSV}")
    print(f"Rows written: {len(rows)}")

if __name__ == "__main__":
    main()