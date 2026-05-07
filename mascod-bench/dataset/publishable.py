import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ============================================================
# USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Enter the ALREADY-EXTRACTED folder paths
# ------------------------------------------------------------

MUTATION_AGENT_FOLDER = "/Users/pranava/Downloads/Dataset-Final-3/Mutation_Agent"

EXECUTION_AGENT_FOLDER = "/Users/pranava/Downloads/execution_outputs/outputs/Execution_Agent"

OUTPUT_DATASET_JSON = "mascod_bench_dataset.json"

# ============================================================
# IGNORE RULES
# ============================================================

def should_ignore_path(path_obj):
    """
    Ignore macOS metadata folders/files.
    """

    ignored_parts = {
        "__MACOSX",
        ".DS_Store"
    }

    for part in path_obj.parts:

        if part in ignored_parts:
            return True

    return False


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# ============================================================
# PROGRAM ID GENERATOR
# ============================================================

def build_program_id(
    seed,
    dialect,
    program_name,
    mutation_name
):
    """
    Converts:

        Seed_1
        program_11.mlir
        L4_R1.json

    into:

        seed1_linalg_prog11_L4_R1
    """

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_num = (
        seed.lower()
        .replace("seed_", "")
        .replace("seed", "")
    )

    # --------------------------------------------------------
    # Program
    # --------------------------------------------------------

    program_num = (
        program_name
        .replace("program_", "")
        .replace(".mlir", "")
    )

    # --------------------------------------------------------
    # Mutation Rule
    # --------------------------------------------------------

    mutation_rule = (
        mutation_name
        .replace(".json", "")
    )

    # --------------------------------------------------------
    # Final Program ID
    # --------------------------------------------------------

    return (
        f"seed{seed_num}_"
        f"{dialect}_"
        f"prog{program_num}_"
        f"{mutation_rule}"
    )


# ============================================================
# MUTATION AGENT PARSER
# ============================================================

def parse_mutation_agent_results(root_dir):

    mutation_results = {}

    stats = {
        "total_samples": 0,
        "dialects": defaultdict(int),
        "levels": defaultdict(int),
        "rules": defaultdict(int)
    }

    # --------------------------------------------------------
    # Expected Structure
    #
    # linalg/
    #   Seed_1/
    #       program_11.mlir/
    #           L4_R1.json
    # --------------------------------------------------------

    for dialect_dir in Path(root_dir).iterdir():

        if should_ignore_path(dialect_dir):
            continue

        if not dialect_dir.is_dir():
            continue

        dialect = dialect_dir.name.lower()

        for seed_dir in dialect_dir.iterdir():

            if should_ignore_path(seed_dir):
                continue

            if not seed_dir.is_dir():
                continue

            for program_dir in seed_dir.iterdir():

                if should_ignore_path(program_dir):
                    continue

                if not program_dir.is_dir():
                    continue

                for mutation_json in program_dir.glob("*.json"):

                    if should_ignore_path(mutation_json):
                        continue

                    try:

                        data = load_json(mutation_json)

                        # ------------------------------------------------
                        # Generate Program ID
                        # ------------------------------------------------

                        program_id = build_program_id(
                            seed=seed_dir.name,
                            dialect=dialect,
                            program_name=program_dir.name,
                            mutation_name=mutation_json.name
                        )

                        # ------------------------------------------------
                        # Extract Data
                        # ------------------------------------------------

                        context_results = data.get(
                            "context_results",
                            {}
                        )

                        blind_results = data.get(
                            "blind_results",
                            {}
                        )

                        level = context_results.get("level")

                        requirement = context_results.get(
                            "requirement",
                            {}
                        )

                        rule_id = requirement.get("id")

                        # ------------------------------------------------
                        # Final Mutation Object
                        # ------------------------------------------------

                        mutation_results[program_id] = {

                            "program_id": program_id,

                            "dialect": dialect,

                            "seed": seed_dir.name,

                            "program": program_dir.name,

                            "mutation_rule_file": mutation_json.name,

                            "mutation_agent": {

                                "context_results": context_results,

                                "blind_results": blind_results
                            }
                        }

                        # ------------------------------------------------
                        # Stats
                        # ------------------------------------------------

                        stats["total_samples"] += 1

                        stats["dialects"][dialect] += 1

                        if level is not None:
                            stats["levels"][f"L{level}"] += 1

                        if rule_id:
                            stats["rules"][rule_id] += 1

                    except Exception as e:

                        print(f"[ERROR] Failed parsing:")
                        print(mutation_json)
                        print(e)

    return mutation_results, stats


# ============================================================
# EXECUTION SANDBOX PARSER
# ============================================================

def parse_execution_agent_results(root_dir):

    execution_results = {}

    stats = {
        "total_samples": 0,
        "statuses": defaultdict(int),
        "compile_A": defaultdict(int),
        "compile_B": defaultdict(int)
    }

    benchmark_dir = (
        Path(root_dir) / "pairs_benchmark"
    )

    if not benchmark_dir.exists():
        raise Exception(
            "pairs_benchmark folder not found"
        )

    for jsonl_file in benchmark_dir.glob("*.jsonl"):

        if should_ignore_path(jsonl_file):
            continue

        rows = load_jsonl(jsonl_file)

        for row in rows:

            program_id = row.get("program_id")

            if not program_id:
                continue

            execution_results[program_id] = row

            stats["total_samples"] += 1

            status = row.get(
                "status",
                "unknown"
            )

            compile_A = row.get(
                "compile_status_A",
                "unknown"
            )

            compile_B = row.get(
                "compile_status_B",
                "unknown"
            )

            stats["statuses"][status] += 1

            stats["compile_A"][compile_A] += 1

            stats["compile_B"][compile_B] += 1

    return execution_results, stats


# ============================================================
# MERGE DATASETS
# ============================================================

def merge_datasets(
    mutation_results,
    execution_results
):

    merged_samples = []

    all_program_ids = sorted(
        set(mutation_results.keys()) |
        set(execution_results.keys())
    )

    missing_mutation = 0
    missing_execution = 0

    for program_id in all_program_ids:

        mutation_data = mutation_results.get(
            program_id
        )

        execution_data = execution_results.get(
            program_id
        )

        if mutation_data is None:
            missing_mutation += 1

        if execution_data is None:
            missing_execution += 1

        sample = {

            "program_id": program_id,

            "dialect": (
                mutation_data.get("dialect")
                if mutation_data else
                execution_data.get("dialect")
            ),

            "mutation_agent": (
                mutation_data.get("mutation_agent")
                if mutation_data else None
            ),

            "execution_sandbox_agent": execution_data
        }

        merged_samples.append(sample)

    return merged_samples, {

        "missing_mutation_entries":
            missing_mutation,

        "missing_execution_entries":
            missing_execution
    }


# ============================================================
# DATASET OVERVIEW
# ============================================================

def build_dataset_overview(
    mutation_stats,
    execution_stats,
    merge_stats,
    final_sample_count
):

    return {

        "dataset_name": "MASCoD-Bench",

        "generated_at": (
            datetime.utcnow().isoformat() + "Z"
        ),

        "description": (
            "MASCoD-Bench is a benchmark "
            "dataset for evaluating semantic "
            "reasoning capabilities of large "
            "language models over MLIR IR "
            "mutations."
        ),

        "total_samples": final_sample_count,

        # ----------------------------------------------------
        # Mutation Agent Stats
        # ----------------------------------------------------

        "mutation_agent": {

            "total_samples":
                mutation_stats["total_samples"],

            "dialect_distribution":
                dict(mutation_stats["dialects"]),

            "mutation_level_distribution":
                dict(mutation_stats["levels"]),

            "mutation_rule_distribution":
                dict(mutation_stats["rules"])
        },

        # ----------------------------------------------------
        # Execution Sandbox Stats
        # ----------------------------------------------------

        "execution_sandbox_agent": {

            "total_samples":
                execution_stats["total_samples"],

            "status_distribution":
                dict(execution_stats["statuses"]),

            "compile_status_A_distribution":
                dict(execution_stats["compile_A"]),

            "compile_status_B_distribution":
                dict(execution_stats["compile_B"])
        },

        # ----------------------------------------------------
        # Merge Stats
        # ----------------------------------------------------

        "merge_statistics": merge_stats
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MASCoD-Bench Dataset Curation")
    print("=" * 80)

    # ========================================================
    # PARSE MUTATION AGENT
    # ========================================================

    print("\n[1] Parsing Mutation Agent Results...")

    mutation_results, mutation_stats = (
        parse_mutation_agent_results(
            MUTATION_AGENT_FOLDER
        )
    )

    print(
        f"    Parsed "
        f"{len(mutation_results)} "
        f"mutation samples"
    )

    # ========================================================
    # PARSE EXECUTION SANDBOX
    # ========================================================

    print("\n[2] Parsing Execution Sandbox Results...")

    execution_results, execution_stats = (
        parse_execution_agent_results(
            EXECUTION_AGENT_FOLDER
        )
    )

    print(
        f"    Parsed "
        f"{len(execution_results)} "
        f"execution samples"
    )

    # ========================================================
    # MERGE
    # ========================================================

    print("\n[3] Merging Datasets...")

    merged_samples, merge_stats = merge_datasets(
        mutation_results,
        execution_results
    )

    print(
        f"    Final merged samples: "
        f"{len(merged_samples)}"
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    print("\n[4] Building Dataset Overview...")

    dataset_overview = build_dataset_overview(
        mutation_stats,
        execution_stats,
        merge_stats,
        len(merged_samples)
    )

    # ========================================================
    # FINAL DATASET
    # ========================================================

    final_dataset = {

        "dataset_overview": dataset_overview,

        "samples": merged_samples
    }

    # ========================================================
    # SAVE
    # ========================================================

    print("\n[5] Saving Final Dataset...")

    with open(
        OUTPUT_DATASET_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDataset saved to: "
        f"{OUTPUT_DATASET_JSON}"
    )

    print("\nDONE")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()