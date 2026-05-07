import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, UTC
import re
from copy import deepcopy

# ============================================================
# USER CONFIGURATION
# ============================================================

MUTATION_AGENT_FOLDER = "/Users/pranava/Downloads/Dataset-Final-3/Mutation_Agent"

EXECUTION_AGENT_FOLDER = "/Users/pranava/Downloads/execution_outputs/outputs/Execution_Agent"

OUTPUT_DATASET_JSON = "mascod_bench_dataset_final2.json"

# ============================================================
# IGNORE RULES
# ============================================================

def should_ignore_path(path_obj):

    ignored_parts = {
        "__MACOSX",
        ".DS_Store"
    }

    for part in path_obj.parts:

        if part in ignored_parts:
            return True

    return False

# ============================================================
# LOADERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(line)
                )

            except Exception as e:

                print(
                    f"[JSONL ERROR] {path}"
                )

                print(
                    f"Line: {line_num}"
                )

                print(e)

    return rows

# ============================================================
# ANONYMIZATION
# ============================================================

ANONYMIZATION_RULES = [

    # Names
    (
        re.compile(
            r"Subrahmanya\s+Sree\s+Pranava\s+Sai\s+Maganti",
            re.IGNORECASE
        ),
        "mascod-mlirbench"
    ),

    (
        re.compile(
            r"Pranava\s+Sai\s+Maganti",
            re.IGNORECASE
        ),
        "mascod-mlirbench"
    ),

    (
        re.compile(
            r"\bPranava\b",
            re.IGNORECASE
        ),
        "mascod-mlirbench"
    ),

    # Emails
    (
        re.compile(
            r"[\w\.-]+@iastate\.edu",
            re.IGNORECASE
        ),
        "mascodbench@gmail.com"
    ),

    (
        re.compile(
            r"[\w\.-]+@gmail\.com",
            re.IGNORECASE
        ),
        "mascodbench@gmail.com"
    ),

    # Mac paths
    (
        re.compile(
            r"/Users/pranava",
            re.IGNORECASE
        ),
        "/Users/mascod-mlirbench"
    ),

    # GitHub usernames
    (
        re.compile(
            r"pranava[-_]sai",
            re.IGNORECASE
        ),
        "mascod-mlirbench"
    ),

    (
        re.compile(
            r"\bpranava7\b",
            re.IGNORECASE
        ),
        "mascod-mlirbench"
    ),
]

def anonymize_text(text):

    if not isinstance(text, str):
        return text

    updated = text

    for pattern, replacement in ANONYMIZATION_RULES:

        updated = pattern.sub(
            replacement,
            updated
        )

    return updated

def anonymize_object(obj):

    if isinstance(obj, dict):

        return {
            anonymize_text(k):
                anonymize_object(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):

        return [
            anonymize_object(x)
            for x in obj
        ]

    elif isinstance(obj, str):

        return anonymize_text(obj)

    else:
        return obj

# ============================================================
# PROGRAM ID GENERATOR
# ============================================================

def build_program_id(
    seed,
    dialect,
    program_name,
    mutation_name
):

    # ========================================================
    # NORMALIZE SEED
    # ========================================================

    seed_num = (
        seed.lower()
        .replace("seed_", "")
        .replace("seed", "")
    )

    # ========================================================
    # NORMALIZE PROGRAM
    # ========================================================

    program_num = (
        program_name
        .replace("program_", "")
        .replace(".mlir", "")
    )

    # ========================================================
    # NORMALIZE MUTATION RULE
    # ========================================================

    mutation_rule = (
        mutation_name
        .replace(".json", "")
    )

    # ========================================================
    # MATCH EXECUTION AGENT FORMAT
    # ========================================================

    # Correct format:
    # linalg_seed1_prog1_L1_R3
    # torch_seed4_prog31_T1_R1

    return (
        f"{dialect}_"
        f"seed{seed_num}_"
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

    root = Path(root_dir)

    for dialect_dir in root.iterdir():

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

                        program_id = build_program_id(
                            seed=seed_dir.name,
                            dialect=dialect,
                            program_name=program_dir.name,
                            mutation_name=mutation_json.name
                        )

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

                        mutation_results[program_id] = {

                            "program_id": program_id,

                            "dialect": dialect,

                            "seed": seed_dir.name,

                            "program": program_dir.name,

                            "mutation_rule_file":
                                mutation_json.name,

                            "source_file":
                                str(mutation_json),

                            "mutation_agent": {

                                "context_results":
                                    context_results,

                                "blind_results":
                                    blind_results
                            }
                        }

                        stats["total_samples"] += 1

                        stats["dialects"][
                            dialect
                        ] += 1

                        if level is not None:

                            stats["levels"][
                                f"L{level}"
                            ] += 1

                        if rule_id:

                            stats["rules"][
                                rule_id
                            ] += 1

                    except Exception as e:

                        print(
                            f"[ERROR] Failed parsing:"
                        )

                        print(mutation_json)

                        print(e)

    return mutation_results, stats

# ============================================================
# EXECUTION ARTIFACT DETECTOR
# ============================================================

def detect_artifact_type(path, data):

    path_str = str(path).lower()

    if "judge_payload" in path_str:
        return "judge_payload"

    if "pairs_benchmark" in path_str:
        return "pair_benchmark"

    if (
        "benchmark_A" in data or
        "benchmark_B" in data
    ):
        return "pair_benchmark"

    if (
        "semantic_explanation" in data
    ):
        return "judge_payload"

    return "unknown"

# ============================================================
# PROCESS EXECUTION ARTIFACT
# ============================================================

def process_execution_artifact(
    path,
    data,
    execution_results,
    stats
):

    if not isinstance(data, dict):
        return

    program_id = data.get("program_id")

    if not program_id:
        return

    entry = execution_results[program_id]

    entry["program_id"] = program_id

    artifact = {

        "source_file": str(path),

        "artifact_type":
            detect_artifact_type(path, data),

        "data": data
    }

    artifact_type = artifact["artifact_type"]

    if artifact_type == "judge_payload":

        entry["judge_payloads"].append(
            artifact
        )

        stats["judge_payloads"] += 1

    elif artifact_type == "pair_benchmark":

        entry["pair_benchmarks"].append(
            artifact
        )

        stats["pair_benchmarks"] += 1

    else:

        entry["other_artifacts"].append(
            artifact
        )

    stats["total_artifacts"] += 1

# ============================================================
# EXECUTION PARSER
# ============================================================

def parse_execution_agent_results(root_dir):

    execution_results = defaultdict(

        lambda: {

            "program_id": None,

            "judge_payloads": [],

            "pair_benchmarks": [],

            "other_artifacts": []
        }
    )

    stats = {

        "total_artifacts": 0,

        "judge_payloads": 0,

        "pair_benchmarks": 0,

        "json_files": 0,

        "jsonl_rows": 0
    }

    root = Path(root_dir)

    for path in root.rglob("*"):

        if should_ignore_path(path):
            continue

        if not path.is_file():
            continue

        try:

            # ====================================================
            # JSON FILES
            # ====================================================

            if path.suffix.lower() == ".json":

                stats["json_files"] += 1

                data = load_json(path)

                process_execution_artifact(
                    path,
                    data,
                    execution_results,
                    stats
                )

            # ====================================================
            # JSONL FILES
            # ====================================================

            elif path.suffix.lower() == ".jsonl":

                rows = load_jsonl(path)

                stats["jsonl_rows"] += len(rows)

                for row in rows:

                    process_execution_artifact(
                        path,
                        row,
                        execution_results,
                        stats
                    )

        except Exception as e:

            print(
                f"[ERROR] Failed parsing:"
            )

            print(path)

            print(e)

    return execution_results, stats

# ============================================================
# MERGE DATASETS
# ============================================================

# ============================================================
# PROGRAM ID NORMALIZER
# ============================================================

def normalize_program_id(program_id):

    if not program_id:
        return None

    pid = str(program_id).lower()

    # ========================================================
    # REMOVE EXTENSIONS
    # ========================================================

    pid = pid.replace(".json", "")
    pid = pid.replace(".mlir", "")

    # ========================================================
    # STANDARDIZE TOKENS
    # ========================================================

    pid = pid.replace("program_", "prog")
    pid = pid.replace("program", "prog")
    pid = pid.replace("prog_", "prog")

    pid = pid.replace("seed_", "seed")

    # ========================================================
    # REMOVE DOUBLE UNDERSCORES
    # ========================================================

    while "__" in pid:
        pid = pid.replace("__", "_")

    return pid.strip("_")

# ============================================================
# MERGE DATASETS
# ============================================================

def merge_datasets(
    mutation_results,
    execution_results
):

    merged_samples = []

    # ========================================================
    # NORMALIZE EXECUTION LOOKUP
    # ========================================================

    normalized_execution = {}

    for exec_pid, exec_data in execution_results.items():

        normalized_pid = normalize_program_id(
            exec_pid
        )

        normalized_execution[
            normalized_pid
        ] = exec_data

    # ========================================================
    # NORMALIZE MUTATION LOOKUP
    # ========================================================

    normalized_mutation = {}

    for mut_pid, mut_data in mutation_results.items():

        normalized_pid = normalize_program_id(
            mut_pid
        )

        normalized_mutation[
            normalized_pid
        ] = mut_data

    # ========================================================
    # UNION OF IDS
    # ========================================================

    all_program_ids = sorted(

        set(normalized_mutation.keys()) |
        set(normalized_execution.keys())
    )

    # ========================================================
    # MERGE
    # ========================================================

    for normalized_pid in all_program_ids:

        mutation_data = normalized_mutation.get(
            normalized_pid
        )

        execution_data = normalized_execution.get(
            normalized_pid,
            {}
        )

        sample = {

            "program_id":
                normalized_pid,

            "dialect":
                mutation_data.get("dialect")
                if mutation_data else None,

            "seed":
                mutation_data.get("seed")
                if mutation_data else None,

            "program":
                mutation_data.get("program")
                if mutation_data else None,

            "mutation_agent":
                mutation_data.get(
                    "mutation_agent"
                )
                if mutation_data else None,

            "execution_sandbox_agent": {

                "judge_payloads":
                    execution_data.get(
                        "judge_payloads",
                        []
                    ),

                "pair_benchmarks":
                    execution_data.get(
                        "pair_benchmarks",
                        []
                    ),

                "other_artifacts":
                    execution_data.get(
                        "other_artifacts",
                        []
                    )
            }
        }

        merged_samples.append(sample)

    return merged_samples

# ============================================================
# DATASET OVERVIEW
# ============================================================

def build_dataset_overview(
    mutation_stats,
    execution_stats,
    final_sample_count
):

    return {

        "dataset_name": "MASCoD-Bench",

        "generated_at":
            datetime.now(UTC).isoformat(),

        "description":
            (
                "MASCoD-Bench is a benchmark "
                "dataset for evaluating "
                "semantic reasoning "
                "capabilities of LLMs "
                "over MLIR mutations."
            ),

        "total_samples":
            final_sample_count,

        "mutation_agent": {

            "total_samples":
                mutation_stats["total_samples"],

            "dialect_distribution":
                dict(
                    mutation_stats["dialects"]
                ),

            "mutation_level_distribution":
                dict(
                    mutation_stats["levels"]
                ),

            "mutation_rule_distribution":
                dict(
                    mutation_stats["rules"]
                )
        },

        "execution_sandbox_agent": {

            "total_artifacts":
                execution_stats[
                    "total_artifacts"
                ],

            "judge_payloads":
                execution_stats[
                    "judge_payloads"
                ],

            "pair_benchmarks":
                execution_stats[
                    "pair_benchmarks"
                ],

            "json_files":
                execution_stats[
                    "json_files"
                ],

            "jsonl_rows":
                execution_stats[
                    "jsonl_rows"
                ]
        }
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MASCoD-Bench Dataset Curation")
    print("=" * 80)

    # ========================================================
    # MUTATION AGENT
    # ========================================================

    print(
        "\n[1] Parsing Mutation Agent Results..."
    )

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
    # EXECUTION SANDBOX
    # ========================================================

    print(
        "\n[2] Parsing Execution Sandbox Results..."
    )

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

    merged_samples = merge_datasets(
        mutation_results,
        execution_results
    )

    print(
        f"    Final merged samples: "
        f"{len(merged_samples)}"
    )

    # ========================================================
    # MERGE VALIDATION
    # ========================================================

    with_execution = 0
    without_execution = 0

    for sample in merged_samples:

        execution = sample.get(
            "execution_sandbox_agent",
            {}
        )

        has_execution = (
            len(
                execution.get(
                    "judge_payloads",
                    []
                )
            ) > 0
            or
            len(
                execution.get(
                    "pair_benchmarks",
                    []
                )
            ) > 0
        )

        if has_execution:
            with_execution += 1
        else:
            without_execution += 1

    print(
        f"    Samples WITH execution data: "
        f"{with_execution}"
    )

    print(
        f"    Samples WITHOUT execution data: "
        f"{without_execution}"
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    print(
        "\n[4] Building Dataset Overview..."
    )

    dataset_overview = (
        build_dataset_overview(
            mutation_stats,
            execution_stats,
            len(merged_samples)
        )
    )

    # ========================================================
    # FINAL DATASET
    # ========================================================

    final_dataset = {

        "dataset_overview":
            dataset_overview,

        "samples":
            merged_samples
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

        anonymized_dataset = anonymize_object(
            deepcopy(final_dataset)
        )

        json.dump(
            anonymized_dataset,
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