import subprocess
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import os

ALL_MODELS = ["phi", "qwen", "mistral7b", "gemma2", "llama"]
ALL_METHODS = ["cat", "ties", "individual_dialect", "lora_grouping", "base_instruct"]

def run_evaluation(model, method, output_dir="results"):

    print(f"Running evaluation: {model} with {method}")

    cmd = [
        "python",
        "evaluator_unified/evaluate_merge_methods_fewshot.py",
        "--model", model,
        "--method", method,
        "--output-dir", output_dir
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False, 
            text=True
        )
        print(f"Completed: {model} with {method}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed: {model} with {method}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"Interrupted by user")
        sys.exit(1)


def aggregate_results(output_dir="results_besstie_unified"):
    
    print("Aggregating results...")
    
    all_results = {}

    for model in ALL_MODELS:
        all_results[model] = {}

        for method in ALL_METHODS:
            result_file = os.path.join(output_dir, model, method, "results.json")

            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    all_results[model][method] = {
                        "summary": data.get("summary", {}),
                        "timestamp": data.get("timestamp", ""),
                        "results": data.get("results", {})
                    }
            else:
                all_results[model][method] = None
                print(f"Missing results for {model}/{method}")

    summary_file = os.path.join(output_dir, "all_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Aggregated results saved to: {summary_file}")
    print("Results Summary Table")
    print(f"{'Model':<15} {'Method':<15} {'Avg Accuracy':<15} {'Avg F1':<15}")

    for model in ALL_MODELS:
        for method in ALL_METHODS:
            if all_results[model][method]:
                summary = all_results[model][method]["summary"]
                acc = summary.get("average_accuracy", 0)
                f1 = summary.get("average_f1", 0)
                print(f"{model:<15} {method:<15} {acc:<15.4f} {f1:<15.4f}")
            else:
                print(f"{model:<15} {method:<15} {'N/A':<15} {'N/A':<15}")

def main():
    parser = argparse.ArgumentParser(description="Run all LoRA merging evaluations")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="Methods to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip aggregation step at the end"
    )

    args = parser.parse_args()

    evaluations = [
        (model, method)
        for model in args.models
        for method in args.methods
    ]

    total_evals = len(evaluations)

    print(f"Batch Evaluation Runner")
    print(f"Models: {', '.join(args.models)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Total evaluations: {total_evals}")
    print(f"Output directory: {args.output_dir}")

    successful = []
    failed = []
    
    start_time = datetime.now()

    for idx, (model, method) in enumerate(evaluations, 1):
        print(f"[{idx}/{total_evals}] Starting: {model} with {method}")

        success = run_evaluation(model, method, args.output_dir)

        if success:
            successful.append((model, method))
        else:
            failed.append((model, method))

    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Batch Evaluation Complete")
    print(f"Total time: {duration}")
    print(f"Successful: {len(successful)}/{total_evals}")
    print(f"Failed: {len(failed)}/{total_evals}")

    if failed:
        print(f"Failed evaluations:")
        for model, method in failed:
            print(f"  - {model} with {method}")

    if not args.skip_aggregation:
        aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
