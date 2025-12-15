"""
Batch runner script to evaluate all models with all merging methods

Runs 18 evaluations total:
- 6 models: phi, mistral7b, mistral2409, qwen, llama, gemma2
- 3 methods: lora_grouping, cat, ties

Usage:
    python run_all_evaluations.py
    python run_all_evaluations.py --models phi qwen --methods cat ties
"""

import subprocess
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import os

# All available models and methods
# ALL_MODELS = ["phi", "qwen", "llama", "mistral7b", "gemma2"]
ALL_MODELS = ["qwen"]
ALL_METHODS = ["lora_grouping", "cat", "ties", "base_instruct", "individual_dialect"]

def run_evaluation(model, method, output_dir="results"):
    """Run a single evaluation"""
    
    print(f"\n{'='*80}")
    print(f"Running evaluation: {model} with {method}")
    print(f"{'='*80}\n")

    cmd = [
        "python",
        "evaluator_unified/evaluate_merge_methods.py",
        "--model", model,
        "--method", method,
        "--output-dir", output_dir
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ Completed: {model} with {method}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {model} with {method}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
        sys.exit(1)


def aggregate_results(output_dir="results_besstie_unified"):
    """Aggregate all results into a single summary file"""
    print(f"\n{'='*80}")
    print("Aggregating results...")
    print(f"{'='*80}\n")

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
                print(f"⚠ Missing results for {model}/{method}")

    # Save aggregated results
    summary_file = os.path.join(output_dir, "all_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Aggregated results saved to: {summary_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("Results Summary Table")
    print(f"{'='*80}\n")
    print(f"{'Model':<15} {'Method':<15} {'Avg Accuracy':<15} {'Avg F1':<15}")
    print(f"{'-'*60}")

    for model in ALL_MODELS:
        for method in ALL_METHODS:
            if all_results[model][method]:
                summary = all_results[model][method]["summary"]
                acc = summary.get("average_accuracy", 0)
                f1 = summary.get("average_f1", 0)
                print(f"{model:<15} {method:<15} {acc:<15.4f} {f1:<15.4f}")
            else:
                print(f"{model:<15} {method:<15} {'N/A':<15} {'N/A':<15}")

    print(f"\n{'='*80}\n")


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
        default="results_besstie_final/zeroshot",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip aggregation step at the end"
    )

    args = parser.parse_args()

    # Create list of all evaluations to run
    evaluations = [
        (model, method)
        for model in args.models
        for method in args.methods
    ]

    total_evals = len(evaluations)

    print(f"\n{'='*80}")
    print(f"Batch Evaluation Runner")
    print(f"{'='*80}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Total evaluations: {total_evals}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Track results
    successful = []
    failed = []

    # Run all evaluations
    start_time = datetime.now()

    for idx, (model, method) in enumerate(evaluations, 1):
        print(f"\n[{idx}/{total_evals}] Starting: {model} with {method}")

        success = run_evaluation(model, method, args.output_dir)

        if success:
            successful.append((model, method))
        else:
            failed.append((model, method))

    end_time = datetime.now()
    duration = end_time - start_time

    # Print final summary
    print(f"\n{'='*80}")
    print(f"Batch Evaluation Complete")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful: {len(successful)}/{total_evals}")
    print(f"Failed: {len(failed)}/{total_evals}")

    if failed:
        print(f"\nFailed evaluations:")
        for model, method in failed:
            print(f"  - {model} with {method}")

    print(f"{'='*80}\n")

    # Aggregate results
    if not args.skip_aggregation:
        aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
