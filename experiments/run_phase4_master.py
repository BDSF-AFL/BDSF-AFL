"""BDSF-AFL Phase 4: Master CLI Controller & Orchestration Suite.

Provides a unified interface for:
  1. Fast Smoke-Testing and Pipeline Validation (--smoke-test)
  2. Baseline Benchmark Comparisons (--benchmarks)
  3. 7-Way Component Ablation Matrix (--ablations)
  4. Publication Figure and LaTeX/Markdown Table Generation (--generate-artifacts)
"""

import sys
import os
import argparse
import subprocess

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.generate_paper_figures import generate_all_publication_figures
from experiments.generate_report_tables import generate_all_report_tables
from experiments.run_benchmarks import run_benchmarks_from_manifest
from experiments.run_ablation_matrix import run_ablations_from_manifest


def main():
    parser = argparse.ArgumentParser(description="BDSF-AFL Phase 4 Master Orchestration CLI")
    parser.add_argument("--smoke-test", action="store_true", help="Runs fast mock verification and artifact generation")
    parser.add_argument("--benchmarks", action="store_true", help="Runs baseline and proposed benchmark comparisons")
    parser.add_argument("--ablations", action="store_true", help="Runs 7-way component ablation matrix")
    parser.add_argument("--generate-artifacts", action="store_true", help="Generates publication figures and LaTeX/MD tables")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of training rounds")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override list of random seeds")
    parser.add_argument("--benchmark-manifest", type=str, default="experiments/manifests/benchmark_matrix.yaml")
    parser.add_argument("--ablation-manifest", type=str, default="experiments/manifests/ablation_matrix.yaml")
    args = parser.parse_args()

    # Default action if no flags provided
    if not (args.smoke_test or args.benchmarks or args.ablations or args.generate_artifacts):
        print("\n[INFO] No action specified. Running --smoke-test and generating all publication artifacts...\n")
        args.smoke_test = True
        args.generate_artifacts = True

    if args.smoke_test:
        print("\n=======================================================")
        print("RUNNING PHASE 4 FAST PIPELINE VERIFICATION & SMOKE TEST")
        print("=======================================================\n")
        # Run scratch verification
        verify_script = os.path.join("scratch", "verify_phase4_pipeline.py")
        if os.path.exists(verify_script):
            ret = subprocess.call([sys.executable, verify_script])
            if ret != 0:
                print(f"[ERROR] Smoke test failed with exit code {ret}")
                sys.exit(ret)

    if args.benchmarks:
        print("\n[RUNNING] Baseline Benchmark Suite...")
        run_benchmarks_from_manifest(
            manifest_path=args.benchmark_manifest,
            seeds=args.seeds,
            rounds=args.rounds
        )

    if args.ablations:
        print("\n[RUNNING] 7-Way Ablation Matrix Suite...")
        run_ablations_from_manifest(
            manifest_path=args.ablation_manifest,
            seeds=args.seeds,
            rounds=args.rounds
        )

    if args.generate_artifacts or args.smoke_test:
        print("\n[GENERATING] Publication Figures & Tables...")
        generate_all_publication_figures()
        generate_all_report_tables()

    print("\n=======================================================")
    print("PHASE 4 ORCHESTRATION COMPLETE")
    print("Figures : logs/phase4_results/artifacts/figures/")
    print("Tables  : logs/phase4_results/artifacts/tables/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
