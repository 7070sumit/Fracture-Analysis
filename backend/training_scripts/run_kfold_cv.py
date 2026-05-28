"""
K-Fold Cross Validation — OAI Fracture Detection
=================================================
Trains the model K times on different data splits.
Each fold is an independent test — this proves the model generalizes.

For research paper: reports Mean ± Std across all folds.

Usage:
  backend/venv/bin/python backend/run_kfold_cv.py --folds 5 --epochs 10
"""

import logging
import argparse
import json
import numpy as np
import sys, os
sys.path.insert(0, "backend")
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kfold_cv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation for OAI Fracture Detection")
    parser.add_argument("--folds",      type=int,   default=5,      help="Number of folds (default: 5)")
    parser.add_argument("--epochs",     type=int,   default=10,     help="Epochs per fold (default: 10)")
    parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size")
    parser.add_argument("--lr",         type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--data-dir",   type=str,
        default="/Users/piyushkumar/Project/Fracture-Analysis-main/backend/datasets/OAI_Processed",
        help="Path to OAI dataset")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  {args.folds}-Fold Cross Validation")
    print(f"  Epochs per fold : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print("=" * 65)

    manager = ModelManager(data_dir=args.data_dir)

    try:
        results = manager.run_kfold_cv(
            k_folds=args.folds,
            epochs_per_fold=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        if results:
            # ── Extract per-fold metrics ──────────────────────────────────────
            fold_accs = results.get("fold_accuracies", [])
            mean_acc  = results.get("mean_accuracy",  0)
            std_acc   = results.get("std_accuracy",   0)

            print()
            print("=" * 65)
            print("  K-FOLD CROSS VALIDATION RESULTS")
            print("=" * 65)
            for i, acc in enumerate(fold_accs):
                print(f"  Fold {i+1}: Balanced Acc = {acc*100:.2f}%")
            print("-" * 65)
            print(f"  Mean Balanced Accuracy : {mean_acc*100:.2f}%")
            print(f"  Std  Balanced Accuracy : {std_acc*100:.2f}%")
            print(f"  95% CI                 : [{(mean_acc - 1.96*std_acc)*100:.2f}%, {(mean_acc + 1.96*std_acc)*100:.2f}%]")
            print("=" * 65)
            print()
            print("  For your research paper Methods section:")
            print(f"  'The model achieved a mean balanced accuracy of")
            print(f"   {mean_acc*100:.1f}% ± {std_acc*100:.1f}% across {args.folds}-fold cross-validation,'")
            print(f"   demonstrating robust generalizability across patient subsets.'")
            print()

            # Save results
            out_path = "backend/models/kfold_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {out_path}")

        else:
            logger.error("K-Fold CV returned no results.")

    except Exception as e:
        logger.error(f"K-Fold CV failed: {e}")
        raise
