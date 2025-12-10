#!/usr/bin/env python3
"""
AutoFJ batch processing script
Process four datasets in data directory: autofj, kbwt, ss, wt
"""

import os
import pandas as pd
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import time

try:
    from autofj import AutoFJ
except ImportError:
    print("Error: autofj package not installed")
    print("Please run: pip install autofj")
    sys.exit(1)


def load_dataset_autofj(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load autofj format dataset (left.csv, right.csv, gt.csv)"""
    left_file = dataset_path / "left.csv"
    right_file = dataset_path / "right.csv"
    gt_file = dataset_path / "gt.csv"
    
    if not left_file.exists() or not right_file.exists():
        return None, None, None
    
    left_table = pd.read_csv(left_file)
    right_table = pd.read_csv(right_file)
    gt_table = pd.read_csv(gt_file) if gt_file.exists() else None
    
    # Ensure id column exists
    if "id" not in left_table.columns:
        left_table.insert(0, "id", range(len(left_table)))
    if "id" not in right_table.columns:
        right_table.insert(0, "id", range(len(right_table)))
    
    return left_table, right_table, gt_table


def load_dataset_kbwt_ss_wt(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load kbwt/ss/wt format dataset (source.csv, target.csv, ground truth.csv)"""
    source_file = dataset_path / "source.csv"
    target_file = dataset_path / "target.csv"
    gt_file = dataset_path / "ground truth.csv"
    rows_file = dataset_path / "rows.txt"
    
    if not source_file.exists() or not target_file.exists():
        return None, None, None
    
    source_table = pd.read_csv(source_file)
    target_table = pd.read_csv(target_file)
    gt_table = pd.read_csv(gt_file) if gt_file.exists() else None
    
    # Parse rows.txt to determine columns to join
    source_data_col = None
    target_data_col = None
    if rows_file.exists():
        try:
            with open(rows_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) > 0 and ':' in lines[0]:
                parts = lines[0].split(':')
                if len(parts) == 2:
                    source_data_col = parts[0].strip()
                    target_data_col = parts[1].strip()
        except Exception as e:
            print(f"  Warning: Unable to parse rows.txt: {e}")
    
    # If column names not obtained from rows.txt, use first non-id column
    if not source_data_col:
        source_data_cols = [c for c in source_table.columns if c != "id"]
        source_data_col = source_data_cols[0] if len(source_data_cols) > 0 else None
    
    if not target_data_col:
        target_data_cols = [c for c in target_table.columns if c != "id"]
        target_data_col = target_data_cols[0] if len(target_data_cols) > 0 else None
    
    # Add id column for datasets without id column (at first position)
    if "id" not in source_table.columns:
        source_table.insert(0, "id", range(len(source_table)))
    else:
        # Ensure id column is in first position
        if source_table.columns[0] != "id":
            cols = source_table.columns.tolist()
            cols.remove("id")
            source_table = source_table[["id"] + cols]
    
    if "id" not in target_table.columns:
        target_table.insert(0, "id", range(len(target_table)))
    else:
        if target_table.columns[0] != "id":
            cols = target_table.columns.tolist()
            cols.remove("id")
            target_table = target_table[["id"] + cols]
    
    # Ensure id column is integer type
    source_table["id"] = source_table["id"].astype(int)
    target_table["id"] = target_table["id"].astype(int)
    
    # Unify column name to "value" for joining
    if source_data_col and source_data_col in source_table.columns:
        # Ensure join column is converted to string type (AutoFJ requires string type)
        source_table[source_data_col] = source_table[source_data_col].astype(str)
        source_table.rename(columns={source_data_col: "value"}, inplace=True)
    
    if target_data_col and target_data_col in target_table.columns:
        # Ensure join column is converted to string type (AutoFJ requires string type)
        target_table[target_data_col] = target_table[target_data_col].astype(str)
        target_table.rename(columns={target_data_col: "value"}, inplace=True)
    
    # Final verification: ensure id column exists, is in first position, and is integer type
    # Check again after renaming columns to ensure id column was not accidentally removed
    if "id" not in source_table.columns:
        source_table.insert(0, "id", range(len(source_table)))
    elif source_table.columns[0] != "id":
        # If id column is not in first position, move it to first position
        cols = [c for c in source_table.columns if c != "id"]
        source_table = source_table[["id"] + cols]
    
    if "id" not in target_table.columns:
        target_table.insert(0, "id", range(len(target_table)))
    elif target_table.columns[0] != "id":
        cols = [c for c in target_table.columns if c != "id"]
        target_table = target_table[["id"] + cols]
    
    # Ensure id column is integer type (confirm again at the end)
    source_table["id"] = source_table["id"].astype(int)
    target_table["id"] = target_table["id"].astype(int)
    
    return source_table, target_table, gt_table


def process_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    precision_target: float = 0.9,
    verbose: bool = False
) -> dict:
    """Process a single dataset"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*60}")
    
    # Select loading function based on dataset type
    if dataset_path.parent.name == "autofj":
        left_table, right_table, gt_table = load_dataset_autofj(dataset_path)
    else:
        left_table, right_table, gt_table = load_dataset_kbwt_ss_wt(dataset_path)
    
    if left_table is None or right_table is None:
        print(f"  Skipped: Unable to load dataset files")
        return {"status": "skipped", "reason": "Unable to load files"}
    
    # Data validation
    if len(left_table) == 0:
        print(f"  Skipped: Left table is empty")
        return {"status": "skipped", "reason": "Left table is empty"}
    if len(right_table) == 0:
        print(f"  Skipped: Right table is empty")
        return {"status": "skipped", "reason": "Right table is empty"}
    
    # Check if there are common columns (besides id)
    common_cols = set(left_table.columns) & set(right_table.columns)
    common_cols.discard("id")
    if len(common_cols) == 0:
        print(f"  Skipped: Two tables have no common columns (besides id)")
        return {"status": "skipped", "reason": "No common columns"}
    
    print(f"  Left table size: {len(left_table)} rows, {len(left_table.columns)} columns")
    print(f"  Right table size: {len(right_table)} rows, {len(right_table.columns)} columns")
    print(f"  Common columns: {list(common_cols)}")
    print(f"  Left column names: {list(left_table.columns)}")
    print(f"  Right column names: {list(right_table.columns)}")
    
    # Determine columns to join
    # According to documentation, autofj datasets use "title" column
    # For other datasets, use all common columns (besides id)
    if dataset_path.parent.name == "autofj":
        # autofj dataset: prioritize "title" column
        if "title" in common_cols:
            join_columns = ["title"]
            print(f"  Using columns for join: {join_columns} (autofj standard)")
        else:
            # If no title column, use all common columns
            join_columns = list(common_cols)
            print(f"  Using columns for join: {join_columns} (title column not found, using all common columns)")
    else:
        # Other datasets: use all common columns
        join_columns = list(common_cols)
        print(f"  Using columns for join: {join_columns}")
    
    # Data cleaning and validation
    # Important: minimize DataFrame modifications, let AutoFJ handle most of the work
    # AutoFJ internally renames id column to autofj_id, so we only need to ensure id column exists
    
    # 1. Ensure id column exists and is integer type
    if "id" not in left_table.columns:
        left_table.insert(0, "id", range(len(left_table)))
    else:
        # Ensure id column is integer type
        if left_table["id"].dtype != 'int64':
            left_table["id"] = pd.to_numeric(left_table["id"], errors='coerce').fillna(0).astype(int)
    
    if "id" not in right_table.columns:
        right_table.insert(0, "id", range(len(right_table)))
    else:
        if right_table["id"].dtype != 'int64':
            right_table["id"] = pd.to_numeric(right_table["id"], errors='coerce').fillna(0).astype(int)
    
    # 2. Check and fix duplicate ids (if any)
    if left_table["id"].duplicated().any():
        print(f"  Warning: Left table id column has duplicate values, regenerating")
        left_table["id"] = range(len(left_table))
        left_table["id"] = left_table["id"].astype(int)
    if right_table["id"].duplicated().any():
        print(f"  Warning: Right table id column has duplicate values, regenerating")
        right_table["id"] = range(len(right_table))
        right_table["id"] = right_table["id"].astype(int)
    
    # 3. Handle NaN values and ensure data type is string (only for join columns, avoid affecting id column)
    # AutoFJ requires string type for fuzzy matching
    # Use inplace=True to avoid creating new DataFrame
    for col in join_columns:
        if col in left_table.columns:
            # Convert to string type, handle NaN
            left_table[col] = left_table[col].fillna("").astype(str).str.strip()
        if col in right_table.columns:
            # Convert to string type, handle NaN
            right_table[col] = right_table[col].fillna("").astype(str).str.strip()
    
    print(f"  Ready to process - Left table: {len(left_table)} rows, Right table: {len(right_table)} rows")
    print(f"  Left table columns: {list(left_table.columns)}")
    print(f"  Right table columns: {list(right_table.columns)}")
    
    # Final verification: ensure id column exists and is integer type
    if "id" not in left_table.columns:
        print(f"  Error: Left table missing id column, attempting to add")
        left_table.insert(0, "id", range(len(left_table)))
        left_table["id"] = left_table["id"].astype(int)
    if "id" not in right_table.columns:
        print(f"  Error: Right table missing id column, attempting to add")
        right_table.insert(0, "id", range(len(right_table)))
        right_table["id"] = right_table["id"].astype(int)
    
    # Ensure id column is integer type
    left_table["id"] = left_table["id"].astype(int)
    right_table["id"] = right_table["id"].astype(int)
    
    print(f"  Left id column type: {left_table['id'].dtype}, unique: {left_table['id'].is_unique}")
    print(f"  Right id column type: {right_table['id'].dtype}, unique: {right_table['id'].is_unique}")
    
    # Final check before calling AutoFJ
    if "id" not in left_table.columns or "id" not in right_table.columns:
        error_msg = f"id column missing: left={'id' in left_table.columns}, right={'id' in right_table.columns}"
        print(f"  Error: {error_msg}")
        return {"status": "error", "error": error_msg}
    
    try:
        # Create AutoFJ instance
        fj = AutoFJ(precision_target=precision_target, verbose=verbose)
        
        # Execute join
        # According to documentation: if on=None, will join all common columns (besides id)
        # If on is specified, only join specified columns
        # Note: AutoFJ internally renames id column to autofj_id
        start_time = time.time()
        
        # For all datasets, let AutoFJ handle automatically (do not specify on parameter)
        # AutoFJ will automatically use all common columns (besides id) for joining
        # This avoids potential issues when manually specifying column names
        result = fj.join(left_table, right_table, "id")
        
        elapsed_time = time.time() - start_time
        
        print(f"  Join completed! Time elapsed: {elapsed_time:.2f} seconds")
        print(f"  Result size: {len(result)} rows")
        
        # Save results (replace path separators)
        safe_name = dataset_name.replace("/", "_")
        output_file = output_dir / f"{safe_name}_result.csv"
        result.to_csv(output_file, index=False)
        print(f"  Results saved to: {output_file}")
        
        # If ground truth exists, calculate evaluation metrics
        metrics = {}
        if gt_table is not None:
            metrics = evaluate_results(result, gt_table, dataset_path, left_table, right_table)
            if metrics:
                print(f"  Evaluation metrics:")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall: {metrics.get('recall', 0):.4f}")
                print(f"    F1: {metrics.get('f1', 0):.4f}")
                print(f"    TP: {metrics.get('true_positives', 0)}, FP: {metrics.get('false_positives', 0)}, FN: {metrics.get('false_negatives', 0)}")
            else:
                print(f"  Unable to calculate evaluation metrics (ground truth format may not match)")
        
        return {
            "status": "success",
            "result_size": len(result),
            "elapsed_time": elapsed_time,
            "metrics": metrics,
            "output_file": str(output_file)
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"  Error: {error_msg}")
        import traceback
        full_traceback = traceback.format_exc()
        print(full_traceback)
        
        # Save error log
        error_log_dir = output_dir / "error_logs"
        error_log_dir.mkdir(exist_ok=True, parents=True)
        safe_name = dataset_name.replace("/", "_")
        error_log_file = error_log_dir / f"{safe_name}_error.txt"
        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Path: {dataset_path}\n")
            f.write(f"Error message: {error_msg}\n")
            f.write(f"\nFull stack trace:\n{full_traceback}\n")
        
        return {
            "status": "error", 
            "error": error_msg,
            "error_file": str(error_log_file)
        }


def evaluate_results(result: pd.DataFrame, gt_table: pd.DataFrame, dataset_path: Path, 
                     left_table: pd.DataFrame, right_table: pd.DataFrame) -> dict:
    """Evaluate results (if ground truth exists)"""
    try:
        # Extract matching pairs from results
        if "id_l" not in result.columns or "id_r" not in result.columns:
            return {}
        
        result_pairs = set(zip(result["id_l"], result["id_r"]))
        
        # Determine ground truth format based on dataset type
        if dataset_path.parent.name == "autofj":
            # autofj format: id_l, id_r
            if "id_l" in gt_table.columns and "id_r" in gt_table.columns:
                gt_pairs = set(zip(gt_table["id_l"], gt_table["id_r"]))
            else:
                return {}
        else:
            # kbwt/ss/wt format: ground truth format is "source-{colname},target-{colname}"
            # Need to find corresponding ids through value matching
            if len(gt_table.columns) < 2:
                return {}
            
            col1, col2 = gt_table.columns[0], gt_table.columns[1]
            
            # For kbwt/ss/wt datasets, we have unified column names to "value"
            # So directly use "value" column to create mapping
            left_col = "value" if "value" in left_table.columns else None
            right_col = "value" if "value" in right_table.columns else None
            
            if not left_col or not right_col:
                # If "value" column doesn't exist, try to extract original column name from ground truth column names
                left_col = col1.replace("source-", "").replace("target-", "")
                right_col = col2.replace("source-", "").replace("target-", "")
                # If extracted column names also don't exist, return empty
                if left_col not in left_table.columns or right_col not in right_table.columns:
                    return {}
            
            # Create value to index mapping
            # Note: id_l and id_r returned by AutoFJ correspond to DataFrame index positions, not the values of the id column we added
            # So we need to use index positions to build the mapping
            left_value_to_idx = {}
            if left_col in left_table.columns:
                for idx, val in left_table[left_col].items():
                    if pd.notna(val):
                        # Use index position, not id column value
                        left_value_to_idx[str(val).strip()] = idx
            
            right_value_to_idx = {}
            if right_col in right_table.columns:
                for idx, val in right_table[right_col].items():
                    if pd.notna(val):
                        # Use index position, not id column value
                        right_value_to_idx[str(val).strip()] = idx
            
            # Build ground truth matching pairs (using index positions)
            gt_pairs = set()
            for _, row in gt_table.iterrows():
                val1 = str(row[col1]).strip() if pd.notna(row[col1]) else None
                val2 = str(row[col2]).strip() if pd.notna(row[col2]) else None
                
                if val1 and val2 and val1 in left_value_to_idx and val2 in right_value_to_idx:
                    gt_pairs.add((left_value_to_idx[val1], right_value_to_idx[val2]))
        
        if len(gt_pairs) == 0:
            return {}
        
        # Calculate metrics
        true_positives = len(gt_pairs & result_pairs)
        false_positives = len(result_pairs - gt_pairs)
        false_negatives = len(gt_pairs - result_pairs)
        
        precision = true_positives / len(result_pairs) if len(result_pairs) > 0 else 0
        recall = true_positives / len(gt_pairs) if len(gt_pairs) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "gt_size": len(gt_pairs),
            "result_size": len(result_pairs)
        }
    except Exception as e:
        print(f"    Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AutoFJ batch processing script")
    parser.add_argument("--precision", type=float, default=0.9, 
                       help="Precision target (default: 0.9)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show verbose logs")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Only process specified dataset type (autofj, kbwt, ss, wt)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: results/)")
    
    args = parser.parse_args()
    
    # Set paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = Path(args.output) if args.output else base_dir / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dataset directories
    all_dataset_dirs = {
        # "autofj": data_dir / "autofj",
        "kbwt": data_dir / "kbwt",
        # "ss": data_dir / "ss",
        "wt": data_dir / "wt"
    }
    
    # If dataset type is specified, only process that type
    if args.dataset:
        if args.dataset not in all_dataset_dirs:
            print(f"Error: Unknown dataset type '{args.dataset}'")
            print(f"Available types: {', '.join(all_dataset_dirs.keys())}")
            sys.exit(1)
        dataset_dirs = {args.dataset: all_dataset_dirs[args.dataset]}
    else:
        dataset_dirs = all_dataset_dirs
    
    # Configuration parameters
    precision_target = args.precision
    verbose = args.verbose
    
    # Statistics
    all_results = {}
    total_datasets = 0
    successful = 0
    failed = 0
    skipped = 0
    
    print("="*60)
    print("AutoFJ Batch Processing Script")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Precision Target: {precision_target}")
    print("="*60)
    
    # Process each dataset directory
    for dataset_type, dataset_dir in dataset_dirs.items():
        if not dataset_dir.exists():
            print(f"\nWarning: Dataset directory does not exist: {dataset_dir}")
            continue
        
        print(f"\nProcessing dataset type: {dataset_type}")
        print(f"Directory: {dataset_dir}")
        
        # Get all subdatasets
        subdatasets = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"Found {len(subdatasets)} subdatasets")
        
        for subdataset_path in sorted(subdatasets):
            subdataset_name = f"{dataset_type}/{subdataset_path.name}"
            total_datasets += 1
            
            result = process_dataset(
                subdataset_name,
                subdataset_path,
                output_dir,
                precision_target=precision_target,
                verbose=verbose
            )
            
            all_results[subdataset_name] = result
            
            if result["status"] == "success":
                successful += 1
            elif result["status"] == "error":
                failed += 1
            else:
                skipped += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    print(f"Total datasets: {total_datasets}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    # Save summary results
    summary_file = output_dir / "summary.csv"
    summary_data = []
    
    # Define all possible columns (ensure all rows have the same columns)
    metric_columns = [
        "metric_precision", "metric_recall", "metric_f1",
        "metric_true_positives", "metric_false_positives", "metric_false_negatives",
        "metric_gt_size", "metric_result_size"
    ]
    
    for name, result in all_results.items():
        row = {
            "dataset": name,
            "status": result["status"],
            "result_size": result.get("result_size", 0),
            "elapsed_time": result.get("elapsed_time", 0),
        }
        # Add error information
        if result["status"] == "error":
            row["error"] = result.get("error", "Unknown error")
            row["error_file"] = result.get("error_file", "")
        else:
            row["error"] = ""
            row["error_file"] = ""
        # Add skip reason
        if result["status"] == "skipped":
            row["reason"] = result.get("reason", "")
        # Add evaluation metrics (set to empty if not available)
        if "metrics" in result and result["metrics"]:
            row.update({f"metric_{k}": v for k, v in result["metrics"].items()})
        else:
            # Ensure all metric columns exist (even if no values)
            for col in metric_columns:
                row[col] = ""
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    # Ensure column order is consistent
    base_columns = ["dataset", "status", "result_size", "elapsed_time"]
    all_columns = base_columns + metric_columns + ["error", "error_file"]
    # Only keep columns that actually exist
    existing_columns = [col for col in all_columns if col in summary_df.columns]
    summary_df = summary_df[existing_columns]
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")
    
    # If there are errors, print error summary
    if failed > 0:
        print(f"\nError dataset list:")
        for name, result in all_results.items():
            if result["status"] == "error":
                print(f"  - {name}: {result.get('error', 'Unknown error')}")
                if "error_file" in result:
                    print(f"    Detailed error log: {result['error_file']}")


if __name__ == "__main__":
    main()

