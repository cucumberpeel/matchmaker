#!/usr/bin/env python3
"""Update metrics_report.txt, add metrics for kbwt dataset"""

import pandas as pd
import numpy as np

# Read summary.csv
df = pd.read_csv('results/summary.csv')

# Filter kbwt datasets, exclude error status
kbwt_df = df[(df['dataset'].str.startswith('kbwt/')) & (df['status'] == 'success')].copy()

# Filter out datasets with empty metrics
kbwt_valid = kbwt_df[
    kbwt_df['metric_precision'].notna() & 
    kbwt_df['metric_recall'].notna() & 
    kbwt_df['metric_f1'].notna()
].copy()

if len(kbwt_valid) == 0:
    print("Warning: No valid kbwt metrics data")
    exit(1)

# Use result_size as weights
weights = kbwt_valid['result_size'].values

# Calculate weighted average
weighted_precision = np.average(kbwt_valid['metric_precision'].values, weights=weights)
weighted_recall = np.average(kbwt_valid['metric_recall'].values, weights=weights)
weighted_f1 = np.average(kbwt_valid['metric_f1'].values, weights=weights)

# Calculate accuracy (weighted average)
kbwt_valid['accuracy'] = kbwt_valid['metric_true_positives'] / (
    kbwt_valid['metric_true_positives'] + 
    kbwt_valid['metric_false_positives'] + 
    kbwt_valid['metric_false_negatives']
)
kbwt_valid['accuracy'] = kbwt_valid['accuracy'].replace([np.inf, -np.inf], np.nan)
weighted_accuracy = np.average(kbwt_valid['accuracy'].dropna().values, 
                               weights=weights[kbwt_valid['accuracy'].notna()])

# Total statistics
total_tp = kbwt_valid['metric_true_positives'].sum()
total_fp = kbwt_valid['metric_false_positives'].sum()
total_fn = kbwt_valid['metric_false_negatives'].sum()
total_gt = kbwt_valid['metric_gt_size'].sum()

accuracy_method1 = total_tp / total_gt if total_gt > 0 else 0
accuracy_method2 = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

# Read existing report file
with open('results/metrics_report.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find insertion position (after SS dataset)
insert_pos = len(lines)
for i, line in enumerate(lines):
    if line.startswith('DATASET: SS'):
        # Find the end position of SS dataset
        for j in range(i, len(lines)):
            if j > i and lines[j].startswith('=') and len(lines[j].strip()) > 10:
                insert_pos = j
                break
        break

# Generate KBWT dataset report section
kbwt_report = []
kbwt_report.append("")
kbwt_report.append("=" * 80)
kbwt_report.append("DATASET: KBWT")
kbwt_report.append("=" * 80)
kbwt_report.append("")
kbwt_report.append(f"{'Metric':<30} {'Value':<20} {'Description'}")
kbwt_report.append("-" * 80)
kbwt_report.append(f"{'Accuracy (Acc)':<30} {weighted_accuracy:<20.4f} Weighted average accuracy")
kbwt_report.append(f"{'Precision (Pre)':<30} {weighted_precision:<20.4f} Weighted average precision")
kbwt_report.append(f"{'Recall (Rec)':<30} {weighted_recall:<20.4f} Weighted average recall")
kbwt_report.append(f"{'F1 Score (F1)':<30} {weighted_f1:<20.4f} Weighted average F1 score")
kbwt_report.append("")
kbwt_report.append(f"{'Number of Datasets':<30} {int(len(kbwt_valid)):<20}")
kbwt_report.append(f"{'Total Result Size':<30} {int(weights.sum()):<20}")
kbwt_report.append("")
kbwt_report.append(f"{'Total TP':<30} {int(total_tp):<20}")
kbwt_report.append(f"{'Total FP':<30} {int(total_fp):<20}")
kbwt_report.append(f"{'Total FN':<30} {int(total_fn):<20}")
kbwt_report.append(f"{'Total GT':<30} {int(total_gt):<20}")
kbwt_report.append("")
kbwt_report.append(f"{'Accuracy (TP/GT)':<30} {accuracy_method1:<20.4f} TP / GT_size")
kbwt_report.append(f"{'Accuracy (TP/(TP+FP+FN))':<30} {accuracy_method2:<20.4f} TP / (TP + FP + FN)")
kbwt_report.append("")
kbwt_report.append("")

# Insert into file
new_lines = lines[:insert_pos] + [line + '\n' if not line.endswith('\n') else line for line in kbwt_report] + lines[insert_pos:]

# Save updated file
with open('results/metrics_report.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Updated metrics_report.txt, added KBWT dataset metrics")
print(f"KBWT dataset count: {len(kbwt_valid)}")
print(f"Weighted average Precision: {weighted_precision:.4f}")
print(f"Weighted average Recall: {weighted_recall:.4f}")
print(f"Weighted average F1: {weighted_f1:.4f}")
print(f"Weighted average Accuracy: {weighted_accuracy:.4f}")

