#!/usr/bin/env python3
"""Generate formatted metrics report"""

import pandas as pd

# Read metrics summary
df = pd.read_csv('results/metrics_summary.csv')

# Generate report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("DATASET METRICS SUMMARY (Weighted by result_size)")
report_lines.append("=" * 80)
report_lines.append("")

for _, row in df.iterrows():
    dataset_type = row['dataset_type'].upper()
    report_lines.append("=" * 80)
    report_lines.append(f"DATASET: {dataset_type}")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'Metric':<30} {'Value':<20} {'Description'}")
    report_lines.append("-" * 80)
    
    # Accuracy
    report_lines.append(f"{'Accuracy (Acc)':<30} {row['weighted_accuracy']:<20.4f} Weighted average accuracy")
    
    # Precision
    report_lines.append(f"{'Precision (Pre)':<30} {row['weighted_precision']:<20.4f} Weighted average precision")
    
    # Recall
    report_lines.append(f"{'Recall (Rec)':<30} {row['weighted_recall']:<20.4f} Weighted average recall")
    
    # F1
    report_lines.append(f"{'F1 Score (F1)':<30} {row['weighted_f1']:<20.4f} Weighted average F1 score")
    
    report_lines.append("")
    report_lines.append(f"{'Number of Datasets':<30} {int(row['num_datasets']):<20}")
    report_lines.append(f"{'Total Result Size':<30} {int(row['total_result_size']):<20}")
    report_lines.append("")
    report_lines.append(f"{'Total TP':<30} {int(row['total_tp']):<20}")
    report_lines.append(f"{'Total FP':<30} {int(row['total_fp']):<20}")
    report_lines.append(f"{'Total FN':<30} {int(row['total_fn']):<20}")
    report_lines.append(f"{'Total GT':<30} {int(row['total_gt']):<20}")
    report_lines.append("")
    report_lines.append(f"{'Accuracy (TP/GT)':<30} {row['accuracy_method1_tp_over_gt']:<20.4f} TP / GT_size")
    report_lines.append(f"{'Accuracy (TP/(TP+FP+FN))':<30} {row['accuracy_method2_tp_over_all']:<20.4f} TP / (TP + FP + FN)")
    report_lines.append("")
    report_lines.append("")

# Save to file
output_file = 'results/metrics_report.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"Report saved to: {output_file}")
print("\n" + "=" * 80)
print("Report preview:")
print("=" * 80)
print('\n'.join(report_lines[:50]))  # Show first 50 lines preview

