#!/usr/bin/env python3
"""Generate complete metrics report, including autofj, ss and kbwt"""

import pandas as pd
import numpy as np

# Read summary_auto_ss.csv (autofj and ss)
df_auto_ss = pd.read_csv('results/summary_auto_ss.csv')

# Read summary.csv (kbwt and wt)
df_all = pd.read_csv('results/summary.csv')

# Filter each dataset type, exclude error
autofj_df = df_auto_ss[(df_auto_ss['dataset'].str.startswith('autofj/')) & (df_auto_ss['status'] == 'success')].copy()
ss_df = df_auto_ss[(df_auto_ss['dataset'].str.startswith('ss/')) & (df_auto_ss['status'] == 'success')].copy()
kbwt_df = df_all[(df_all['dataset'].str.startswith('kbwt/')) & (df_all['status'] == 'success')].copy()
wt_df = df_all[(df_all['dataset'].str.startswith('wt/')) & (df_all['status'] == 'success')].copy()

def calculate_metrics(df, dataset_type):
    """Calculate weighted average metrics"""
    valid_df = df[
        df['metric_precision'].notna() & 
        df['metric_recall'].notna() & 
        df['metric_f1'].notna()
    ].copy()
    
    if len(valid_df) == 0:
        return None
    
    weights = valid_df['result_size'].values
    
    weighted_precision = np.average(valid_df['metric_precision'].values, weights=weights)
    weighted_recall = np.average(valid_df['metric_recall'].values, weights=weights)
    weighted_f1 = np.average(valid_df['metric_f1'].values, weights=weights)
    
    # Accuracy
    valid_df['accuracy'] = valid_df['metric_true_positives'] / (
        valid_df['metric_true_positives'] + 
        valid_df['metric_false_positives'] + 
        valid_df['metric_false_negatives']
    )
    valid_df['accuracy'] = valid_df['accuracy'].replace([np.inf, -np.inf], np.nan)
    weighted_accuracy = np.average(valid_df['accuracy'].dropna().values, 
                                   weights=weights[valid_df['accuracy'].notna()])
    
    total_tp = valid_df['metric_true_positives'].sum()
    total_fp = valid_df['metric_false_positives'].sum()
    total_fn = valid_df['metric_false_negatives'].sum()
    total_gt = valid_df['metric_gt_size'].sum()
    
    accuracy_method1 = total_tp / total_gt if total_gt > 0 else 0
    accuracy_method2 = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    
    return {
        'dataset_type': dataset_type,
        'num_datasets': len(valid_df),
        'total_result_size': weights.sum(),
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_accuracy': weighted_accuracy,
        'accuracy_method1_tp_over_gt': accuracy_method1,
        'accuracy_method2_tp_over_all': accuracy_method2,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_gt
    }

# Calculate metrics for each dataset
autofj_metrics = calculate_metrics(autofj_df, 'autofj')
ss_metrics = calculate_metrics(ss_df, 'ss')
kbwt_metrics = calculate_metrics(kbwt_df, 'kbwt')
wt_metrics = calculate_metrics(wt_df, 'wt')

# Generate report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("DATASET METRICS SUMMARY (Weighted by result_size)")
report_lines.append("=" * 80)
report_lines.append("")

def add_dataset_section(metrics, name):
    """Add dataset section"""
    if not metrics:
        return
    
    report_lines.append("=" * 80)
    report_lines.append(f"DATASET: {name.upper()}")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'Metric':<30} {'Value':<20}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Accuracy (Acc)':<30} {metrics['weighted_accuracy']:<20.4f}")
    report_lines.append(f"{'Precision (Pre)':<30} {metrics['weighted_precision']:<20.4f}")
    report_lines.append(f"{'Recall (Rec)':<30} {metrics['weighted_recall']:<20.4f}")
    report_lines.append(f"{'F1 Score (F1)':<30} {metrics['weighted_f1']:<20.4f}")
    report_lines.append("")
    report_lines.append(f"{'Number of Datasets':<30} {int(metrics['num_datasets']):<20}")
    report_lines.append(f"{'Total Result Size':<30} {int(metrics['total_result_size']):<20}")
    report_lines.append("")
    report_lines.append(f"{'Total TP':<30} {int(metrics['total_tp']):<20}")
    report_lines.append(f"{'Total FP':<30} {int(metrics['total_fp']):<20}")
    report_lines.append(f"{'Total FN':<30} {int(metrics['total_fn']):<20}")
    report_lines.append(f"{'Total GT':<30} {int(metrics['total_gt']):<20}")
    report_lines.append("")
    report_lines.append(f"{'Accuracy (TP/GT)':<30} {metrics['accuracy_method1_tp_over_gt']:<20.4f}")
    report_lines.append(f"{'Accuracy (TP/(TP+FP+FN))':<30} {metrics['accuracy_method2_tp_over_all']:<20.4f}")
    report_lines.append("")
    report_lines.append("")

# Add sections for each dataset
if autofj_metrics:
    add_dataset_section(autofj_metrics, 'AUTOFJ')

if ss_metrics:
    add_dataset_section(ss_metrics, 'SS')

if kbwt_metrics:
    add_dataset_section(kbwt_metrics, 'KBWT')

if wt_metrics:
    add_dataset_section(wt_metrics, 'WT')

# Calculate average across all datasets
all_metrics = []
if autofj_metrics:
    all_metrics.append(autofj_metrics)
if ss_metrics:
    all_metrics.append(ss_metrics)
if kbwt_metrics:
    all_metrics.append(kbwt_metrics)
if wt_metrics:
    all_metrics.append(wt_metrics)

if len(all_metrics) > 0:
    # Calculate simple average (unweighted)
    avg_accuracy = np.mean([m['weighted_accuracy'] for m in all_metrics])
    avg_precision = np.mean([m['weighted_precision'] for m in all_metrics])
    avg_recall = np.mean([m['weighted_recall'] for m in all_metrics])
    avg_f1 = np.mean([m['weighted_f1'] for m in all_metrics])
    
    # Add average section
    report_lines.append("=" * 80)
    report_lines.append("OVERALL AVERAGE")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"{'Metric':<30} {'Value':<20}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Accuracy (Acc)':<30} {avg_accuracy:<20.4f}")
    report_lines.append(f"{'Precision (Pre)':<30} {avg_precision:<20.4f}")
    report_lines.append(f"{'Recall (Rec)':<30} {avg_recall:<20.4f}")
    report_lines.append(f"{'F1 Score (F1)':<30} {avg_f1:<20.4f}")
    report_lines.append("")

# Save report
with open('results/metrics_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("Generated complete metrics_report.txt")
datasets = []
if autofj_metrics:
    datasets.append("AUTOFJ")
if ss_metrics:
    datasets.append("SS")
if kbwt_metrics:
    datasets.append("KBWT")
if wt_metrics:
    datasets.append("WT")
print(f"Included datasets: {', '.join(datasets)}")

