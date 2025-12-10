#!/usr/bin/env python3
import csv
import os

def calculate_weighted_average(csv_file):
    results = {
        'P': [],
        'R': [],
        'F1': [],
        'correct': [],
        'len': [],
        'avg_edit_dist': [],
        'avg_norm_edit_dist': [],
        'Time': []
    }
    
    total_len = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            len_val = int(row['len'])
            total_len += len_val
            
            results['P'].append((float(row['P']), len_val))
            results['R'].append((float(row['R']), len_val))
            results['F1'].append((float(row['F1']), len_val))
            results['correct'].append((int(row['correct']), len_val))
            results['len'].append(len_val)
            results['avg_edit_dist'].append((float(row['avg_edit_dist']), len_val))
            results['avg_norm_edit_dist'].append((float(row['avg_norm_edit_dist']), len_val))
            results['Time'].append((float(row['Time']), len_val))
    
    weighted_metrics = {}
    for key in ['P', 'R', 'F1', 'avg_edit_dist', 'avg_norm_edit_dist', 'Time']:
        weighted_sum = sum(val * weight for val, weight in results[key])
        weighted_metrics[key] = weighted_sum / total_len if total_len > 0 else 0
    
    weighted_metrics['correct'] = sum(val for val, _ in results['correct'])
    weighted_metrics['len'] = sum(results['len'])
    
    weighted_metrics['Acc'] = weighted_metrics['correct'] / weighted_metrics['len'] if weighted_metrics['len'] > 0 else 0
    
    return weighted_metrics

def main():
    result_dir = "baseline/dtt/dtt_result"
    output_file = "baseline/dtt/dtt_result/dataset_metrics_summary.txt"
    datasets = ['autofj', 'ss', 'wt', 'kbwt']
    
    output_lines = []
    
    def output(text):
        print(text)
        output_lines.append(text)
    
    output("=" * 80)
    output("DATASET METRICS SUMMARY (Weighted by len)")
    output("=" * 80)
    output("")
    
    for dataset in datasets:
        csv_file = os.path.join(result_dir, f"{dataset}.csv")
        if not os.path.exists(csv_file):
            output(f"Warning: {csv_file} not found, skipping...")
            continue
        
        metrics = calculate_weighted_average(csv_file)
        
        output("=" * 80)
        output(f"DATASET: {dataset.upper()}")
        output("=" * 80)
        output("")
        output(f"{'Metric':<25} {'Value':<15} {'Description'}")
        output("-" * 80)
        output(f"{'Accuracy (Acc)':<25} {metrics['Acc']:<15.4f} {'correct/len'}")
        output(f"{'Precision (P)':<25} {metrics['P']:<15.4f} {'weighted by len'}")
        output(f"{'Recall (R)':<25} {metrics['R']:<15.4f} {'weighted by len'}")
        output(f"{'F1 Score':<25} {metrics['F1']:<15.4f} {'weighted by len'}")
        output(f"{'Correct':<25} {metrics['correct']:<15.0f} {'total correct'}")
        output(f"{'Total (len)':<25} {metrics['len']:<15.0f} {'total samples'}")
        output(f"{'Avg Edit Dist':<25} {metrics['avg_edit_dist']:<15.4f} {'weighted by len'}")
        output(f"{'Avg Norm Edit Dist':<25} {metrics['avg_norm_edit_dist']:<15.4f} {'weighted by len'}")
        output(f"{'Time (seconds)':<25} {metrics['Time']:<15.4f} {'weighted by len'}")
        output("")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nResult saved to: {output_file}")

if __name__ == "__main__":
    main()

