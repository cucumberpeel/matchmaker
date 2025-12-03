import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split

# def read_datasets_old():
#     formatted_datasets = []
#     dataset_paths = ['../data/Hospital/gt.csv', '../data/Country/gt.csv']

#     for dataset_path in dataset_paths:
#         raw_dataset = pd.read_csv(dataset_path)
#         source_col = 'title_l'
#         target_col = 'title_r'
#         all_targets = raw_dataset['title_r'].unique().tolist()
#         pairs = []
#         for _, row in raw_dataset.iterrows():
#             pairs.append({
#                 'source_value': row[source_col],
#                 'gold_value': row[target_col],
#                 'target_values': all_targets,
#             })
#         formatted_datasets.append({'source_column': source_col,
#                                    'target_column': target_col,
#                                    'pairs': pairs})

#     return formatted_datasets

def read_datasets():
    formatted_datasets = []

    autofj_datasets_path = os.path.join("data", "autofj")
    autofj_count = 0
    for root, dir, files in os.walk(autofj_datasets_path):
        gt_files = [f for f in files if f == 'gt.csv']
        if len(gt_files) > 1:
            print(f"Error: more than 1 ground truth file in this directory: {gt_files}")
            return []
        if gt_files:
            dataset_path = os.path.join(root, gt_files[0])
            try:
                raw_dataset = pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Error reading {dataset_path}: {e}")
                return []
            source_col = 'title_l'
            target_col = 'title_r'
            all_targets = raw_dataset[target_col].unique().tolist()

            for _, row in raw_dataset.iterrows():
                formatted_datasets.append({
                    'source_column': source_col,
                    'target_column': target_col,
                    'source_value': row[source_col],
                    'gold_value': row[target_col],
                    'target_values': all_targets,
                })
    print(f"Read {autofj_count} datasets from {autofj_datasets_path}")
    
    ss_datasets_path = os.path.join("data", "ss")
    ss_count = 0
    for root, dir, files in os.walk(ss_datasets_path):
        gt_files = [f for f in files if f == 'ground truth.csv']
        if len(gt_files) > 1:
            print(f"Error: more than 1 ground truth file in this directory: {gt_files}")
            return []
        if gt_files:
            ss_count += 1
            dataset_path = os.path.join(root, gt_files[0])
            try:
                raw_dataset = pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Error reading {dataset_path}: {e}")
                return []
            source_col = 'source-value'
            target_col = 'target-value'
            all_targets = raw_dataset[target_col].unique().tolist()

            for _, row in raw_dataset.iterrows():
                formatted_datasets.append({
                    'source_column': source_col,
                    'target_column': target_col,
                    'source_value': row[source_col],
                    'gold_value': row[target_col],
                    'target_values': all_targets,
                })
    print(f"Read {ss_count} datasets from {ss_datasets_path}")
    
    kbwt_datasets_path = os.path.join("data", "kbwt")
    kbwt_count = 0
    for root, dir, files in os.walk(kbwt_datasets_path):
        gt_files = [f for f in files if f == 'ground truth.csv']
        if len(gt_files) > 1:
            print(f"Error: more than 1 ground truth file in this directory: {gt_files}")
            return []
        if gt_files:
            kbwt_count += 1
            dataset_path = os.path.join(root, gt_files[0])
            try:
                raw_dataset = pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Error reading {dataset_path}: {e}")
                return []

            source_cols = [col for col in raw_dataset.columns if col.startswith("source")]
            target_cols = [col for col in raw_dataset.columns if col.startswith("target")]
            if len(source_cols) > 1 or len(target_cols) > 1:
                print(f"Error locating source-target columns: {raw_dataset.columns}")
                return []
            source_col = source_cols[0]
            target_col = target_cols[0]
            all_targets = raw_dataset[target_col].unique().tolist()

            for _, row in raw_dataset.iterrows():
                formatted_datasets.append({
                    'source_column': source_col,
                    'target_column': target_col,
                    'source_value': row[source_col],
                    'gold_value': row[target_col],
                    'target_values': all_targets,
                })
    print(f"Read {kbwt_count} datasets from {kbwt_datasets_path}")

    return formatted_datasets


def split_datasets(datasets, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(datasets, test_size=test_size, random_state=random_state)
    return train_set, test_set

def print_datasets_info(datasets):
    print("Previewing datasets...")
    for dataset in datasets:
        print(f"Source Column: {dataset['source_column']}")
        print(f"Target Column: {dataset['target_column']}")

        # for read_datasets_old
        print(f"Number of Pairs: {len(dataset['pairs'])}")
        for pair in dataset['pairs']:
            print(f"Source Value: {pair['source_value']}, Gold: {pair['gold_value']}, Targets: {pair['target_values'][:3]}...")  # Print first 3 targets for brevity