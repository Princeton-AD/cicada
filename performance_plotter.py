import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from generator import RegionETGenerator
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import numpy.typing as npt
from utils import quantize, loss, get_teacher_model, get_qkeras_student_model, get_ebops_from_log, get_student_models

def load_data_and_config(config_file: str) -> Tuple:
    """Loads config and datasets."""
    print("\n--- Loading all configurations and data ---")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    color_map = {item['name']: item['color'] for item in config.get('signal', []) if item.get('use')}
    processes_of_interest = {name: None for name in color_map.keys()}

    gen = RegionETGenerator()
    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]
    _, _, X_test = gen.get_data_split(datasets)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    for process_name, signal_data in X_signal.items():
        if process_name in processes_of_interest:
            processes_of_interest[process_name] = signal_data
            
    background_processes = config.get('background', [])
    for process in background_processes:
        if process.get('name') == 'Zero Bias' and process.get('use'):
            color_map['Zero Bias'] = process.get('color')
            processes_of_interest['Zero Bias'] = X_test
            break
            
    return config, color_map, processes_of_interest

def get_teacher_anomaly_scores(model: keras.Model, data: np.ndarray) -> np.ndarray:
    """Calculates the raw reconstruction error (MSE) for the teacher model."""
    reconstructed_data = model.predict(data, batch_size=512, verbose=0)
    mse = loss(data,reconstructed_data)
    return mse

def get_student_anomaly_scores(model: keras.Model, data: np.ndarray) -> np.ndarray:
    """Gets quantized anomaly scores from a student model."""
    if data.ndim > 2 and model.input_shape[1] != data.shape[1]:
        data = data.reshape(-1, np.prod(data.shape[1:]), 1)
    return model.predict(data, batch_size=512, verbose=0)

def analyze_all_students(teacher, students_dict, processes, background_key='Zero Bias'):
    """Analyzes all student models against the teacher and returns a DataFrame."""
    print("\n--- Starting analysis of all students ---")
    
    # Pre-calculate all teacher scores (raw and transformed)
    teacher_scores_raw = {}
    teacher_scores_transformed = {}
    print("Pre-calculating teacher scores for all processes...")
    for name, data in processes.items():
        raw_scores = get_teacher_anomaly_scores(teacher, data)
        teacher_scores_raw[name] = raw_scores
        teacher_scores_transformed[name] = quantize(np.log(raw_scores) * 32)

    results = []
    X_test = processes[background_key]

    for student_key, student_info in students_dict.items():
        student_model = student_info['model']
        student_name = student_info['name']
        log_path = student_info['log_path']
        ebops = get_ebops_from_log(log_path)
        student_label = f"{student_name}_{int(ebops)}" if not np.isnan(ebops) else student_name
        print(f"\n--- Analyzing student: {student_label} ---")

        # Pre-calculate all student scores
        student_scores = {}
        for name, data in processes.items():
            student_scores[name] = get_student_anomaly_scores(student_model, data)
        
        # Calculate metrics for each signal process
        for name, data in processes.items():
            if name == background_key:
                continue

            # 1. Wasserstein Distance
            dist = wasserstein_distance(
                teacher_scores_transformed[name].flatten(),
                student_scores[name].flatten()
            )

            # 2. ROC AUC
            y_true = np.concatenate([np.ones(data.shape[0]), np.zeros(X_test.shape[0])])
            y_pred_student = np.concatenate([student_scores[name], student_scores[background_key]])
            fpr, tpr, _ = roc_curve(y_true, y_pred_student)
            roc_auc = auc(fpr, tpr)
            
            results.append({
                'student_label': student_label,
                'process': name,
                'wasserstein_distance': dist,
                'roc_auc': roc_auc
            })
            print(f"  Process: {name} | Wasserstein Dist: {dist:.4f} | ROC AUC: {roc_auc:.4f}")

        ## add wasserstein distance for Zero Bias
        dist = wasserstein_distance(
                teacher_scores_transformed['Zero Bias'].flatten(),
                student_scores['Zero Bias'].flatten()
        )
        results.append({
                'student_label': student_label,
                'process': name,
                'wasserstein_distance': dist,
                'roc_auc': 0
        })

    print("\n--- Analyzing Teacher model as test of code ---")
    for name, data in processes.items():
        if name == background_key: continue
        dist = wasserstein_distance(teacher_scores_transformed[name].flatten(), teacher_scores_transformed[name].flatten())
        
        y_true = np.concatenate([np.ones(data.shape[0]), np.zeros(X_test.shape[0])])
        y_pred_teacher = np.concatenate([teacher_scores_transformed[name], teacher_scores_transformed[background_key]])
        fpr, tpr, _ = roc_curve(y_true, y_pred_teacher)
        roc_auc = auc(fpr, tpr)
        
        results.append({'student_label': 'Teacher', 'process': name, 'wasserstein_distance': dist, 'roc_auc': roc_auc})# not student but can reuse existing df
        print(f"  Process: {name} | Wasserstein Dist: {dist:.4f} | ROC AUC: {roc_auc:.4f}")

    return pd.DataFrame(results)

def plot_summary_results(df: pd.DataFrame, color_map: Dict):
    """Generates two summary plots comparing all students."""
    if df.empty:
        print("No results to plot.")
        return
    print("\n--- Generating summary plots ---")
    sns.set_theme(style="whitegrid", context="talk")

    plot_order = sorted(
    df['student_label'].unique(), 
    key=lambda x: (x.split('_')[0] != 'Teacher', int(x.split('_')[1]) if '_' in x else 0)
    )

   # Plot 1: Wasserstein Distance vs. Student Model
    plt.figure(figsize=(14, 8))
    ax1 = sns.barplot(
        data=df,
        x='student_label',
        y='wasserstein_distance',
        hue='process',
        palette=color_map,
        order=plot_order
    )
    ax1.set_title('Student-Teacher Anomaly Score Distance', fontsize=18, pad=20)
    ax1.set_xlabel('Model (name_ebops)', fontsize=14)
    ax1.set_ylabel('Wasserstein Distance', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax1.legend(title='Process', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("summary_wasserstein_distance_test_withZB.png", dpi=300, bbox_inches='tight')
    print("Saved Wasserstein distance summary plot.")
    plt.show()

    # Plot 2: ROC AUC vs. Student Model
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(
        data=df,
        x='student_label',
        y='roc_auc',
        hue='process',
        palette=color_map,
        order=plot_order
    )
    ax2.set_title('Model ROC Performance', fontsize=18, pad=20)
    ax2.set_xlabel('Model (name_ebops)', fontsize=14)
    ax2.set_ylabel('ROC Area Under Curve (AUC)', fontsize=14)
    ax2.set_ylim(bottom=max(0, df['roc_auc'].min() - 0.05), top=1.0)
    plt.xticks(rotation=45, ha='right')
    ax2.legend(title='Process', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("summary_roc_auc_test_withZB.png", dpi=300, bbox_inches='tight')
    print("Saved ROC AUC summary plot.")
    plt.show()


def main():
    """Main function to drive the loading, analysis, and plotting."""
    config_file = 'misc/config.yml'
    
    config, color_map, processes = load_data_and_config(config_file)
    teacher = get_teacher_model()
    students = get_student_models()

    if not students:
        print("No student models were loaded. Exiting.")
        return

    results_df = analyze_all_students(teacher, students, processes)
    
    if not results_df.empty:
        print("\nPlotting")
        print(results_df.round(4))
        plot_summary_results(results_df, color_map)
    else:
        print("\nAnalysis finished with no results.")

if __name__ == '__main__':
    main()
