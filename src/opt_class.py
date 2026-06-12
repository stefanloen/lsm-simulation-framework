import copy
from multiprocessing import Pool
import os
from pprint import pprint
import shutil

import numpy as np
import optuna
from analysis import get_performance_metrics
from cfgs import cfg_class
from sklearn.model_selection import KFold, StratifiedKFold

from framework import run
from interfaces import EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, NetworkLocation, Neuron, ReservoirConfig, SimpleLIF, SimpleSyn, Split, Synapse, TaskType
from plotting import plot_readout_analysis, plot_record
from train import split_records, split_records_labelled

trial_root = os.path.join("..", "tmp", "trial", f"trial_0")
# trial_root = os.path.join("/dev/shm", "tmp", "trial", f"trial_0")
brian_dir = os.path.join(trial_root, "brian")
cache_dir = os.path.join(trial_root, "cache")

def average_reports(reports_list):
    keys = reports_list[0].keys()
    averaged = {}
    
    for key in keys:
        # Check if the value is a dictionary (like 'ICTAL' or 'macro avg')
        if isinstance(reports_list[0][key], dict):
            metrics = reports_list[0][key].keys()
            averaged[key] = {m: np.mean([r[key][m] for r in reports_list]) for m in metrics}
        else:
            # If it's a scalar (like 'accuracy'), just average the values directly
            averaged[key] = np.mean([r[key] for r in [x for x in reports_list if key in x]])
            
    return averaged

def run_iteration(reservoirconfig: ReservoirConfig | None, n_splits : int) -> list[dict]:
    cfg = copy.deepcopy(cfg_class.reservoir)
    
    if reservoirconfig is not None:
        cfg.reservoir_config = reservoirconfig

    process_n = 5

    # RUN 1: RUN RESERVOIR
    print("Running reservoir")
    records, global_record = run(
    start_loc=NetworkLocation.DATA, 
    end_loc=NetworkLocation.RESERVOIR_OUT, 
    train_filter= None,
    validate_filter={EEGPhase.INTERICTAL: 140, EEGPhase.ICTAL: 140, 'patients': [i for i in range(1, 25) if i != 12]},
    max_processes=process_n,
    config=cfg,
    noise_seed=12,
    brian_dir=brian_dir,
    cache_dir=cache_dir)

    # 3. Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12)

    reports = []

    params_list = []

    complete_set, complete_set_labels = split_records_labelled(records)
    print(f'complete set: {complete_set}')

    for fold, (train_ids, validation_ids) in enumerate(skf.split(complete_set, complete_set_labels)):
        print(f"Fold {fold}")
        train_set = complete_set[train_ids]
        validation_set = complete_set[validation_ids]

        print(f"train_set: {train_set}")
        print(f"validation_set: {validation_set}")
        print(f"{len(complete_set)} complete")
        print(f"{len(train_set)} Train, {len(validation_set)} Validate")

        # TRAIN AND VALIDATE
        print("TRAIN AND VALIDATE")
        records, global_record = run(
        start_loc=NetworkLocation.RESERVOIR_OUT, 
        end_loc=NetworkLocation.OUTPUT, 
        train_filter= train_set,
        validate_filter= validation_set,
        max_processes=process_n,
        config=cfg_class.trainvalidate,
        noise_seed=12,
        brian_dir=brian_dir,
        cache_dir=cache_dir)

        # for record in records:
        #     if record.sample_metadata.split == Split.VALIDATE:
        #         plot_record(cfg, record, global_record)

        params_list.append(global_record.get("trained_params"))

        reports.append(get_performance_metrics(
        TaskType.CLASSIFICATION,
        [EEGPhase.INTERICTAL, EEGPhase.ICTAL],
        records,
        'model',
        balance=True)["classification_report"])

    for params in params_list:
        pprint(params)

    plot_readout_analysis(params_list)

    return reports

reports = run_iteration(cfg_class.reservoir_config, 5)

# Plotting and printing:

for i in np.arange(len(reports)):
    print(f"Fold {i}:")
    pprint(reports[i])

    det_f1 = reports[i]['ICTAL']['f1-score']

avg_report = average_reports(reports)

pprint("Average result:")
pprint(avg_report)

def print_metrics(data, task_name, pos_class, neg_class):
    # Sensitivity (Recall of positive class)
    sens = data[pos_class]['recall']
    # Specificity (Recall of negative class)
    spec = data[neg_class]['recall']
    # Accuracy
    acc = data['accuracy']
    # F1-score (of positive class)
    f1 = data[pos_class]['f1-score']
    
    print(f"--- {task_name} ---")
    print(f"SENS: {sens:.3f}")
    print(f"SPEC: {spec:.3f}")
    print(f"ACC:  {acc:.3f}")
    print(f"F1:   {f1:.3f}\n")

# Print for Detection
print_metrics(avg_report, "Detection (Ictal vs Interictal)", 'ICTAL', 'INTERICTAL')