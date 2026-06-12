import copy
from multiprocessing import Pool
import os
from pprint import pprint
import shutil

import numpy as np
import optuna
from analysis import get_performance_metrics
from cfgs import cfg_main_nonid
from sklearn.model_selection import KFold, StratifiedKFold

from framework import run
from interfaces import EEGPhase, LIFConfig, NetworkLocation, ReservoirConfig, TaskType
from plotting import plot_readout_analysis, plot_record
from train import split_records, split_records_labelled

brian_dir = "../tmp/main/brian"
cache_dir = "../tmp/main/cache"

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

# GET ENCODED DATASET
records, global_record = run(
start_loc=NetworkLocation.DATA, 
end_loc=NetworkLocation.DATA, 
train_filter = None,
validate_filter= {EEGPhase.INTERICTAL: 45, EEGPhase.PREICTAL: 45, EEGPhase.ICTAL: 45, 'patients': range(1, 11)},
max_processes=3,
config=cfg_main_nonid.config_encoder,
noise_seed=12,
brian_dir=brian_dir,
cache_dir=cache_dir)

complete_set, complete_set_labels = split_records_labelled(records)
print(complete_set)

process_n = 3
n_splits = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=12)
det_reports = []
params_list = []

for fold, (train_ids, validation_ids) in enumerate(skf.split(complete_set, complete_set_labels)):
    print(f"Fold {fold}")
    train_set = complete_set[train_ids]
    validation_set = complete_set[validation_ids]
    print(validation_set)
    print(f"{len(complete_set)} complete")
    print(f"{len(train_set)} Train, {len(validation_set)} Validate")

    # RUN RESERVOIR NOISELESS AND TRAIN DETECTION
    print("TRAIN DETECTION")
    _, global_record = run(
    start_loc=NetworkLocation.ENCODER_OUT, 
    end_loc=NetworkLocation.TRAIN, 
    train_filter= (train_set, [EEGPhase.INTERICTAL, EEGPhase.ICTAL], False),
    validate_filter= (validation_set, [EEGPhase.INTERICTAL, EEGPhase.ICTAL], False),
    max_processes=process_n,
    config=cfg_main_nonid.config_train,
    noise_seed=12,
    brian_dir=brian_dir,
    cache_dir=cache_dir)

    params_list.append(global_record.get("trained_params"))

    # VALIDATE DETECTION WITH NOISE
    print("VALIDATE DETECTION WITH NOISE")
    records, global_record = run(
    start_loc=NetworkLocation.ENCODER_OUT, 
    end_loc=NetworkLocation.OUTPUT, 
    train_filter= None,
    validate_filter= (validation_set, [EEGPhase.INTERICTAL, EEGPhase.ICTAL], False),
    max_processes=process_n,
    config=cfg_main_nonid.config_validate,
    noise_seed=24,
    brian_dir=brian_dir,
    cache_dir=cache_dir)

    det_reports.append(get_performance_metrics(
    TaskType.CLASSIFICATION,
    [EEGPhase.INTERICTAL, EEGPhase.ICTAL],
    records,
    'res',
    balance=True)["classification_report"])

    # for record in records:
    #     plot_record(cfg_main_nonid.config_validate, record, global_record)


# Plotting and printing results
for params in params_list:
    pprint(params)

plot_readout_analysis(params_list)

for i in np.arange(len(det_reports)):
    print(f"Fold {i}:")
    pprint(det_reports[i])

    det_f1 = det_reports[i]['ICTAL']['f1-score']

avg_det = average_reports(det_reports)

pprint("Average result:")
pprint(avg_det)

def print_metrics(data, task_name, pos_class, neg_class):
    sens = data[pos_class]['recall']
    spec = data[neg_class]['recall']
    acc = data['accuracy']
    f1 = data[pos_class]['f1-score']
    
    print(f"--- {task_name} ---")
    print(f"SENS: {sens:.3f}")
    print(f"SPEC: {spec:.3f}")
    print(f"ACC:  {acc:.3f}")
    print(f"F1:   {f1:.3f}\n")

print_metrics(avg_det, "Detection (Ictal vs Interictal)", 'ICTAL', 'INTERICTAL')
