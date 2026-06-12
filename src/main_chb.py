import brian2 as b2
import numpy as np
from multiprocessing import Pool
from pprint import pprint

from framework import (
    Config, Mode, run, NetworkLocation,
    CHBPreprocessingConfig, ReservoirConfig, 
    LIFConfig, LIConfig, EncodingConfig, OutputConfig, 
    EncodingType, TrainConfig
)
from interfaces import (
    EEGPhase, PoissonPreprocessingConfig, SimConfig, SolverType, 
    TaskType, WhiteNoiseConfig
)
from analysis import (
    calculate_pca_distance, get_convergence_factor, get_separation_factor,
    get_performance_metrics
)
from plotting import plot_record, plot_reservoir_convergence

from cfgs import cfg_main_chb

raw_data = """
  0   3   8   9  13  18  19  20  21  22  33  35  38  43  44  47  57  61
  62  63  64  68  69  71  80  87  98 107 108 109 113 121 123 124 126 128
 130 132 135 136 138 139 140 141 147 148 149 150 154 157 163 164 166 167
 174 175 179 187 192 198 201 204 209 212 215 216 218
"""
validation_set = np.array([int(x) for x in raw_data.split()])

pprint(validation_set)

records, global_record = run(
    start_loc=NetworkLocation.DATA, 
    end_loc=NetworkLocation.OUTPUT, 
    train_filter = None,
    validate_filter = {EEGPhase.INTERICTAL: 5, EEGPhase.ICTAL: 5, 'patients': range(17, 22)},#None, # np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    max_processes=3,
    config=cfg_main_chb.config,
    brian_dir='../tmp/main/brian',
    cache_dir='../tmp/main/cache') # './' for current directory. Could also be '/dev/shm/brian' to spare SSD,

print(global_record.trained_params)

# res_performance = get_performance_metrics(
#     config.train_config.task_type,
#     config.train_config.classes,
#     records,
#     'res',
#     balance=True)

# pprint(res_performance)

metrics = get_performance_metrics(
        TaskType.CLASSIFICATION,
        [EEGPhase.INTERICTAL, EEGPhase.ICTAL],
        records,
        'res',
        balance=False)["classification_report"]

pprint(metrics)

for record in records: 
    print(record.encoder_rank)
    print(record.reservoir_rank)
    plot_record(cfg_main_chb.config, record, global_record, f"Sample {record.get('sample_metadata').sample_id}")