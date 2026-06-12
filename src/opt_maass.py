import copy
from multiprocessing import Pool
import os
from pprint import pprint
import shutil

import numpy as np
import optuna
from analysis import get_performance_metrics
from cfgs import cfg_maass
from sklearn.model_selection import KFold, StratifiedKFold

from framework import run
from interfaces import ClassNumber, EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, NetworkLocation, Neuron, ReservoirConfig, SimpleLIF, SimpleSyn, Split, Synapse, TaskType, Uniform
from plotting import plot_readout_analysis, plot_record
from train import split_records, split_records_labelled

PARAM_RANGES = {
    # "lamda": (1.126*0.9, 1.126*1.1),
    # "leak_tc": (0.361*0.9, 0.361*1.1),
    # "v_spike_delta": (1.0, 1.0),
    # "syn_delay": (1.669*0.9, 1.669*1.1),
    # "refractory_period": (0.0893*0.9, 0.0893*1.1),
    # "scale": (3.562*0.9, 3.562*1.1),
    # "W_IN": (6.9095*0.9, 6.9095*1.1),
    "F_IN": (1, 60),
    "spectral_radius": (1.0,2.0),
    "lamda": (1.0,2.0)
}

FLOAT_PARAMS =  ["lamda", "spectral_radius"] # ["lamda", "leak_tc", "v_spike_delta", "syn_delay", "refractory_period", "scale", "W_IN"]
INT_PARAMS = ["F_IN"]

trial_root = os.path.join("..", "tmp", "trial", f"trial_0")
# trial_root = os.path.join("/dev/shm", "tmp", "trial", f"trial_0")
brian_dir = os.path.join(trial_root, "brian")
cache_dir = os.path.join(trial_root, "cache")

def run_iteration(reservoirconfig: ReservoirConfig | None) -> dict:
    cfg = copy.deepcopy(cfg_maass.config)
    
    if reservoirconfig is not None:
        cfg.reservoir_config = reservoirconfig

    process_n = 4

    # Running the reservoir
    print("Running reservoir")
    records, global_record = run(
    start_loc=NetworkLocation.ENCODER_OUT, 
    end_loc=NetworkLocation.OUTPUT, 
    train_filter=np.arange(0, 80) ,
    validate_filter=np.arange(80, 100),
    max_processes=process_n,
    config=cfg,
    noise_seed=12,
    brian_dir=brian_dir,
    cache_dir=cache_dir)

    report = get_performance_metrics(
    TaskType.CLASSIFICATION,
    [ClassNumber.ZERO, ClassNumber.ONE],
    records,
    'model',
    balance=True)["classification_report"]

    pprint(report)
    print(records[0].encoder_rate, records[0].reservoir_rate)
    plot_record(cfg, records[0], global_record)

    for record in records:
        if record.sample_metadata.split == Split.VALIDATE:
            print(record.encoder_rate, record.reservoir_rate)
            plot_record(cfg, record, global_record)

    print(records[0].reservoir_rate)
    return report

# def optimizer(trial: optuna.trial.Trial) -> float:
#     os.makedirs(brian_dir, exist_ok=True)
#     os.makedirs(cache_dir, exist_ok=True)

#     float_ranges = {k: v for k, v in PARAM_RANGES.items() if k not in INT_PARAMS}
    
#     params = {name: trial.suggest_float(name, low, high) 
#               for name, (low, high) in float_ranges.items()}

#     # 2. Suggest integers only for the parameters in INT_PARAMS
#     params.update({name: trial.suggest_int(name, *PARAM_RANGES[name]) 
#                    for name in INT_PARAMS})

#     ISI = 1.0/15.0

#     reservoir_config=ReservoirConfig(
#             N=125,
#             F_in=params['F_IN'],
#             levels_in=0,
#             input_write_noise=0,

#             factor_inh=0.2, 
#             C_EE=0.3,
#             C_EI=0.2,
#             C_IE=0.4,
#             C_II=0.1,

#             W_IN_MIN= -1.1,
#             W_IN_MAX=1.1,
#             W_EE = 3*1,
#             W_EI = 6*1,
#             W_IE = -2*1,
#             W_II = -2*1,
#             spectral_radius=params["spectral_radius"],

#             lamda=params["lamda"],
#             syn_delay= ISI * 0.02,

#             lif_config=LIFConfig(
#                 v_threshold=1,
#                 leak_tc= ISI * 0.6, 
#                 v_rest=0,
#                 v_spike_delta=1.0,
#                 refractory_period= ISI * 0.04
#             )
#         )

#     try:
#         report = run_iteration(reservoir_config)

#         f1_score = report['f1_macro']

#         return f1_score

#     except Exception as e:
#         print(f"Trial failed due to: {e}")
#         return 0.0
    
#     finally:
#         # 2. Cleanup: Delete the unique directory to save space
#         # Comment this out if you need to inspect files for failed trials

#         if os.path.exists(brian_dir):
#             shutil.rmtree(brian_dir)

#         if os.path.exists(os.path.join(cache_dir, 'reservoir')):
#             shutil.rmtree(os.path.join(cache_dir, 'reservoir'))

#         if os.path.exists(os.path.join(cache_dir, 'train')):
#             shutil.rmtree(os.path.join(cache_dir, 'train'))

# # 3. Verify the split
# print(f"Complete IDs ({len(complete_set)}): {complete_set}")
# print(f"Training IDs ({len(train_set)}): {train_set}")
# print(f"Validation IDs ({len(validation_set)}): {validation_set}")

# study = optuna.create_study(
#     study_name="TEST2-class-6-mar",
#     storage="sqlite:///../optuna/optuna.db", 
#     direction= "maximize",
#     load_if_exists=True,
#     sampler=optuna.samplers.TPESampler(multivariate=True)
# )

# suggested_params = {
#     "lamda": 1.1261620758016635,
#     "F_IN": 10,
#     "spectral_radius": 2
# }

# # Enqueue it into the study
# study.enqueue_trial(suggested_params)

# study.optimize(optimizer) 

# sim_multiplier = 0.12

sim_multiplier = 0.005

global_strength = 1.0
in_strength = 1.0
res_strength = 1.0

in_x = in_strength * global_strength
res_x = res_strength * global_strength

synapse_in = Synapse(
    SimpleSyn(Gaussian(18e-9*in_x, 18e-9*in_x),0),
    SimpleSyn(Gaussian(9e-9*in_x, 9e-9*in_x),0),
    SimpleSyn(Fixed(0),0),
    SimpleSyn(Fixed(0),0), 
    0, 
    0)

synapse_simple_res = Synapse(
    EE=SimpleSyn(Fixed(30e-9 *res_x* sim_multiplier),0.001),
    EI=SimpleSyn(Fixed(60e-9 *res_x*sim_multiplier),0.001),
    IE=SimpleSyn(Fixed(-19e-9 *res_x* sim_multiplier),0.001),
    II=SimpleSyn(Fixed(-19e-9 *res_x* sim_multiplier),0.001),
    levels=0,
    write_noise=0
)

synapse_markram_res = Synapse(
    EE=MarkramSyn(Gaussian(30e-9*res_x, 30e-9*res_x), Gaussian(0.5, 0.25), Gaussian(1.1, 0.55), Gaussian(0.05, 0.025), 0.0015),
    EI=MarkramSyn(Gaussian(60e-9*res_x, 60e-9*res_x), Gaussian(0.05, 0.025), Gaussian(0.125, 0.0625), Gaussian(1.2, 0.6), 0.0008),
    IE=MarkramSyn(Gaussian(-19e-9*res_x, 19e-9*res_x), Gaussian(0.25, 0.125), Gaussian(0.7, 0.35), Gaussian(0.02, 0.01), 0.0008),
    II=MarkramSyn(Gaussian(-19e-9*res_x, 19e-9*res_x), Gaussian(0.32, 0.16), Gaussian(0.144, 0.072), Gaussian(0.06, 0.03), 0.0008),
    
    levels=0,
    write_noise=0
)

simplelif = SimpleLIF(
    threshold_v=0.015,
    leak_tau=0.03,
    v_rest=0.0135,
    refractory_period=0.002
)

maass_e = MaassLIF(
    leak_tau=0.03,
    threshold_v=0.015,
    v_rest=0.0,
    reset_v=0.0135,
    background_I=13.5e-9,
    input_resistance=1e6,
    tau_syn_exc=0.003,      
    tau_syn_inh=0.006,     
    refractory_period=0.003 
)

maass_i = MaassLIF(
    leak_tau=0.03,
    threshold_v=0.015,
    v_rest=0.0,
    reset_v=0.0135,
    background_I=13.5e-9,
    input_resistance=1e6,
    tau_syn_exc=0.003,      
    tau_syn_inh=0.006,
    refractory_period=0.002
)

simple_neuron_res = Neuron(simplelif, simplelif)
maass_neuron_res = Neuron(maass_e, maass_i)


reservoir_config=ReservoirConfig(
    N=125,
    factor_inh=0.2,

    F_in=40,
    synapse_in=synapse_in,

    C_EE=0.3,
    C_EI=0.2,
    C_IE=0.4,
    C_II=0.1,

    lamda=2.0,
    spectral_radius=0.0001, # unused
    synapse_res=synapse_markram_res,

    neuron_res=maass_neuron_res
)

report = run_iteration(reservoir_config)
print(report['accuracy'])