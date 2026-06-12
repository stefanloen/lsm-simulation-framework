import itertools
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
from interfaces import Beta, ClassNumber, EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, NetworkLocation, Neuron, ReservoirConfig, SimpleLIF, SimpleSyn, Split, Synapse, TaskType, Uniform
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

    process_n = 5

    # Running the reservoir
    print("Running reservoir")
    records, global_record = run(
    start_loc=NetworkLocation.ENCODER_OUT, 
    end_loc=NetworkLocation.MODEL_OUTPUT, 
    train_filter=np.arange(0, 20) ,
    validate_filter=np.arange(20, 25),
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

    return report


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

maass_neuron_res = Neuron(maass_e, maass_i)

# 1. Define the ranges you want to sweep
# You can adjust these lists to your specific needs
global_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
in_range = [0.25, 0.5, 0.75, 1.0]
res_range = [0.0, 0.25, 0.5, 0.75, 1.0]

results = []

print(f"Starting sweep: {len(global_range) * len(in_range) * len(res_range)} iterations total.")

# 2. Nested Sweep
for g_str, i_str, r_str in itertools.product(global_range, in_range, res_range):
    
    # Calculate current multipliers
    in_x = i_str * g_str
    res_x = r_str * g_str
    
    # Define Synapses with current multipliers
    synapse_in = Synapse(
        SimpleSyn(Beta(18e-9*in_x, 0.9*18e-9*in_x),0),
        SimpleSyn(Beta(9e-9*in_x, 0.9*9e-9*in_x),0),
        SimpleSyn(Fixed(0),0),
        SimpleSyn(Fixed(0),0), 
        0, 
        0)

    synapse_markram_res = Synapse(
        EE=MarkramSyn(Beta(30e-9*res_x, 0.9*30e-9*res_x), Beta(0.5, 0.45), Beta(1.1, 0.99), Beta(0.05, 0.045), 0.0015),
        EI=MarkramSyn(Beta(60e-9*res_x, 0.9*60e-9*res_x), Beta(0.05, 0.045), Beta(0.125, 0.1125), Beta(1.2, 1.08), 0.0008),
        IE=MarkramSyn(Beta(-19e-9*res_x, 0.9*19e-9*res_x), Beta(0.25, 0.225), Beta(0.7, 0.63), Beta(0.02, 0.018), 0.0008),
        II=MarkramSyn(Beta(-19e-9*res_x, 0.9*19e-9*res_x), Beta(0.32, 0.288), Beta(0.144, 0.1296), Beta(0.06, 0.054), 0.0008),
        
        levels=0,
        write_noise=0
    )

    # Re-build config with current parameters
    reservoir_config = ReservoirConfig(
        N=125,
        factor_inh=0.2,
        F_in=40,
        synapse_in=synapse_in,
        C_EE=0.3, C_EI=0.2, C_IE=0.4, C_II=0.1,
        lamda=2.0,
        synapse_res=synapse_markram_res,
        neuron_res=maass_neuron_res, # Assuming maass_neuron_res is defined above
        spectral_radius=1.0
    )

    # 3. Run and Store
    try:
        report = run_iteration(reservoir_config)
        accuracy = report['accuracy']
        results.append({
            'global': g_str,
            'in': i_str,
            'res': r_str,
            'accuracy': accuracy
        })
        print(f"G: {g_str:.1f}, In: {i_str:.1f}, Res: {r_str:.1f} -> Acc: {accuracy:.4f}")
    except Exception as e:
        print(f"Error at G:{g_str} In:{i_str} Res:{r_str}: {e}")

# 4. Final Summary
print("\n--- SWEEP FINISHED ---")
print(f"{'Global':<8} | {'Input':<8} | {'Res':<8} | {'Accuracy':<10}")
print("-" * 45)
for r in results:
    print(f"{r['global']:<8.2f} | {r['in']:<8.2f} | {r['res']:<8.2f} | {r['accuracy']:<10.4f}")