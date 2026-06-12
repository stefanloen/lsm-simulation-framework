import json
import os
from pprint import pprint
import shutil
from matplotlib import pyplot as plt
import optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
import copy

from analysis import get_convergence_factor, get_samples_to_converge, get_separation_factor
from framework import run
from interfaces import SimConfig, Config, EncodingConfig, EncodingType, LIConfig, LIFConfig, Mode, NetworkLocation, OutputConfig, PoissonPreprocessingConfig, ReservoirConfig, TrainConfig
from contextlib import redirect_stdout

from cfgs import cfg_metric_exp_sep, cfg_metric_conv
from plotting import plot_record, plot_reservoir_convergence

PARAM_RANGES = {
    "lamda": (1.0, 3.0),
    "leak_tc": (0.001, 1.0),
    "v_spike_delta": (1.0, 1.0),
    "syn_delay": (0.001, 1.000),
    "refractory_period": (0.001, 0.500),
    "scale": (0.1, 5),
    "W_IN": (0.01, 32),
    "F_IN": (1, 125)
}

global_best_score = -np.inf

def get_utility(x, low, high):
    """
    Returns a utility score from 0 to 1.
    x=low  => 0.05 utility
    x=high => 0.95 utility
    """
    if high == low: return 0.0
    
    midpoint = (high + low) / 2
    # The constant 5.888 is 2 * ln(19)
    steepness = 5.888 / (high - low)
    
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def run_iteration(reservoir_config: ReservoirConfig | None, brian_dir: str, cache_dir: str):
    exp_sep_config = copy.deepcopy(cfg_metric_exp_sep.config) 
    conv_config = copy.deepcopy(cfg_metric_conv.config)
    
    if reservoir_config is not None:
        exp_sep_config.reservoir_config = reservoir_config
        conv_config.reservoir_config = reservoir_config

    records, global_record = run(
        start_loc=NetworkLocation.ENCODER_OUT, 
        end_loc=NetworkLocation.RESERVOIR_OUT, 
        sample_filter=np.arange(8), # Ensure this is an even number for pairs
        max_processes=1,
        config=exp_sep_config,
        brian_dir=brian_dir,
        cache_dir=cache_dir
    )

    # Expansion
    res_ranks = [res.get('reservoir_rank') for res in records]
    enc_ranks = [res.get('encoder_rank') for res in records]

    expansion_factors = [
        res / enc if enc > 0 else 0.0 
        for res, enc in zip(res_ranks, enc_ranks)
    ]
    expansion_factor = np.mean(expansion_factors)

    last_n = int(10.0 / conv_config.sim_config.analysis_dt)

    # Separation
    separation_factors = []
    for i in range(0, len(records), 2):
        res_a = records[i]
        res_b = records[i+1]
        
        sep_f = get_separation_factor(
            in_a=res_a.get('encoder_trace'),
            in_b=res_b.get('encoder_trace'),
            out_a=res_a.get('reservoir_trace'),
            out_b=res_b.get('reservoir_trace'),
            last_n=last_n
        )
        
        separation_factors.append(sep_f)

    separation_factor = np.mean(separation_factors)

    # Convergence
    records, global_record = run(
        start_loc=NetworkLocation.ENCODER_OUT, 
        end_loc=NetworkLocation.RESERVOIR_OUT, 
        sample_filter=np.arange(16), # Ensure this is an even number for pairs
        max_processes=1,
        config=conv_config,
        brian_dir=brian_dir,
        cache_dir=cache_dir
    )

    convergence_times = []
    for i in range(0, len(records), 2):
        res_a = records[i]
        res_b = records[i+1]

        from_t = conv_config.preprocessing_config.random_duration #type: ignore

        samples_to_converge = get_samples_to_converge(
            run1=res_a.get('reservoir_trace'),
            run2=res_b.get('reservoir_trace'),
            from_N=int(from_t / conv_config.sim_config.analysis_dt),
            threshold=0.0001
        )

        convergence_time = samples_to_converge * conv_config.sim_config.analysis_dt 
        convergence_times.append(convergence_time)

        conv_f, distance_reservoir = get_convergence_factor(
            run1=res_a.get('reservoir_trace'),
            run2=res_b.get('reservoir_trace'),
            last_n=last_n
        )

        plot_record(conv_config, res_a, global_record)
        plot_reservoir_convergence(distance_reservoir,res_a.label ,1/conv_config.sim_config.analysis_dt, from_t+convergence_time)

    if np.any(np.array(convergence_times) < 0.0):
        convergence_time = -1.0
    else:
        convergence_time = np.mean(convergence_times)
    
    return float(expansion_factor), float(separation_factor), float(convergence_time)

def opt_exp_sep_conv(trial: optuna.trial.Trial):
    #trial_root = os.path.join("..", "tmp", "trial", f"trial_{trial.number}")
    trial_root = os.path.join("/dev/shm", "tmp", "trial", f"trial_{trial.number}")
    brian_dir = os.path.join(trial_root, "brian")
    cache_dir = os.path.join(trial_root, "cache")

    os.makedirs(brian_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    global global_best_score
    params = {name: trial.suggest_float(name, low, high) 
              for name, (low, high) in PARAM_RANGES.items()}

    reservoir_config = ReservoirConfig(
        N=125,
        F_in = int(params['F_IN']),
        factor_inh=0.2,
        C_EE=0.3,
        C_EI=0.2,
        C_IE=0.4,
        C_II=0.1,

        W_IN= params['W_IN'],
        W_EE= 3 * params['scale'],
        W_EI= 6 * params['scale'],
        W_IE= -2 * params['scale'],
        W_II= -2 * params['scale'],

        lamda = params['lamda'],
        syn_delay=params['syn_delay'],

        lif_config=LIFConfig(
            v_threshold=1,
            leak_tc=params['leak_tc'], 
            v_rest=0,
            v_spike_delta=params['v_spike_delta'],
            refractory_period=params['refractory_period']
        )
    )

    try:
        expansion_factor, separation_factor, convergence_time = run_iteration(reservoir_config, brian_dir, cache_dir)

        if convergence_time < 0.0:
            raise ValueError("Reservoir did not converge")

        score = expansion_factor * separation_factor * (1+convergence_time)

        trial.set_user_attr("expansion_factor", float(expansion_factor))
        trial.set_user_attr("separation_factor", float(separation_factor))
        trial.set_user_attr("convergence_time", float(convergence_time))
        trial.set_user_attr("score", float(score))

        return expansion_factor, separation_factor, convergence_time

    except Exception as e:
        print(f"Trial failed due to: {e}")
        return 0.0, 0.0, 0.0
    
    finally:
        if os.path.exists(trial_root):
            shutil.rmtree(trial_root)

os.makedirs("../optuna", exist_ok=True)

# def run_optimization(pool_id):
#     study = optuna.create_study(
#         study_name="exp_sep_conv_3_mar",
#         storage="sqlite:///../optuna/optuna.db", 
#         directions=["maximize", "maximize", "maximize"],
#         load_if_exists=True
#     )

#     # initial_params = {
#     #     "lamda": 2.0,
#     #     "leak_tc": 0.05,
#     #     "v_spike_delta": 1.0,
#     #     "syn_delay": 0.003,
#     #     "refractory_period": 0.002,
#     #     "scale": 0.4,
#     #     "W_IN": 8.0,
#     #     "F_IN": 4
#     # }

#     # study.enqueue_trial(initial_params)

#     study.optimize(opt_exp_sep_conv) 

# with Pool(processes=16) as pool:
#     pool.map(run_optimization, range(16))

trial_root = os.path.join("..", "tmp", "trial", f"manual")
# trial_root = os.path.join("/dev/shm", "tmp", "trial", f"trial_{trial.number}")
brian_dir = os.path.join(trial_root, "brian")
cache_dir = os.path.join(trial_root, "cache")


reservoir_config=ReservoirConfig(
        N=125,
        F_in=8,
        factor_inh=0.2,
        C_EE=0.3,
        C_EI=0.2,
        C_IE=0.4,
        C_II=0.1,

        W_IN=1,
        W_EE=3 * 0.50,
        W_EI=6 * 0.50,
        W_IE=-2 * 0.50,
        W_II=-2 * 0.50,

        lamda=2.0,
        syn_delay=0.005,

        lif_config=LIFConfig(
            v_threshold=1,
            leak_tc=0.035, 
            v_rest=0,
            v_spike_delta=1.0,
            refractory_period=0.002
        )
    )

print(run_iteration(reservoir_config, brian_dir, cache_dir))