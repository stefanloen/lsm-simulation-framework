# Get N encoder and reservoir traces of mixed Interictal and Ictal samples

# calculate distance between all the input traces and output traces creating a matrix of size 4, N(N-1)
# each Row: in-dist, out-dist, class1, class2

# Plot the matrix

import itertools
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

from analysis import calculate_separation_metrics, get_convergence_factor, get_samples_to_converge, get_separation_factor
from framework import run
from interfaces import EEGPhase, SimConfig, Config, EncodingConfig, EncodingType, LIConfig, LIFConfig, Mode, NetworkLocation, OutputConfig, PoissonPreprocessingConfig, ReservoirConfig, TrainConfig
from contextlib import redirect_stdout

from cfgs import cfg_separation
from plotting import plot_record, plot_reservoir_convergence, plot_separation_ratio

cache_dir = "../tmp/trial2/cache"

STRIDE = 1.0 # seconds

PARAM_RANGES = {
    # "lamda": (1.126*0.9, 1.126*1.1),
    # "leak_tc": (0.361*0.9, 0.361*1.1),
    # "v_spike_delta": (1.0, 1.0),
    # "syn_delay": (1.669*0.9, 1.669*1.1),
    # "refractory_period": (0.0893*0.9, 0.0893*1.1),
    # "scale": (3.562*0.9, 3.562*1.1),
    # "W_IN": (6.9095*0.9, 6.9095*1.1),
    "F_IN": (1, 60),
    "W_IN_MAX": (0 , 1.1),
    "W_IN_WIDTH": (0, 1),
    "spectral_radius": (0.5,4.0),
    "lamda": (1.0,3.0),
    "ISI": (0.005, 1.0),
    "delta": (0.001, 1),
    "delay_factor": (0.01, 2),
    "leak_tc_factor": (0.01, 10),
    "refractory": (0.01, 2)
}

FLOAT_PARAMS =  ["lamda", "W_IN_MAX", "W_IN_WIDTH", "ISI", "spectral_radius", "delta", "delay_factor", "leak_tc_factor", "refractory"] # ["lamda", "leak_tc", "v_spike_delta", "syn_delay", "refractory_period", "scale", "W_IN"]
INT_PARAMS = ["F_IN"]

def run_iteration(
          reservoirconfig: ReservoirConfig, 
          encodingconfig: EncodingConfig,
          brian_dir: str) -> tuple[float,float,float, float, float]:
     
    cfg = copy.deepcopy(cfg_separation.config)
    
    if reservoirconfig is not None:
        cfg.reservoir_config = reservoirconfig

    if encodingconfig is not None:
        cfg.encoding_config = encodingconfig

    records, global_record = run(
        start_loc=NetworkLocation.PREPROCESSOR_OUT, 
        end_loc=NetworkLocation.RESERVOIR_OUT, 
        train_filter = None,
        validate_filter={EEGPhase.INTERICTAL: 20, EEGPhase.PREICTAL: 20, 'patients': range(1, 11)},
        max_processes=1,
        config= cfg,
        brian_dir=brian_dir,
        cache_dir=cache_dir
    )

    results = []

    for rec_i, rec_j in itertools.combinations(records, 2):        
            # Extract traces
            u_i = rec_i.get("encoder_trace")    # Input trace u
            u_j = rec_j.get("encoder_trace")
            x_i = rec_i.get("reservoir_trace")  # State trace x
            x_j = rec_j.get("reservoir_trace")

            min_len = min(u_i.shape[0], u_j.shape[0])
            
            dt = cfg.sim_config.analysis_dt

            for idx in range(int(cfg.train_config.washout_period / dt), min_len, int(STRIDE / dt)):

                dist_u = np.linalg.norm(u_i[idx] - u_j[idx])
                dist_x = np.linalg.norm(x_i[idx] - x_j[idx])
                
                valid_i, class_i = rec_i.get("label")[idx]
                valid_j, class_j = rec_j.get("label")[idx]

                if valid_i and valid_j:
                    results.append([dist_u, dist_x, class_i, class_j])

    plot_separation_ratio(results)

    cv, cd, sep, slope, intercept = calculate_separation_metrics(np.array(results))

    return cv, cd, sep, slope, intercept

def optimizer(trial: optuna.trial.Trial) -> float:
    trial_root = os.path.join("/dev/shm/tmp", "trial", f"trial_{trial.number}")
    brian_dir = os.path.join(trial_root, "brian")
    
    os.makedirs(cache_dir, exist_ok=True)

    float_ranges = {k: v for k, v in PARAM_RANGES.items() if k not in INT_PARAMS}
    
    params = {name: trial.suggest_float(name, low, high) 
              for name, (low, high) in float_ranges.items()}

    params.update({name: trial.suggest_int(name, *PARAM_RANGES[name]) 
                   for name in INT_PARAMS})

    
    ISI = params["ISI"]
    W_IN_MAX = params["W_IN_MAX"]
    W_IN_MIN =  W_IN_MAX - (W_IN_MAX+1.1) * params["W_IN_WIDTH"]

    reservoir_config=ReservoirConfig(
            N=125,
            F_in=params['F_IN'],
            factor_inh=0.2, 
            C_EE=0.3,
            C_EI=0.2,
            C_IE=0.4,
            C_II=0.1,

            W_IN_MIN= W_IN_MIN,
            W_IN_MAX= W_IN_MAX,
            W_EE = 3*1,
            W_EI = 6*1,
            W_IE = -2*1,
            W_II = -2*1,
            spectral_radius=params["spectral_radius"],

            lamda=params["lamda"],
            syn_delay= ISI * params["delay_factor"],

            lif_config=LIFConfig(
                v_threshold=1,
                leak_tc= ISI * params["leak_tc_factor"], 
                v_rest=0,
                v_spike_delta=1.0,
                refractory_period= ISI * params["refractory"]
            )
        )
    
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.DELTA,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=80,
        binning_sigma=0.1,
        binning_k=1,
        delta_size=params["delta"]
    )

    try:
        cv, cd, sep, slope, intercept = run_iteration(reservoir_config, encoding_config, brian_dir)

        zone_penalty = (1-slope)**2
        stability_penalty = (intercept / ((cv+cd)/2+1e-6))**2
        separation_reward = sep

        loss = 100*zone_penalty + 10*stability_penalty - separation_reward

        trial.set_user_attr(f"zone_penalty", zone_penalty)
        trial.set_user_attr(f"stability_penalty", stability_penalty)
        trial.set_user_attr(f"separation_reward", separation_reward)

        trial.set_user_attr(f"cv", cv)
        trial.set_user_attr(f"cd", cd)
        trial.set_user_attr(f"sep", sep)
        trial.set_user_attr(f"slope", slope)
        trial.set_user_attr(f"intercept", intercept)

        return loss

    except Exception as e:
        print(f"Trial failed due to: {e}")
        return 1000.0
    
    finally:
        if os.path.exists(trial_root):
            shutil.rmtree(trial_root)

cfg = copy.deepcopy(cfg_separation.config)
cfg.sim_config.cache_preprocessor = True
cfg.sim_config.encoder_spikes = False
cfg.sim_config.encoder_trace = False
cfg.sim_config.reservoir_spikes = False
cfg.sim_config.reservoir_trace = False

records, global_record = run(
    start_loc=NetworkLocation.DATA, 
    end_loc=NetworkLocation.PREPROCESSOR_OUT, 
    train_filter = None,
    validate_filter={EEGPhase.INTERICTAL: 20, EEGPhase.PREICTAL: 20, 'patients': range(1, 11)},
    max_processes=1,
    config= cfg,
    brian_dir="../tmp/preprocessing/brian",
    cache_dir=cache_dir
)

# def run_optimization(pool_id):
#     study = optuna.create_study(
#         study_name="Separation-12-mar",
#         storage="sqlite:///../optuna/optuna.db", 
#         direction= "minimize",
#         load_if_exists=True,
#         sampler=optuna.samplers.TPESampler(multivariate=True)
#     )

#     study.optimize(optimizer) 

# with Pool(processes=4) as pool:
#     pool.map(run_optimization, range(4))





encoding_config=EncodingConfig(
        encoding_type=EncodingType.DELTA,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=80,
        binning_sigma=0.1,
        binning_k=1,
        delta_size=0.078
    )

ISI = 0.069

reservoir_config=ReservoirConfig(
    N=125,
    F_in=20,
    factor_inh=0.2, 
    C_EE=0.3,
    C_EI=0.2,
    C_IE=0.4,
    C_II=0.1,

    W_IN_MIN=0.0,
    W_IN_MAX=0.3,
    W_EE = 3*1,
    W_EI = 6*1,
    W_IE = -2*1,
    W_II = -2*1,
    spectral_radius=0.5,

    lamda=2.5,
    syn_delay= ISI * 1.14,

    lif_config=LIFConfig(
        v_threshold=1,
        leak_tc= ISI * 1.78, 
        v_rest=0,
        v_spike_delta=1.0,
        refractory_period= ISI * 0.22
    )
)

cv, cd, sep, slope, intercept = run_iteration(reservoir_config, encoding_config, "../tmp/manual/brian")

print(f"{'Metric':<10} | {'Value':<10}")
print("-" * 23)
print(f"{'CV':<10} | {cv:>10.2f}")
print(f"{'CD (R2)':<10} | {cd:>10.4f}")
print(f"{'SEP':<10} | {sep:>10.4f}")
print(f"{'Slope':<10} | {slope:>10.4f}")
print(f"{'Intercept':<10} | {intercept:>10.4f}")

zone_penalty = (1-slope)**2
stability_penalty = (intercept / ((cv+cd)/2+1e-6))**2
separation_reward = sep

loss = 100*zone_penalty + 10*stability_penalty - separation_reward

pprint(loss)

# plot_separation_ratio(results)

