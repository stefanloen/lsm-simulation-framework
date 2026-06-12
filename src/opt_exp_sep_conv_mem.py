import json
import os
import shutil
from matplotlib import pyplot as plt
import optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
import copy

from analysis import get_convergence_factor, get_performance_metrics, get_separation_factor
from framework import run
from interfaces import SimConfig, Config, EncodingConfig, EncodingType, LIConfig, LIFConfig, Mode, NetworkLocation, OutputConfig, PoissonPreprocessingConfig, ReservoirConfig, SolverType, TaskType, TrainConfig, WhiteNoiseConfig
from contextlib import redirect_stdout

white_noise_config = WhiteNoiseConfig(
        sample_count=20,
        duration=100,
        cutoff_freq=0.5
    )

poisson_preprocessing_config = PoissonPreprocessingConfig(
        sample_count=100,
        random_duration=50,
        controlled_duration=50,
        similarity=1,
        channels=1,
        rate=50,
        min_isi = 0.001
    )

config = Config(
    sim_config = SimConfig(
        brian_dt=0.001,
        analysis_dt=0.01,

        train=True,

        encoder_spikes=False,
        encoder_trace=False,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=True,
        reservoir_trace=True,
        reservoir_pca=False,

        get_res_x = True,
        get_true_y = False,
        get_model_y = False,
        get_out_y = False,

        get_model_trace=True,

        cache_preprocessor=False,
        cache_encoder=False,
        cache_reservoir=False,
        cache_res_x=False,
        cache_y_true=False,
        cache_y_model=False,
        cache_y_out=False,

        cache_trained_params=False,
        cache_output=False
    ),
    preprocessing_config= white_noise_config,
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.BINNING_GAUSSIAN,
        binning_bins=10,
        binning_vmin=0.0,
        binning_vmax=1.0,
        binning_rate=50,
        binning_sigma=0.1
    ),
    reservoir_config=ReservoirConfig(
        N=125,
        input_reservoir_p=0.46,
        factor_inhibitory=0.2,
        C_EE=0.3,
        C_EI=0.2,
        C_IE=0.4,
        C_II=0.1,
        lamda=0.15,
        syn_delay=0.0875,

        lif_config=LIFConfig(
            v_threshold=1,
            leak_tc=3, 
            v_rest=0,
            v_spike_delta=0.09,
            refractory_period=0.0001
        )
    ),
    output_config=OutputConfig(
        li_config=LIConfig(
            leak_tc=3,
            v_rest=0.5,
            v_spike_delta= 0.09
        )
    ),
    train_config=TrainConfig(
        task_type= TaskType.REGRESSION,
        solver_type= SolverType.RIDGE,
        adam_epochs=10000,
        adam_learning_rate=0.00050,
        ridge_alphas= np.array([0.0001, 0.001, 0.1, 1.0, 10.0, 100.0]),
        stride=1,
        washout_period=50,
        delays = np.arange(0.0, 4.0, 0.5),

        encoder_trace_tau=1.0,
        reservoir_trace_tau=1.0
    )
)

PARAM_RANGES = {
    "binning_sigma": (0.05, 0.3),
    "lamda": (0.1, 2.0),
    "leak_tc": (0.1, 100.0),
    "v_spike_delta": (0.01, 1.0),
    "input_p": (0.05, 0.5),
    "syn_delay": (0.01, 2.0),
    "trace_tau": (0.1, 5)
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

def optimize_big_boss(trial: optuna.trial.Trial):
    #trial_root = os.path.join("..", "tmp", "trial", f"trial_{trial.number}")
    trial_root = os.path.join("/dev/shm", "tmp", "trial", f"trial_{trial.number}")
    brian_dir = os.path.join(trial_root, "brian")
    cache_dir = os.path.join(trial_root, "cache")

    os.makedirs(brian_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    global global_best_score
    params = {name: trial.suggest_float(name, low, high) 
              for name, (low, high) in PARAM_RANGES.items()}

    current_config = copy.deepcopy(config) 

    # Architecture settings
    current_config.encoding_config.binning_sigma = params['binning_sigma']
    current_config.reservoir_config.lamda = params['lamda']
    current_config.reservoir_config.input_reservoir_p = params['input_p']
    current_config.reservoir_config.lif_config.leak_tc = params['leak_tc']
    current_config.reservoir_config.lif_config.v_spike_delta = params['v_spike_delta']
    current_config.reservoir_config.syn_delay = params['syn_delay']
    current_config.train_config.encoder_trace_tau = params['trace_tau']
    current_config.train_config.reservoir_trace_tau = params['trace_tau']

    # Simulation settings
    setattr(current_config.sim_config, 'train', False )
    setattr(current_config.sim_config, 'get_model_trace', False)
    setattr(current_config.sim_config, 'encoder_spikes', True)
    setattr(current_config.sim_config, 'encoder_trace', True)
    setattr(current_config.sim_config, 'encoder_pca', True)
    setattr(current_config.sim_config, 'reservoir_spikes', True)
    setattr(current_config.sim_config, 'reservoir_trace', True)
    setattr(current_config.sim_config, 'reservoir_pca', True)
    setattr(current_config.sim_config, 'get_res_x', False)
    setattr(current_config.sim_config, 'get_true_y', False)
    setattr(current_config.sim_config, 'get_model_y', False)
    current_config.preprocessing_config = poisson_preprocessing_config
    setattr(current_config.preprocessing_config, 'random_duration', 0)
    setattr(current_config.preprocessing_config, 'controlled_duration', 100)
    setattr(current_config.preprocessing_config, 'similarity', 0.95)
    setattr(current_config.preprocessing_config, 'sample_count', 8)
    setattr(current_config.train_config, 'task_type', TaskType.CLASSIFICATION)
    setattr(current_config.train_config, 'delays', np.array([0.0]))

    try:
        records, _ = run(
            start_loc=NetworkLocation.ENCODER_OUT, 
            end_loc=NetworkLocation.RESERVOIR_OUT, 
            sample_filter=np.arange(8), # Ensure this is an even number for pairs
            max_processes=1,
            config=current_config,
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

        last_n = int(10.0 / config.sim_config.analysis_dt)

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
        setattr(current_config.sim_config, 'train', False )
        setattr(current_config.sim_config, 'get_model_trace', False)
        setattr(current_config.sim_config, 'encoder_spikes', False)
        setattr(current_config.sim_config, 'encoder_trace', False)
        setattr(current_config.sim_config, 'encoder_pca', False)
        setattr(current_config.sim_config, 'reservoir_spikes', True)
        setattr(current_config.sim_config, 'reservoir_trace', True)
        setattr(current_config.sim_config, 'reservoir_pca', False)
        setattr(current_config.sim_config, 'get_res_x', False)
        setattr(current_config.sim_config, 'get_true_y', False)
        setattr(current_config.sim_config, 'get_model_y', False)
        current_config.preprocessing_config = poisson_preprocessing_config
        setattr(current_config.preprocessing_config, 'random_duration', 50)
        setattr(current_config.preprocessing_config, 'controlled_duration', 50)
        setattr(current_config.preprocessing_config, 'similarity', 1.0)
        setattr(current_config.preprocessing_config, 'sample_count', 8)
        setattr(current_config.train_config, 'task_type', TaskType.CLASSIFICATION)
        setattr(current_config.train_config, 'delays', np.array([0.0]))

        records, _ = run(
            start_loc=NetworkLocation.ENCODER_OUT, 
            end_loc=NetworkLocation.RESERVOIR_OUT, 
            sample_filter=np.arange(8), # Ensure this is an even number for pairs
            max_processes=1,
            config=current_config,
            brian_dir=brian_dir,
            cache_dir=cache_dir
        )

        convergence_factors = []
        for i in range(0, len(records), 2):
            res_a = records[i]
            res_b = records[i+1]

            # Ensure traces exist
            conv_f, _ = get_convergence_factor(
                run1=res_a.get('reservoir_trace'),
                run2=res_b.get('reservoir_trace'),
                last_n=last_n
            )
            convergence_factors.append(conv_f)

        convergence_factor = np.mean(convergence_factors)

        # Memory
        setattr(current_config.sim_config, 'train', True )
        setattr(current_config.sim_config, 'get_model_trace', True)
        setattr(current_config.sim_config, 'encoder_spikes', False)
        setattr(current_config.sim_config, 'encoder_trace', False)
        setattr(current_config.sim_config, 'encoder_pca', False)
        setattr(current_config.sim_config, 'reservoir_spikes', True)
        setattr(current_config.sim_config, 'reservoir_trace', True)
        setattr(current_config.sim_config, 'reservoir_pca', False)
        setattr(current_config.sim_config, 'get_res_x', True)
        setattr(current_config.sim_config, 'get_true_y', True)
        setattr(current_config.sim_config, 'get_model_y', True)
        current_config.preprocessing_config = white_noise_config
        setattr(current_config.train_config, 'task_type', TaskType.REGRESSION)
        setattr(current_config.train_config, 'delays', np.arange(0.0, 10.0, 0.5))

        records, _ = run(
            start_loc=NetworkLocation.PREPROCESSOR_OUT, 
            end_loc=NetworkLocation.MODEL_OUTPUT, 
            sample_filter=np.arange(20),
            max_processes=1,
            config=current_config,
            brian_dir=brian_dir,
            cache_dir=cache_dir
        )

        model_performance = get_performance_metrics(
            TaskType.REGRESSION,
            records,
            'model')
        
        memory_sum_r2_scores = model_performance['sum_r2_scores']

        # Collecting results
        u_exp = get_utility(expansion_factor, low=1.0, high=100.0)
        u_sep = get_utility(separation_factor, low=1.0, high=100.0)
        u_conv = get_utility(convergence_factor, low=1.0, high=100.0)
        u_mem = get_utility(memory_sum_r2_scores, low=0.1, high=20.0)

        score = u_exp * u_sep * u_conv * u_mem

        trial.set_user_attr("expansion_factor", float(expansion_factor))
        trial.set_user_attr("separation_factor", float(separation_factor))
        trial.set_user_attr("convergence_factor", float(convergence_factor))
        trial.set_user_attr('memory_r2_sum', float(memory_sum_r2_scores))
        trial.set_user_attr("score", float(score))
        
        return u_exp, u_sep, u_conv, u_mem

    except Exception as e:
        print(f"Trial failed due to: {e}")
        return 0.0, 0.0, 0.0, 0.0
    
    finally:
        # 2. Cleanup: Delete the unique directory to save space
        # Comment this out if you need to inspect files for failed trials
        if os.path.exists(trial_root):
            shutil.rmtree(trial_root)

os.makedirs("../optuna", exist_ok=True)

def run_optimization(_):
    study = optuna.create_study(
        study_name="opt_exp_sep_conv_mem_V1",
        storage="sqlite:///../optuna/optuna.db", 
        directions=["maximize", "maximize", "maximize", "maximize"],
        load_if_exists=True
    )
    study.optimize(optimize_big_boss) 

with Pool(processes=16) as pool:
    pool.map(run_optimization, range(16))