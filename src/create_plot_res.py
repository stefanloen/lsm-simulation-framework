# Reservoir config

import os
from pprint import pprint
import numpy as np
from typing import List, Tuple, cast
from interfaces import BONNPreprocessingConfig, Beta, Config, GlobalRecord, PoissonMaassPreprocessingConfig, PoissonPreprocessingConfig, Record, SampleMetadata, Split

# Reservoir config
import numpy as np

from framework import (
    Config, Mode, run, NetworkLocation,
    CHBPreprocessingConfig, ReservoirConfig, EncodingConfig, OutputConfig, 
    EncodingType, TrainConfig
)
from interfaces import (
    ClassNumber, EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, Neuron, PoissonPreprocessingConfig, SimConfig, SimpleLI, SimpleLIF, SimpleSyn, SolverType, Synapse, 
    TaskType, Uniform, WhiteNoiseConfig, PoissonMaassPreprocessingConfig
)

sim_EE_multiplier = 0.010
sim_EI_multiplier = 0.070
sim_IE_multiplier = 0.010
sim_II_multiplier = 0.040

global_strength = 1.25
in_strength = 0.25
res_strength = 1.0

in_x = in_strength * global_strength
res_x = res_strength * global_strength

s_out = SimpleSyn(Fixed(1), 0)
synapse_out = Synapse(s_out, s_out, s_out, s_out, 0, 0)

synapse_in = Synapse(
    SimpleSyn(Beta(18e-9*in_x, 0.9*18e-9*in_x),0),
    SimpleSyn(Beta(9e-9*in_x, 0.9*9e-9*in_x),0),
    SimpleSyn(Fixed(0),0),
    SimpleSyn(Fixed(0),0), 
    0, 
    0)

synapse_simple_res = Synapse(
    EE=SimpleSyn(Fixed(30e-9 *res_x* sim_EE_multiplier),0.001),
    EI=SimpleSyn(Fixed(60e-9 *res_x* sim_EI_multiplier),0.001),
    IE=SimpleSyn(Fixed(-19e-9 *res_x* sim_IE_multiplier),0.001),
    II=SimpleSyn(Fixed(-19e-9 *res_x* sim_II_multiplier),0.001),
    levels=0,
    write_noise=0
)

synapse_markram_res = Synapse(
    EE=MarkramSyn(Beta(30e-9*res_x, 0.9*30e-9*res_x), Beta(0.5, 0.45), Beta(1.1, 0.99), Beta(0.05, 0.045), 0.0015),
    EI=MarkramSyn(Beta(60e-9*res_x, 0.9*60e-9*res_x), Beta(0.05, 0.045), Beta(0.125, 0.1125), Beta(1.2, 1.08), 0.0008),
    IE=MarkramSyn(Beta(-19e-9*res_x, 0.9*19e-9*res_x), Beta(0.25, 0.225), Beta(0.7, 0.63), Beta(0.02, 0.018), 0.0008),
    II=MarkramSyn(Beta(-19e-9*res_x, 0.9*19e-9*res_x), Beta(0.32, 0.288), Beta(0.144, 0.1296), Beta(0.06, 0.054), 0.0008),
    
    levels=0,
    write_noise=0
)

# synapse_markram_res = Synapse(
#     EE=MarkramSyn(Uniform(30e-9*res_x, 30e-9*res_x), Uniform(0.5, 0.25), Uniform(1.1, 0.55), Uniform(0.05, 0.025), 0.0015),
#     EI=MarkramSyn(Uniform(60e-9*res_x, 60e-9*res_x), Uniform(0.05, 0.025), Uniform(0.125, 0.0625), Uniform(1.2, 0.6), 0.0008),
#     IE=MarkramSyn(Uniform(-19e-9*res_x, 19e-9*res_x), Uniform(0.25, 0.125), Uniform(0.7, 0.35), Uniform(0.02, 0.01), 0.0008),
#     II=MarkramSyn(Uniform(-19e-9*res_x, 19e-9*res_x), Uniform(0.32, 0.16), Uniform(0.144, 0.072), Uniform(0.06, 0.03), 0.0008),
    
#     levels=0,
#     write_noise=0
# )

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
    N=3*3*3,
    factor_inh=0.2,

    F_in=4,
    synapse_in=synapse_in,

    C_EE=0.3,
    C_EI=0.2,
    C_IE=0.4,
    C_II=0.1,

    lamda=1.0,
    spectral_radius=0.0001, # unused
    synapse_res=synapse_markram_res,

    neuron_res=maass_neuron_res
)

config = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,

        spike_threshold=1000,
        spike_threshold_tau=1,

        train=True,

        encoder_spikes=True,
        encoder_trace=True,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=True,
        reservoir_trace=True,
        reservoir_pca=False,

        get_res_x = True,
        get_true_y = True,
        get_model_y = True,
        get_out_y = True,

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
    preprocessing_config= BONNPreprocessingConfig(
        cutoff_freq=40,
        fixed_normalization= True,
        fixed_median= 0.0,
        fixed_scale= 0.001,
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.BINNING,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=80,
        binning_sigma=0.1,
        binning_k=1,
        delta_size=0.05
    ),
    reservoir_config=reservoir_config,
    output_config=OutputConfig(
        synapse=synapse_out,
        li_config=SimpleLI(
            leak_tau=0.3, 
            v_rest=0,
        )
    ),
    train_config=TrainConfig(
        task_type= TaskType.CLASSIFICATION,
        classes=[EEGPhase.INTERICTAL, EEGPhase.ICTAL],
        balance=True,
        solver_type= SolverType.RIDGE,
        adam_epochs=1000,
        adam_learning_rate=0.010,
        ridge_alphas= np.array([1000.0]),
        qat_levels=256,
        patience=50,
        stride=0.01,
        washout_period=2.0, ################################
        delays = np.array([0.0]),

        encoder_trace_tau=0.30,
        reservoir_trace_tau=0.3 ########################################################
    )
)

from sklearn.model_selection import KFold
import numpy as np
from pprint import pprint

from analysis import get_performance_metrics

# 1. Setup indices for the two classes (assuming 100 files each)
interictal_indices = np.arange(0, 100)
ictal_indices = np.arange(100, 200)

# 2. Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=12)

# Lists to store metrics from each fold
all_fold_reports = []
fold_accuracies = []

# 3. Cross-Validation Loop
# We zip the splits of both classes to ensure every fold is balanced
inter_splits = list(kf.split(interictal_indices))
ictal_splits = list(kf.split(ictal_indices))

for fold, ((inter_train, inter_val), (ictal_train, ictal_val)) in enumerate(zip(inter_splits, ictal_splits)):
    print(f"\n--- Starting Fold {fold + 1} / 5 ---")
    
    # Map back to original record indices
    train_idx = np.concatenate([interictal_indices[inter_train], ictal_indices[ictal_train]])
    val_idx = np.concatenate([interictal_indices[inter_val], ictal_indices[ictal_val]])
    
    # Run the simulation for this fold
    records, global_record = run(
        start_loc=NetworkLocation.DATA, 
        end_loc=NetworkLocation.OUTPUT, 
        train_filter=train_idx,
        validate_filter=val_idx,
        max_processes=10,
        config=config,
        noise_seed=12,
        brian_dir=f"../tmp/main/brian_fold_{fold}",
        cache_dir="../tmp/main/cache"
    )

    # Get metrics for this fold
    metrics = get_performance_metrics(
        TaskType.CLASSIFICATION,
        [EEGPhase.INTERICTAL, EEGPhase.ICTAL],
        records,
        'model',
        balance=True
    )
    
    report = metrics["classification_report"]
    all_fold_reports.append(report)
    fold_accuracies.append(report['accuracy'])
    
    print(f"Fold {fold + 1} Accuracy: {report['accuracy']:.4f}")

# 4. Final Summary
print("\n" + "="*30)
print(f"5-Fold CV Mean Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
print("="*30)