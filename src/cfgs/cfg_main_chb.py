import numpy as np

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

ISI = 0.069

config = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,
        spike_threshold=1000,
        spike_threshold_tau=1,

        train=False,

        encoder_spikes=False,
        encoder_trace=False,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=False,
        reservoir_trace=False,
        reservoir_pca=False,

        get_res_x = False,
        get_true_y = True,
        get_model_y = False,
        get_out_y = True,

        get_model_trace=False,

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
    preprocessing_config=CHBPreprocessingConfig(
        cutoff_freq=40,
        notch_freqs=None,
        rereference=True,
        fixed_normalization=True,
        fixed_median=0,
        fixed_scale=2000,
        window_size=1200,
        channels=18,
        preictal_duration=900,
        postictal_duration=900,
        get_interictal=True,
        get_ictal=True,
        get_preictal=True,
        get_onset=False
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.DELTA,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=80,
        binning_sigma=0.1,
        binning_k=1,
        delta_size=0.078
    ),

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
    ),
    output_config=OutputConfig(
        li_config=LIConfig(
            leak_tc=1,
            v_rest=0,
            v_spike_delta = 1
        )
    ),
    train_config=TrainConfig(
        task_type= TaskType.CLASSIFICATION,
        classes=[EEGPhase.INTERICTAL, EEGPhase.ICTAL],
        balance=True,
        solver_type= SolverType.RIDGE,
        adam_epochs=10000,
        adam_learning_rate=0.00050,
        ridge_alphas= np.array([0.001]),
        qat_levels=3,
        patience=50,s
        stride=0.1,
        washout_period=2,
        delays = np.array([0.0]),

        encoder_trace_tau=1.0,
        reservoir_trace_tau=1.0
    )
)