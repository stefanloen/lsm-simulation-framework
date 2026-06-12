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

config = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,
        spike_threshold=100,
        spike_threshold_tau=1,

        train=False,

        encoder_spikes=True,
        encoder_trace=True,
        encoder_pca=True,
        reservoir_v=False,
        reservoir_spikes=True,
        reservoir_trace=True,
        reservoir_pca=True,

        get_res_x = False,
        get_true_y = False,
        get_model_y = False,
        get_out_y = False,

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
    preprocessing_config=PoissonPreprocessingConfig(
        sample_count=8,
        random_duration=0,
        controlled_duration=100,
        silent_duration=0,
        similarity=0.95,
        channels=180,
        rate=80,
        min_isi = 0.001
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.BINNING,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=80,
        binning_sigma=0.1,
        binning_k=1
    ),
    reservoir_config=ReservoirConfig(
        N=125,
        F_in=4,
        factor_inh=0.2,
        C_EE=0.3,
        C_EI=0.2,
        C_IE=0.4,
        C_II=0.1,

        W_IN_MIN=-1.1,
        W_IN_MAX=1.1,
        W_EE = 3*0.6,
        W_EI = 6*0.6,
        W_IE = -2*0.6,
        W_II = -2*0.6,
        spectral_radius=1.0,

        lamda=2.0,
        syn_delay=0.003,

        lif_config=LIFConfig(
            v_threshold=1,
            leak_tc=0.05, 
            v_rest=0,
            v_spike_delta=1,
            refractory_period=0.002
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
        stride=1,
        washout_period=10,
        delays = np.array([0.0]),

        encoder_trace_tau=1.0,
        reservoir_trace_tau=1.0
    )
)