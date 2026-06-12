import numpy as np

from framework import (
    Config, Mode, run, NetworkLocation,
    CHBPreprocessingConfig, ReservoirConfig, 
    EncodingConfig, OutputConfig, 
    EncodingType, TrainConfig
)
from interfaces import (
    EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, Neuron, PoissonPreprocessingConfig, SimConfig, SimpleLI, SimpleLIF, SimpleSyn, SolverType, Synapse, 
    TaskType, WhiteNoiseConfig
)

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
    threshold_v=0.05,
    leak_tau=0.03,
    v_rest=0,
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

s_out = SimpleSyn(Fixed(1), 0)
synapse_out = Synapse(s_out, s_out, s_out, s_out, 0, 0)

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

preprocess = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,
        spike_threshold=1000,
        spike_threshold_tau=1,


        train=False,

        encoder_spikes=True, ######,
        encoder_trace=False,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=False,
        reservoir_trace=False,
        reservoir_pca=False,

        get_res_x = False,
        get_true_y = False,
        get_model_y = False,
        get_out_y = False,

        get_model_trace=False,

        cache_preprocessor=False,
        cache_encoder=True, ######
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
        window_size=10,
        channels=18,
        preictal_duration=900,
        postictal_duration=900,
        get_interictal=True,
        get_ictal=True,
        get_preictal=True,
        get_onset=False
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.BINNING,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=20,
        binning_sigma=0.1,
        binning_k=2.0,
        delta_size=0.078
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
        adam_epochs=10000,
        adam_learning_rate=0.00050,
        ridge_alphas= np.array([0.001]),
        qat_levels=3,
        patience=50,
        stride=0.1,
        washout_period=2,
        delays = np.array([0.0]),

        encoder_trace_tau=0.3,
        reservoir_trace_tau=0.3
    )
)


reservoir = Config(
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
        reservoir_spikes=True,
        reservoir_trace=False,
        reservoir_pca=False,

        get_res_x = False,
        get_true_y = False,
        get_model_y = False,
        get_out_y = False,

        get_model_trace=False,

        cache_preprocessor=False,
        cache_encoder=False,
        cache_reservoir=True,

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
        window_size=10,
        channels=18,
        preictal_duration=900,
        postictal_duration=900,
        get_interictal=True,
        get_ictal=True,
        get_preictal=False,
        get_onset=False
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.DELTA,
        binning_bins=10,
        binning_vmin=-1.0,
        binning_vmax=1.0,
        binning_rate=20,
        binning_sigma=0.1,
        binning_k=2,
        delta_size=0.078
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
        adam_epochs=10000,
        adam_learning_rate=0.00050,
        ridge_alphas= np.array([0.001]),
        qat_levels=3,
        patience=50,
        stride=0.1,
        washout_period=2,
        delays = np.array([0.0]),

        encoder_trace_tau=0.3,
        reservoir_trace_tau=0.3
    )
)

trainvalidate = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,

        spike_threshold=1000,
        spike_threshold_tau=1,

        train=True,

        encoder_spikes=False,
        encoder_trace=False,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=False,
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

        cache_trained_params=True,
        cache_output=False
    ),
    preprocessing_config=CHBPreprocessingConfig(
        cutoff_freq=40,
        notch_freqs=None,
        rereference=True,
        fixed_normalization=True,
        fixed_median=0,
        fixed_scale=2000,
        window_size=10,
        channels=18,
        preictal_duration=900,
        postictal_duration=900,
        get_interictal=True,
        get_ictal=True,
        get_preictal=False,
        get_onset=False
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
        qat_levels=100,
        patience=50,
        stride=0.1,
        washout_period=2,
        delays = np.array([0.0]),

        encoder_trace_tau=0.3,
        reservoir_trace_tau=0.3
    )
)