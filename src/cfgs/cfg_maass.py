import numpy as np

from framework import (
    Config, Mode, run, NetworkLocation,
    CHBPreprocessingConfig, ReservoirConfig, EncodingConfig, OutputConfig, 
    EncodingType, TrainConfig
)
from interfaces import (
    ClassNumber, EEGPhase, Fixed, Neuron, PoissonPreprocessingConfig, SimConfig, SimpleLI, SimpleLIF, SimpleSyn, SolverType, Synapse, 
    TaskType, Uniform, WhiteNoiseConfig, PoissonMaassPreprocessingConfig
)


s = SimpleSyn(Uniform(0, 0.5),0)
synapse_in = Synapse(s,s,s,s, 0, 0)

synapse_res = Synapse(
    EE=SimpleSyn(Fixed(0.6),0),
    EI=SimpleSyn(Fixed(0.3),0),
    IE=SimpleSyn(Fixed(-0.2),0),
    II=SimpleSyn(Fixed(-0.2),0),
    levels=0,
    write_noise=0
)

simplelif = SimpleLIF(
    threshold_v=1.0,
    leak_tau=0.03,
    v_rest=0,
    refractory_period=0.002
)

neuron_res = Neuron(simplelif, simplelif)

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
    spectral_radius=0.0001,
    synapse_res=synapse_res,

    neuron_res=neuron_res
)

config = Config(
    sim_config = SimConfig(
        brian_dt=0.0001,
        analysis_dt=0.01,

        spike_threshold=1000,
        spike_threshold_tau=1,

        train=True,

        encoder_spikes=True,
        encoder_trace=False,
        encoder_pca=False,
        reservoir_v=False,
        reservoir_spikes=True,
        reservoir_trace=True,
        reservoir_pca=False,

        get_res_x = True,
        get_true_y = True,
        get_model_y = True,
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
    preprocessing_config= PoissonMaassPreprocessingConfig(
        sample_count=25,
        channels=18,
        rate=20,
        lag=0.0,
        min_isi=0.0001,
        n_patterns= 2,
        n_repeat_min=10,
        n_repeat_max=50,
        pattern_duration= 0.500,
        num_seq_patterns= 200,
        jitter_std= 1.000,
        deletion_p=0.5,
        injection_rate=10,

        chunk=None ########################################################
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
        classes=[ClassNumber.ZERO, ClassNumber.ONE],
        balance=True,
        solver_type= SolverType.RIDGE,
        adam_epochs=1000,
        adam_learning_rate=0.010,
        ridge_alphas= np.array([1000.0]),
        qat_levels=256,
        patience=50,
        stride=0.01,
        washout_period=2,
        delays = np.array([0.0]),

        encoder_trace_tau=0.30,
        reservoir_trace_tau=0.30
    )
)