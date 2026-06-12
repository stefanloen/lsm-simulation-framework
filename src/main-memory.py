import brian2 as b2
import numpy as np
from multiprocessing import Pool
from pprint import pprint

from framework import (
    Config, Mode, run, NetworkLocation,
    CHBPreprocessingConfig, ReservoirConfig, 
    LIFConfig, LIConfig, EncodingConfig, OutputConfig, 
    EncodingType, TrainConfig
)
from interfaces import (
    PoissonPreprocessingConfig, SimConfig, SolverType, 
    TaskType, WhiteNoiseConfig
)
from analysis import (
    calculate_pca_distance, get_convergence_factor, get_separation_factor,
    get_performance_metrics
)
from plotting import plot_memory_curve, plot_record, plot_reservoir_convergence

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
    preprocessing_config= WhiteNoiseConfig(
        sample_count=100,
        duration=100,
        cutoff_freq=0.5
    ),
    encoding_config=EncodingConfig(
        encoding_type=EncodingType.BINNING_GAUSSIAN,
        binning_bins=100,
        binning_vmin=0.0,
        binning_vmax=1.0,
        binning_rate=50,
        binning_sigma=0.111
    ),
    reservoir_config=ReservoirConfig(
        N=125,
        input_reservoir_p=0.42,
        factor_inhibitory=0.2,
        C_EE=0.3,
        C_EI=0.2,
        C_IE=0.4,
        C_II=0.1,
        lamda=0.298,
        syn_delay=0.773,

        lif_config=LIFConfig(
            v_threshold=1,
            leak_tc=5.46, 
            v_rest=0,
            v_spike_delta=0.596,
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
        ridge_alphas= np.array([1]),
        stride=0.1,
        washout_period=50,
        delays = np.arange(0.0, 5, 0.1),

        encoder_trace_tau=0.99,
        reservoir_trace_tau=0.99
    )
)

records, global_record = run(
    start_loc=NetworkLocation.PREPROCESSOR_OUT, 
    end_loc=NetworkLocation.MODEL_OUTPUT, 
    sample_filter= np.arange(0,100),#None, # np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    max_processes=1,
    config=config,
    brian_dir='../tmp/main/brian',
    cache_dir='../tmp/main/cache') # './' for current directory. Could also be '/dev/shm/brian' to spare SSD,


model_performance = get_performance_metrics(
    config.train_config.task_type,
    records,
    'model')

pprint(model_performance)

plot_memory_curve(config.train_config.delays, model_performance['r2_scores'])

for record in records: 
    print(record.encoder_rank)
    print(record.reservoir_rank)
    plot_record(config, record, global_record, f"Sample {record.get('sample_metadata').sample_id}")





# for i in range(0, len(netresults), 2):
#     reservoirA_trace = netresults[i].reservoir_trace
#     reservoirB_trace = netresults[i+1].reservoir_trace
#     encoderA_trace = netresults[i].encoder_trace
#     encoderB_trace = netresults[i+1].encoder_trace
#     if reservoirA_trace is not None and reservoirB_trace is not None and encoderA_trace is not None and encoderB_trace is not None:
#         score, distances = get_convergence_factor(reservoirA_trace, reservoirB_trace, int(10 * global_metadata.sampling_freq))
#         print(f"Pair {i//2} | generalization_score: {score:.4f}")
        
#         plot_reservoir_convergence(
#             dist_profile=distances, 
#             labels=netresults[i].label,
#             sampling_freq=global_metadata.sampling_freq, 
#             pair_id=i//2
#         )  


#         # separation_factor = get_separation_factor(encoderA_trace, encoderB_trace, reservoirA_trace, reservoirB_trace, int(10 * global_metadata.sampling_freq))
#         # print(f"Pair {i//2} | separation_factor: {separation_factor:.4f}")
#         # plot_reservoir_convergence(
#         #     dist_profile=encoder_distances, 
#         #     labels=netresults[i].label,
#         #     sampling_freq=global_metadata.sampling_freq, 
#         #     pair_id=i//2
#         # )  