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
from plotting import plot_record, plot_reservoir_convergence

from cfgs import cfg_metric

# config = Config(
#     sim_config = SimConfig(
#         brian_dt=0.001,
#         analysis_dt=0.01,

#         train=False,

#         encoder_spikes=True,
#         encoder_trace=True,
#         encoder_pca=True,
#         reservoir_v=False,
#         reservoir_spikes=True,
#         reservoir_trace=True,
#         reservoir_pca=True,

#         get_res_x = False,
#         get_true_y = False,
#         get_model_y = False,
#         get_out_y = False,

#         get_model_trace=False,

#         cache_preprocessor=False,
#         cache_encoder=False,
#         cache_reservoir=False,
#         cache_res_x=False,
#         cache_y_true=False,
#         cache_y_model=False,
#         cache_y_out=False,

#         cache_trained_params=False,
#         cache_output=False
#     ),
#     preprocessing_config= PoissonPreprocessingConfig(
#         sample_count=100,
#         random_duration=50,
#         controlled_duration=50,
#         similarity=1,
#         channels=1,
#         rate=50,
#         min_isi = 0.001
#     ),
#     encoding_config=EncodingConfig(
#         encoding_type=EncodingType.BINNING_GAUSSIAN,
#         binning_bins=100,
#         binning_vmin=0.0,
#         binning_vmax=1.0,
#         binning_rate=50,
#         binning_sigma=0.111
#     ),
#     reservoir_config=ReservoirConfig(
#         N=125,
#         input_reservoir_p=0.42,
#         factor_inhibitory=0.2,
#         C_EE=0.3,
#         C_EI=0.2,
#         C_IE=0.4,
#         C_II=0.1,
#         lamda=0.298,
#         syn_delay=0.773,

#         lif_config=LIFConfig(
#             v_threshold=1,
#             leak_tc=5.46, 
#             v_rest=0,
#             v_spike_delta=0.596,
#             refractory_period=0.0001
#         )
#     ),
#     output_config=OutputConfig(
#         li_config=LIConfig(
#             leak_tc=3,
#             v_rest=0.5,
#             v_spike_delta= 0.09
#         )
#     ),
#     train_config=TrainConfig(
#         task_type= TaskType.REGRESSION,
#         solver_type= SolverType.RIDGE,
#         adam_epochs=10000,
#         adam_learning_rate=0.00050,
#         ridge_alphas= np.array([1]),
#         stride=0.1,
#         washout_period=50,
#         delays = np.arange(0.0, 5, 0.1),

#         encoder_trace_tau=0.99,
#         reservoir_trace_tau=0.99
#     )
# )

records, global_record = run(
    start_loc=NetworkLocation.ENCODER_OUT, 
    end_loc=NetworkLocation.RESERVOIR_OUT, 
    sample_filter= np.arange(0,2),#None, # np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    max_processes=1,
    config=cfg_metric.config,
    brian_dir='../tmp/main/brian',
    cache_dir='../tmp/main/cache') # './' for current directory. Could also be '/dev/shm/brian' to spare SSD,

for i in range(0, len(records), 2):
    reservoirA_trace = records[i].reservoir_trace
    reservoirB_trace = records[i+1].reservoir_trace
    encoderA_trace = records[i].encoder_trace
    encoderB_trace = records[i+1].encoder_trace
    if reservoirA_trace is not None and reservoirB_trace is not None and encoderA_trace is not None and encoderB_trace is not None:
        # score, distances = get_convergence_factor(reservoirA_trace, reservoirB_trace, int(10.0 / cfg_metric.config.sim_config.analysis_dt))
        # print(f"Pair {i//2} | generalization_score: {score:.4f}")
        
        # plot_reservoir_convergence(
        #     dist_profile=distances, 
        #     labels=records[i].label,
        #     sampling_freq=int(1.0 / cfg_metric.config.sim_config.analysis_dt), 
        #     pair_id=i//2
        # )  


        separation_factor = get_separation_factor(encoderA_trace, encoderB_trace, reservoirA_trace, reservoirB_trace, int(10 * global_metadata.sampling_freq))
        print(f"Pair {i//2} | separation_factor: {separation_factor:.4f}")
        plot_reservoir_convergence(
            dist_profile=encoder_distances, 
            labels=netresults[i].label,
            sampling_freq=global_metadata.sampling_freq, 
            pair_id=i//2
        )  