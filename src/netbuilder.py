from pprint import pprint
import brian2 as b2
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
import numpy as np
import warnings

from encoder import build_delta_encoding_layer, build_encoding_binning_layer, build_encoding_logbinning_layer, build_encoding_poisson_layer, build_encoding_threshold_layer, build_gaussian_binning_layer, build_hybrid_encoding_layer, build_noise_layer
from interfaces import Config, EncodingType, GlobalRecord, NetworkLocation, Record
from output import build_LI_output_layer
from plotting import print_active_percentage, visualise_connectivity
from reservoir import build_3D_reservoir_layer

class Netbuilder:
    def __init__(
            self, 
            # datas: list[np.ndarray] | None,
            # input_spikes: list[tuple[np.ndarray, np.ndarray]] | None,
            records: list[Record],
            global_record: GlobalRecord,
            start_loc: NetworkLocation,
            end_loc: NetworkLocation,
            config: Config,
            noise_seed,
            working_dir: str,
            device: str
            ):
        
        warnings.filterwarnings("ignore", message="overflow encountered in cast")
        if start_loc >= end_loc:
            raise ValueError("Start location cannot be after or same as end location when running the net")

        num_par = len(records)

        # Store some variables also necessary for run
        self.num_par = num_par
        self.working_dir = working_dir
        self.config = config
        self.noise_seed = noise_seed
        self.net = b2.Network()
        self.encoding_mon = None
        self.reservoir_mon = None
        self.reservoir_mon_v = None
        self.output_mon = None
        self.global_record = global_record
        self.records = records

        encoding_layer  = start_loc == NetworkLocation.PREPROCESSOR_OUT
        reservoir_layer = start_loc < NetworkLocation.RESERVOIR_OUT and end_loc >= NetworkLocation.RESERVOIR_OUT
        output_layer = end_loc == NetworkLocation.OUTPUT

        encoder_spike_monitor = end_loc == NetworkLocation.ENCODER_OUT or config.sim_config.encoder_spikes
        reservoir_spike_monitor = end_loc == NetworkLocation.RESERVOIR_OUT or config.sim_config.reservoir_spikes
        reservoir_v_monitor = config.sim_config.reservoir_v

        # Build network
        b2.device.reinit()
        b2.device.activate()
        b2.prefs.devices.cpp_standalone.extra_make_args_unix = ['-j4']
        b2.prefs.codegen.cpp.headers += ['"run.h"']
        b2.set_device(device, directory=self.working_dir, build_on_run=False)
        b2.seed(12)
        b2.defaultclock.dt = config.sim_config.brian_dt*1000*b2.ms

        # input layer
        if encoding_layer:  
            concatenated_array = np.concatenate([record.get('input_data') for record in records], axis=1)

            self.timedarray = b2.TimedArray(concatenated_array, dt=config.sim_config.analysis_dt*b2.second)
            match config.encoding_config.encoding_type:
                case EncodingType.POISSON:
                    input_neurons = build_encoding_poisson_layer(self.timedarray, num_par)
                case EncodingType.THRESHOLD:
                    input_neurons = build_encoding_threshold_layer(self.timedarray, num_par)
                case EncodingType.NOISE:
                    input_neurons = build_noise_layer(18, num_par, 50, 100)
                case EncodingType.BINNING:
                    input_neurons = build_encoding_binning_layer(
                        self.timedarray,
                        config.encoding_config.binning_bins,
                        config.encoding_config.binning_vmin,
                        config.encoding_config.binning_vmax,
                        config.encoding_config.binning_rate
                        )
                case EncodingType.BINNING_GAUSSIAN:
                    input_neurons = build_gaussian_binning_layer(
                        self.timedarray,
                        config.encoding_config.binning_bins,
                        config.encoding_config.binning_vmin,
                        config.encoding_config.binning_vmax,
                        config.encoding_config.binning_rate,
                        config.encoding_config.binning_sigma
                        )
                case EncodingType.LOGBINNING:
                    input_neurons = build_encoding_logbinning_layer(
                        self.timedarray,
                        config.encoding_config.binning_bins,
                        config.encoding_config.binning_vmin,
                        config.encoding_config.binning_vmax,
                        config.encoding_config.binning_rate,
                        config.encoding_config.binning_k
                    )
                case EncodingType.HYBRID:
                    input_neurons = build_hybrid_encoding_layer(
                        self.timedarray
                    )
                case EncodingType.DELTA:
                    input_neurons = build_delta_encoding_layer(
                        self.timedarray,
                        self.config.encoding_config.delta_size
                    )

            self.net.add(input_neurons)

            if encoder_spike_monitor:
                self.encoding_mon = b2.SpikeMonitor(input_neurons, name='encoding_monitor')
                self.net.add(self.encoding_mon)
        else:
            match start_loc:
                case NetworkLocation.ENCODER_OUT:
                    input_spikes = [record.get('encoder_spikes') for record in records]
                case NetworkLocation.RESERVOIR_OUT | NetworkLocation.TRAIN:
                    input_spikes = [record.get('reservoir_spikes') for record in records]
                case _:
                    raise ValueError(f"Unexpected start loc: {start_loc}")

            all_times = []
            all_indices = []
            neuron_offset = 0 

            n = 0
            for entry in input_spikes: 
                (t_coords, i_coords), n = entry
                all_indices.append(i_coords + neuron_offset)
                all_times.append(t_coords)
                neuron_offset += n

            final_indices = np.concatenate(all_indices)
            final_times = np.concatenate(all_times) * b2.second

            total_neurons = n * len(input_spikes)
            input_neurons = b2.SpikeGeneratorGroup(total_neurons, final_indices, final_times)
            self.net.add(input_neurons)

        reservoir_neurons = None
        # Reservoir layer
        if reservoir_layer:
            reservoir_neurons, input_reservoir_syn, reservoir_syn = build_3D_reservoir_layer(config.reservoir_config, input_neurons, num_par, noise_seed)
            self.net.add(reservoir_neurons)
            self.net.add(input_reservoir_syn)
            self.net.add(reservoir_syn)

            # BEUN
            self.input_reservoir_syn = input_reservoir_syn
            self.reservoir_syn = reservoir_syn

            if reservoir_spike_monitor:
                self.reservoir_mon = b2.SpikeMonitor(reservoir_neurons, name='reservoir_monitor')
                self.net.add(self.reservoir_mon)
                
            if reservoir_v_monitor:
                self.reservoir_mon_v = b2.StateMonitor(reservoir_neurons, 'v', record=12)
                self.net.add(self.reservoir_mon_v)

            # Spike monitor for early stopping
            @b2.implementation(CPPStandaloneCodeObject, r'''
            double stop_if_too_high(double rate, double threshold) {
                if (rate>threshold) {
                    printf("Warning: Spike threshold reached, stopping early\n");
                    brian_end();  // save all data to disk
                    std::exit(0);
                }
                return 0.0;
            }
            ''')
            @b2.implementation('numpy', discard_units=True)
            @b2.check_units(rate=b2.Hz, threshold=b2.Hz, result=1)
            def stop_if_too_high(rate, threshold):
                if rate > threshold:
                    b2.stop()

            stop_neuron = b2.NeuronGroup(
                N=1, 
                model='drate/dt = -rate/tau: Hz', 
                threshold='True', 
                reset='',
                method='euler',
                namespace={
                    'tau': config.sim_config.spike_threshold_tau*b2.second,
                    'threshold': config.sim_config.spike_threshold*num_par* b2.Hz
                })
            
            stop_syn = b2.Synapses(
                reservoir_neurons, 
                stop_neuron, 
                on_pre='rate += 1.0/tau/N_incoming',
                namespace={
                    'tau': config.sim_config.spike_threshold_tau*b2.second
                })
            
            stop_syn.connect()
            self.net.add(stop_neuron)
            self.net.add(stop_syn)
            stop_neuron.run_regularly('dummy = stop_if_too_high(rate, threshold)', when='after_synapses')

        # Output layer
        if output_layer:
            trained_params = global_record.get('trained_params')
            
            # if np.array(trained_params['fc.weight'], dtype=np.float64).shape[0] != global_record.get('num_classes'):
            #     raise ValueError("Trained parameters class count is incorrect")
            
            if reservoir_neurons is not None:
                output_neurons, reservoir_output_syn = build_LI_output_layer(config.output_config, trained_params, reservoir_neurons, num_par, noise_seed)
            else:
                output_neurons, reservoir_output_syn = build_LI_output_layer(config.output_config, trained_params, input_neurons, num_par, noise_seed)

            self.net.add(output_neurons)
            self.net.add(reservoir_output_syn)

            # # BEUN
            # self.reservoir_output_syn = reservoir_output_syn

            # Output layer always needs monitoring
            self.output_mon = b2.StateMonitor(output_neurons, 'v', dt=config.sim_config.analysis_dt*b2.second, record=True)
            self.net.add(self.output_mon)


        self.net.run(duration=global_record.get('duration') * b2.second, report='text', report_period=10*b2.second, profile=False)

        # For future reference, dynamically loading the input cache
        # with open('insertcode.cpp', 'r') as f:
        #     insertcode = f.read()
        # b2.device.insert_code('main', insertcode)

        b2.device.build(directory=working_dir, compile=True, run=False, debug=False)
        self.device = b2.get_device()

    def run(
            self,
        ): 
        
        warnings.filterwarnings("ignore", message="abstract")

        from brian2.devices import device_module
        device_module.active_device = self.device
        b2.device.run()
        
        # BEUN
        # visualise_connectivity(self.input_reservoir_syn)
        # visualise_connectivity(self.reservoir_syn)

        # if hasattr(self, 'reservoir_syn'):
        #     print_active_percentage(self.reservoir_syn, self.config.reservoir_config.N, self.num_par)

        # Collect results
        encoder_t, encoder_i, encoder_N = None, None, None
        reservoir_t, reservoir_i, reservoir_N = None, None, None
        reservoir_v_t, reservoir_v = None, None
        output_v = None

        if self.encoding_mon is not None:
            # This loads the data into RAM according to Gemini
            encoder_t = np.array(self.encoding_mon.t) 
            encoder_i = np.array(self.encoding_mon.i)
            encoder_N = self.encoding_mon.source.N // self.num_par #type: ignore

        if self.reservoir_mon is not None:
            # This loads the data into RAM according to Gemini
            reservoir_t = np.array(self.reservoir_mon.t) 
            reservoir_i = np.array(self.reservoir_mon.i)
            reservoir_N = self.reservoir_mon.source.N // self.num_par #type: ignore

        if self.reservoir_mon_v is not None:
            reservoir_v = np.array(self.reservoir_mon_v.v[0]) 
            reservoir_v_t = np.array(self.reservoir_mon_v.t)    

        if self.output_mon is not None:
            output_v = np.array(self.output_mon.v)

        for i, record in enumerate(self.records):
            # Getting spike encoder monitor
            if encoder_t is not None and encoder_i is not None and encoder_N is not None:
                start_idx = i * encoder_N
                end_idx = (i + 1) * encoder_N
                mask = (encoder_i >= start_idx) & (encoder_i < end_idx)
                record.encoder_spikes = ((encoder_t[mask], encoder_i[mask] - start_idx), encoder_N)

                if len(encoder_t[mask]>0):
                    record.encoder_rate = len(encoder_t[mask]) / np.max(encoder_t[mask])
                else:
                    record.encoder_rate = 0

            # Getting spike reservoir monitor
            if reservoir_t is not None and reservoir_i is not None and reservoir_N is not None:
                start_idx = i * reservoir_N
                end_idx = (i + 1) * reservoir_N
                mask = (reservoir_i >= start_idx) & (reservoir_i < end_idx)
                record.reservoir_spikes = ((reservoir_t[mask], reservoir_i[mask] - start_idx), reservoir_N)
                
                if len(reservoir_t[mask]>0):
                    record.reservoir_rate = len(reservoir_t[mask]) / np.max(reservoir_t[mask])
                else:
                    record.reservoir_rate = 0

            # Getting voltage output monitor
            if output_v is not None:
                num_classes = output_v.shape[0] // self.num_par
                out_start = i * num_classes
                out_end = (i + 1) * num_classes
                record.output_v = output_v[out_start:out_end, :]

            # Getting reservoir voltage monitors (only on first neuron of first batch)
            if i == 0 and reservoir_v_t is not None and reservoir_v is not None:
                record.reservoir_v = (reservoir_v_t, reservoir_v)