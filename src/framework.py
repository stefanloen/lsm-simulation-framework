from collections import defaultdict
import multiprocessing
import os
import tempfile
import time
from typing import Optional
import analysis
import output
from plotting import plot_array_batches, plot_record, plot_spikes
import preprocessor
from preprocessor import get_records
import encoder
import numpy as np
from pprint import pprint

from interfaces import *
from netbuilder import Netbuilder
import reservoir
import train

def run(
        start_loc: NetworkLocation, 
        end_loc: NetworkLocation, 
        train_filter: float | np.ndarray | tuple[np.ndarray, list, bool] | Dict | None,
        validate_filter: float | np.ndarray | tuple[np.ndarray, list, bool] | Dict | None,
        max_processes: int,
        config: Config,
        noise_seed: int,
        brian_dir: str,
        cache_dir: str):

    if start_loc > end_loc:
        raise ValueError("Start location cannot be after end location.")
    
    if config.sim_config.encoder_trace and not config.sim_config.encoder_spikes:
        raise ValueError("Cannot get encoder trace without encoder spikes")

    # if config.sim_config.reservoir_trace and not config.sim_config.reservoir_spikes:
    #     raise ValueError("Cannot get reservoir trace without reservoir spikes")
    
    if config.sim_config.encoder_pca and not config.sim_config.encoder_trace:
        raise ValueError("Cannot do PCA without encoder trace")
    
    if config.sim_config.reservoir_pca and not config.sim_config.reservoir_trace:
        raise ValueError("Cannot do PCA without reservoir trace")
    
    if config.sim_config.cache_encoder and not config.sim_config.encoder_spikes:
        raise ValueError("Cannot cache encoder spikes without encoder spikes")
    
    if config.sim_config.cache_reservoir and not config.sim_config.reservoir_spikes:
        raise ValueError("Cannot cache reservoir spikes without reservoir spikes")
    
    if config.sim_config.cache_trained_params and not config.sim_config.train:
        raise ValueError("Cannot cache trained params without training")


    need_trained_params = start_loc <= NetworkLocation.XY and end_loc >= NetworkLocation.TRAIN
    do_train_params = need_trained_params and config.sim_config.train
    load_trained_params = need_trained_params and not config.sim_config.train

    sim_after_trained_params = end_loc > NetworkLocation.TRAIN and end_loc <= NetworkLocation.OUTPUT

    # Getting metadata
    records, global_record = get_records(config, 12) # NOISE SEED NOT USED!!!!
    print(f"Total samples in dataset: {len(records)}")
    records = filter_records(records, train_filter, validate_filter)

    remainder = len(records) % max_processes

    if remainder > 0 and len(records) > max_processes:
        records = records[:-remainder]
        print(f"Dropped {remainder} records.")

    if start_loc == NetworkLocation.DATA and end_loc == NetworkLocation.DATA:
        return records, global_record

    if not do_train_params:
            if load_trained_params:
                train.load_trained_params_cache(global_record, cache_dir)

            # Do a direct simulation
            records = run_multiprocess_simulation(
            max_processes, 
            records, 
            global_record, 
            start_loc, 
            end_loc,  
            config,
            noise_seed, 
            brian_dir,
            cache_dir,
        )
    else:
        # Split simulation in parts because of training

        # Run simulation before training
        run1_start_loc = start_loc
        run1_end_loc = NetworkLocation.XY
    
        records = run_multiprocess_simulation(
            max_processes, 
            records, 
            global_record, 
            run1_start_loc, 
            run1_end_loc, 
            config,
            noise_seed, 
            brian_dir,
            cache_dir,
        ) 

        train.get_model(config.train_config, records, global_record)

        if config.sim_config.cache_trained_params:
            train.store_trained_params_cache(global_record, cache_dir)
        
        # Run simulation after training
        if sim_after_trained_params:
            run2_start_loc = NetworkLocation.TRAIN
            run2_end_loc = end_loc
        
            records = run_multiprocess_simulation(
                max_processes, 
                records, 
                global_record, 
                run2_start_loc, 
                run2_end_loc,
                config,
                noise_seed, 
                brian_dir,
                cache_dir,
            )

        print("Done.")

    return records, global_record


def run_multiprocess_simulation(
    max_processes, 
    records, 
    global_record, 
    start_loc, 
    end_loc, 
    config, 
    noise_seed,
    brian_dir,
    cache_dir,
) -> list[Record]:
    
    total_records = len(records)
    process_count = min(total_records, max_processes)

    if process_count == 1:
        print(f"Running single process with {len(records)} samples")
        
        records = run_1_process_safe(
            0, # process_id
            records,
            global_record,
            start_loc,
            end_loc,
            config,
            noise_seed,
            brian_dir,
            cache_dir,
        )

        if records is not None:
            return records
        else:
            raise ValueError("records cannot be None")
    
    else:
        if total_records % process_count != 0:
            raise ValueError(
                f"Splitting failed: {total_records} records cannot be evenly "
                f"divided into {process_count} threads. (Remainder: {total_records % process_count})"
            )

        records_per_process = total_records // process_count

        records_splitted = [
            records[i : i + records_per_process] 
            for i in range(0, total_records, records_per_process)
        ]

        records_queue = multiprocessing.Queue()
        finish_lock = multiprocessing.Lock()
        manager = multiprocessing.Manager()
        confirmation_list = manager.list([False] * process_count)

        processes = []

        records_per_process= len(records_splitted[0])

        # Spawning processes
        for process_id in range(process_count):
            p = multiprocessing.Process(
                target=run_1_process_safe,
                args=(
                    process_id,
                    records_splitted[process_id], 
                    global_record, 
                    start_loc, 
                    end_loc, 
                    config,
                    noise_seed,
                    brian_dir,
                    cache_dir,
                    records_queue,
                    finish_lock,
                    confirmation_list
                )
            )
            p.start()
            processes.append(p)

        print(f"Spawned {process_count} processes with {records_per_process} samples per process.")

        # Collecting results
        records_indexed = [None] * process_count 
            
        try:
            for i in range(process_count):
                pid, data = records_queue.get() 
                records_indexed[pid] = data
                print(f"Main process received {pid}. Signaling worker to exit.")
                confirmation_list[pid] = True

        except Exception as e:
            print(f"Error during collection: {e}")
            raise
        finally:
            for p in processes:
                p.join()

        return_records: list[Record] = []

        # Flatten results
        for i, sublist in enumerate(records_indexed):
            if sublist is not None:
                return_records.extend(sublist)
                records_indexed[i] = None # Immediate memory release
            else:
                raise RuntimeError(f"A process failed to return valid Record data.")
        return return_records

def run_1_process_safe(
    process_id: int,
    records: list[Record],
    global_record: GlobalRecord,
    start_loc: NetworkLocation, 
    end_loc: NetworkLocation, 
    config: Config,
    noise_seed: int,
    brian_dir: str,
    cache_dir: str,
    record_queue: Optional[multiprocessing.Queue] = None,
    queue_lock: Optional[Any] = None,
    confirmation_list: Optional[Any] = None
) -> Optional[list[Record]]:
    try:
        # Run the actual simulation
        processed_data = run_1_process(
            records, global_record, start_loc, end_loc, 
            config, noise_seed, brian_dir, cache_dir
        )

        if record_queue is not None and queue_lock is not None and confirmation_list is not None:
            print(f"Process {os.getpid()} waiting for lock...")
            with queue_lock:
                record_queue.put((process_id, processed_data))
                print(f"Process {process_id} waiting for confirmation...")
                while not confirmation_list[process_id]:
                    time.sleep(0.1) 

                del processed_data 
                print(f"Process {os.getpid()} data sent and memory cleared.")
            return None
        else:
            return processed_data

    except Exception as e:
        print(f"!!! CRITICAL ERROR in Process {process_id} (PID {os.getpid()}): {e}")
        if record_queue is not None:
            record_queue.put((process_id, None)) 
        raise 

def run_1_process(
        records: List[Record],
        global_record: GlobalRecord,
        start_loc: NetworkLocation, 
        end_loc: NetworkLocation, 
        config: Config,
        noise_seed,
        brian_dir: str,
        cache_dir: str,
        ) -> list[Record]:

    simcfg = config.sim_config

    preprocessor_get_samples = start_loc == NetworkLocation.DATA
    preprocessor_fill_records = end_loc > NetworkLocation.DATA

    XY_load_cache = start_loc == NetworkLocation.XY
    output_load_cache = start_loc == NetworkLocation.OUTPUT

    sim_before_trained_params = start_loc < NetworkLocation.RESERVOIR_OUT and end_loc > NetworkLocation.PREPROCESSOR_OUT
    sim_after_trained_params = start_loc >= NetworkLocation.RESERVOIR_OUT and end_loc > NetworkLocation.MODEL_OUTPUT

    get_model_trace = config.sim_config.get_model_trace and start_loc < NetworkLocation.MODEL_OUTPUT and end_loc >= NetworkLocation.MODEL_OUTPUT
    get_model_y = config.sim_config.get_model_y and start_loc < NetworkLocation.MODEL_OUTPUT and end_loc >= NetworkLocation.MODEL_OUTPUT
    get_out_y = config.sim_config.get_out_y and start_loc < NetworkLocation.MODEL_OUTPUT and end_loc >= NetworkLocation.MODEL_OUTPUT

    # These might be already provided
    preprocessor_load_cache = start_loc == NetworkLocation.PREPROCESSOR_OUT and any(record.input_data is None for record in records)
    encoder_load_cache = start_loc == NetworkLocation.ENCODER_OUT and any(record.encoder_spikes is None for record in records)
    reservoir_load_cache = start_loc == NetworkLocation.RESERVOIR_OUT and any(record.reservoir_spikes is None for record in records)
    train_load_cache = start_loc <= NetworkLocation.TRAIN and end_loc > NetworkLocation.TRAIN and global_record.trained_params is None

    get_sample_times = (config.sim_config.get_res_x or
                    config.sim_config.get_true_y or
                    config.sim_config.get_model_y or
                    config.sim_config.get_out_y
                    ) and any(record.sample_times is None for record in records)
    get_res_x = config.sim_config.get_res_x and any(record.res_x is None for record in records)
    get_true_y = config.sim_config.get_true_y and any(record.y_true is None for record in records)

    if preprocessor_get_samples:
        preprocessor.get_samples(records, global_record, simcfg.cache_preprocessor, preprocessor_fill_records, config, cache_dir)

    if preprocessor_load_cache:
        preprocessor.load_cache(records, cache_dir, config)

    if encoder_load_cache:
        encoder.load_cache(records, cache_dir)

    if reservoir_load_cache:
        reservoir.load_cache(records, cache_dir)

    if XY_load_cache:
        train.load_XY_cache(records, cache_dir)

    if train_load_cache:
        train.load_trained_params_cache(global_record, cache_dir)

    if output_load_cache:
        output.load_cache(records, cache_dir)

    # Creating temporary folder
    os.makedirs(brian_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="brian_tmp_", dir=brian_dir) as working_dir:
        # Compiling brian if necessary
        if sim_before_trained_params or sim_after_trained_params:
            match start_loc:
                case NetworkLocation.DATA | NetworkLocation.PREPROCESSOR_OUT:
                    net_start_src = NetworkLocation.PREPROCESSOR_OUT
                case NetworkLocation.ENCODER_OUT:
                    net_start_src = NetworkLocation.ENCODER_OUT
                case NetworkLocation.RESERVOIR_OUT | NetworkLocation.TRAIN:
                    net_start_src = NetworkLocation.RESERVOIR_OUT
                case _:
                    raise ValueError(f"Unexpected start_loc: {start_loc}")
                
            match end_loc:
                case NetworkLocation.ENCODER_OUT:
                    net_end_dst = NetworkLocation.ENCODER_OUT
                case NetworkLocation.RESERVOIR_OUT | NetworkLocation.XY | NetworkLocation.TRAIN | NetworkLocation.MODEL_OUTPUT:
                    net_end_dst = NetworkLocation.RESERVOIR_OUT
                case NetworkLocation.OUTPUT:
                    net_end_dst = NetworkLocation.OUTPUT
                case _:
                    raise ValueError(f"Unexpected end_loc: {end_loc}")  
            
            # with compilation_lock:
            net = Netbuilder(
                records,
                global_record,
                net_start_src, 
                net_end_dst,
                config,
                noise_seed, 
                working_dir, 
                'cpp_standalone')
                
            print(f"Process {os.getpid()} finished compiling")

            net.run()
            print(f"Process {os.getpid()} finished Run")


        print(f"Process {os.getpid()} collecting and caching results")
        for record in records:
            # Getting encoder trace
            if config.sim_config.encoder_trace and record.encoder_trace is None:
                encoder_trace = analysis.get_trace(
                    config.sim_config,
                    record.get('encoder_spikes'),
                    global_record.get('duration'), 
                    config.train_config.encoder_trace_tau)
                record.set('encoder_trace', encoder_trace)

            # Getting Encoder PCA
            if config.sim_config.encoder_pca and record.encoder_rank is None:
                true_rank, _ = analysis.analyze_trace_rank(
                    config.sim_config,
                    record.get('encoder_trace'), 
                    config.train_config.encoder_trace_tau)
                record.set('encoder_rank', true_rank)

            # Getting reservoir trace
            if config.sim_config.reservoir_trace and record.reservoir_trace is None:
                reservoir_trace = analysis.get_trace(
                    config.sim_config,
                    record.get('reservoir_spikes'),
                    global_record.get('duration'), 
                    config.train_config.reservoir_trace_tau)
                record.set('reservoir_trace', reservoir_trace)

            # Getting Reservoir PCA
            if config.sim_config.reservoir_pca and record.reservoir_rank is None:
                true_rank, _ = analysis.analyze_trace_rank(
                    config.sim_config,
                    record.get('reservoir_trace'), 
                    config.train_config.reservoir_trace_tau)
                record.set('reservoir_rank', true_rank)

            # Getting samples for training and prediction
            if get_sample_times:
                train.get_sample_times(config, record, global_record)

            if get_res_x:
                train.get_res_x(config, record)

            if get_true_y:
                train.get_true_y(config, global_record, record)

            if get_model_trace:
                train.model_predict(config.train_config, global_record.get('trained_params'), record)
                
            if get_model_y:    
                train.get_model_y(config, record)

            if get_out_y:
                train.get_res_y(config, record)

            # Caching results
            if config.sim_config.cache_encoder:
                encoder.store_cache(record, cache_dir)
  
            if config.sim_config.cache_reservoir:
                reservoir.store_cache(record, cache_dir)
            
            if config.sim_config.cache_res_x:
                train.store_xy_cache(record, 'cache_res_x', cache_dir)

            if config.sim_config.cache_y_true:
                train.store_xy_cache(record, 'cache_y_true', cache_dir)

            if config.sim_config.cache_y_model:
                train.store_xy_cache(record, 'cache_y_model', cache_dir)

            if config.sim_config.cache_y_out:
                train.store_xy_cache(record, 'cache_y_out', cache_dir)  

            if config.sim_config.cache_output:
                output.store_cache(record, cache_dir)

        return records
    
def filter_records(
        records: List[Record],
        train_filter: float | np.ndarray | tuple[np.ndarray, list, bool] | Dict | None,
        validate_filter: float | np.ndarray | tuple[np.ndarray, list, bool] | Dict | None
    ) -> List[Record]:
    rng = np.random.default_rng(12)

    def apply_filter(current_records, filter_val):
        if filter_val is None:
            return []
        
        if isinstance(filter_val, float):
            stop_idx = int(len(current_records) * filter_val)
            return current_records[:stop_idx]

        elif isinstance(filter_val, np.ndarray):
            return np.array(current_records, dtype=object)[filter_val].tolist()
        
        elif isinstance(filter_val, dict):
            p_req = filter_val.get('patients')
            if isinstance(p_req, int): p_req = [p_req]
            p_set = {f"chb{i:02d}" for i in p_req} if p_req else None

            phase_groups = defaultdict(list)
            for record in current_records:
                if record.sample_metadata is None:
                    raise ValueError("No Sample metadata")
                if p_set and record.sample_metadata.metadata.get('patient_id') not in p_set:
                    continue
                phase_val = record.sample_metadata.metadata['eegphase']
                if not isinstance(phase_val, EEGPhase):
                    phase_val = EEGPhase(phase_val)
                phase_groups[phase_val].append(record)
            
            filtered_records = []
            for phase, requested_count in filter_val.items():
                if not isinstance(phase, EEGPhase):
                    continue
                available = phase_groups[phase]
                if len(available) < requested_count:
                    raise ValueError(f"Requested {requested_count} for {phase.name}, but only {len(available)} available.")
                selected = rng.choice(available, requested_count, replace=False)
                filtered_records.extend(selected)
            return filtered_records
        
        elif isinstance(filter_val, tuple):
            indices, eegphases, balance = filter_val
            filtered = [current_records[i] for i in indices 
                        if current_records[i].sample_metadata.metadata['eegphase'] in eegphases]
            if balance:
                phase_groups = defaultdict(list)
                for rec in filtered:
                    phase = rec.sample_metadata.metadata['eegphase']
                    phase_groups[phase].append(rec)
                min_count = min(len(phase_groups[p]) for p in phase_groups)
                balanced_records = []
                for phase in phase_groups:
                    group_samples = rng.choice(phase_groups[phase], min_count, replace=False)
                    balanced_records.extend(group_samples)
                return balanced_records
            return filtered
        return current_records

    train_recs = apply_filter(records, train_filter)
    for r in train_recs:
        r.sample_metadata.split = Split.TRAIN

    val_recs = apply_filter(records, validate_filter)
    for r in val_recs:
        r.sample_metadata.split = Split.VALIDATE

    return train_recs + val_recs