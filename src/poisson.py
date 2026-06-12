import os
from pprint import pprint
from typing import List, Tuple, cast
import numpy as np
import encoder
from interfaces import Config, GlobalRecord, PoissonPreprocessingConfig, Record, SampleMetadata, Split

CACHE_ENCODER_PATH = '../cache/encoder'

def generate_poisson_metadata(
    global_config: Config,
) -> Tuple[List[Record], GlobalRecord]:
    config = cast(PoissonPreprocessingConfig, global_config.preprocessing_config)

    total_duration = config.random_duration + config.controlled_duration + config.silent_duration
    records = []
    global_record = GlobalRecord(None, total_duration, config.sample_count, None, None)

    for sample_id in range(config.sample_count):
        pair_id = sample_id // 2
        is_second_in_pair = sample_id % 2 == 1
        
        # 1. Seeds
        rng_random = np.random.default_rng(sample_id)
        rng_controlled = np.random.default_rng(5000 + pair_id) 
        
        all_times = []
        all_indices = []

        for i in range(config.channels):
            adj_rate = config.rate / (1.0 - config.rate * config.min_isi)
            
            n_rand = int(adj_rate * config.random_duration * 1.5 + 10)
            t_rand = np.cumsum(config.min_isi + rng_random.exponential(1.0/adj_rate, n_rand))
            t_rand = t_rand[t_rand < config.random_duration]

            # --- Suffix (The Controlled Frozen Noise) ---
            # Generate a Master Suffix for the pair
            n_cont = int(adj_rate * config.controlled_duration * 1.5 + 10)
            t_master = np.cumsum(config.min_isi + rng_controlled.exponential(1.0/adj_rate, n_cont))
            t_master = t_master[t_master < config.controlled_duration]

            if not is_second_in_pair:
                t_cont = t_master
            else:
                rng_mod = np.random.default_rng(sample_id + 999)
                keep_mask = rng_mod.random(len(t_master)) < config.similarity
                t_kept = t_master[keep_mask]
                
                n_new = len(t_master) - len(t_kept)
                t_new = rng_mod.uniform(0, config.controlled_duration, n_new)
                
                t_cont = np.sort(np.concatenate([t_kept, t_new]))
            
            t_cont += config.random_duration
            times = np.concatenate([t_rand, t_cont])
            times = enforce_min_isi_by_shifting(times, config.min_isi)

            all_times.append(times)
            all_indices.append(np.full(len(all_times[-1]), i))

        # Flatten, Sort, and Cache
        times_flat = np.concatenate(all_times)
        indices_flat = np.concatenate(all_indices)
        sort_idx = np.argsort(times_flat)
        encoder_spikes = (times_flat[sort_idx], indices_flat[sort_idx])

        sample_metadata = SampleMetadata(sample_id, Split.TRAIN, {"pair_id": pair_id, "similarity": config.similarity})

        # Label and Store
        n_label_random = int(config.random_duration / global_config.sim_config.analysis_dt)
        n_label_controlled = int(config.controlled_duration / global_config.sim_config.analysis_dt)
        n_label_silent = int(config.silent_duration / global_config.sim_config.analysis_dt)
        n_label_total = n_label_random + n_label_controlled+ n_label_silent

        label_values = np.zeros(n_label_total, dtype=np.float32)
    
        label_values[0 : n_label_random] = 1

        controlled_label = 3 if is_second_in_pair else 2
        label_values[n_label_random:(n_label_random + n_label_controlled)] = controlled_label
        

        validity_mask = np.ones(n_label_total, dtype=np.float32)
        label = np.vstack([validity_mask, label_values]).T

        record = Record(
            sample_metadata=sample_metadata,
            encoder_spikes=(encoder_spikes, config.channels),
            label=label
            )
        records.append(record)

    return records, global_record

def enforce_min_isi_by_shifting(times, min_isi):
    if len(times) < 2: return times
    
    times_fixed = np.copy(times)
    for i in range(1, len(times_fixed)):
        # If this spike is too close to the one before it
        if times_fixed[i] < times_fixed[i-1] + min_isi:
            times_fixed[i] = times_fixed[i-1] + min_isi
            
    return times_fixed