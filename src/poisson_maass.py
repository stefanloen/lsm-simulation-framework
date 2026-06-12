import os
from pprint import pprint
import numpy as np
from typing import List, Tuple, cast
from interfaces import Config, GlobalRecord, PoissonMaassPreprocessingConfig, PoissonPreprocessingConfig, Record, SampleMetadata, Split

def generate_poisson_maass_metadata(
    global_config: Config,
    noise_seed: int,
) -> Tuple[List[Record], GlobalRecord]:
    config = cast(PoissonMaassPreprocessingConfig, global_config.preprocessing_config)
    
    # Constants based on requirements
    total_duration = config.num_seq_patterns * config.pattern_duration

    base_patterns = []
    pattern_rng = np.random.default_rng(noise_seed) # Fixed seed for base patterns
    
    for p_idx in range(config.n_patterns):
        channel_spikes = []

        # multiplier = 2.0 if p_idx == 1 else 1
        multiplier = 1.0

        for c in range(config.channels):
            # Poisson process: inter-spike intervals are exponential
            # Expected spikes = rate * duration
            n_expected = int(multiplier * config.rate * config.pattern_duration * 2.0 + 10)
            intervals = pattern_rng.exponential(1.0 / (config.rate*multiplier), n_expected)
            times = np.cumsum(intervals)
            times = times[times < config.pattern_duration]
            channel_spikes.append(times)
        base_patterns.append(channel_spikes)

    records = []
    global_record = GlobalRecord(None, total_duration, config.sample_count, None, None)

    match_count = 0
    total_lagged_pairs = 0
    lag_steps = 4  # 1.0s / 0.250s = 4 patterns ago

    # 2. Generate Samples
    for sample_id in range(config.sample_count):
        # Unique seed per sample for the sequence and jitter
        sample_rng = np.random.default_rng(noise_seed + sample_id)
        
        # Randomly choose patterns
        # sequence = sample_rng.integers(0, config.n_patterns, config.num_seq_patterns)
        
        # --- NEW BURST SEQUENCE LOGIC ---
        sequence_list = []
        while len(sequence_list) < config.num_seq_patterns:
            # Pick a random pattern
            pattern_idx = sample_rng.integers(0, config.n_patterns)
            # Pick a random burst length (e.g., between 1 and 5)
            burst_len = sample_rng.integers(config.n_repeat_min, config.n_repeat_max+1)
            
            # Add this pattern 'burst_len' times, but don't exceed max length
            remaining = config.num_seq_patterns - len(sequence_list)
            actual_burst = min(burst_len, remaining)
            
            sequence_list.extend([pattern_idx] * actual_burst)
        
        sequence = np.array(sequence_list)
        # --------------------------------

        # --- NEW: CHECK SEQUENCE BIAS ---
        for s_idx in range(lag_steps, len(sequence)):
            current_p = sequence[s_idx]
            lagged_p = sequence[s_idx - lag_steps]
            
            if current_p == lagged_p:
                match_count += 1
            total_lagged_pairs += 1
        # --------------------------------

        all_times = []
        all_indices = []

        # Calculate steps for labeling and the 1-second lag
        dt = global_config.sim_config.analysis_dt
        n_total_steps = int(total_duration / dt)
        n_steps_per_pattern = int(config.pattern_duration / dt)
        n_lag_steps = int(config.lag/ dt) # 1 second worth of steps

        label_values = np.full(n_total_steps, -1, dtype=np.float32)
        validity_mask = np.zeros(n_total_steps, dtype=np.float32)

        for seq_idx, pattern_idx in enumerate(sequence):
            offset = seq_idx * config.pattern_duration
            
            # --- LAG LOGIC ---
            # Original window: [seq_idx * 0.5, (seq_idx + 1) * 0.5]
            # Lagged window:   [seq_idx * 0.5 + 1.0, (seq_idx + 1) * 0.5 + 1.0]
            start_step = (seq_idx * n_steps_per_pattern) + n_lag_steps
            end_step = start_step + n_steps_per_pattern
            
            # Only apply labels if they fall within the simulation duration
            if start_step < n_total_steps:
                actual_end = min(end_step, n_total_steps)
                label_values[start_step:actual_end] = pattern_idx
                # The mask ensures we only calculate error on valid lagged segments
                validity_mask[start_step:actual_end] = 1.0
            # -----------------
            
            # Add spikes from the chosen base pattern
            pattern_data = base_patterns[pattern_idx]
            for ch_idx, spikes in enumerate(pattern_data):
                if len(spikes) == 0:
                    continue
                
                # Apply Jitter: Mean 0, Variance 32ms
                jitter = sample_rng.normal(0, config.jitter_std, len(spikes))
                jittered_spikes = spikes + jitter + offset
                
                # Keep spikes within the bounds of this specific slot to avoid massive drift
                # or allow them to bleed slightly? Usually, we clip to [0, TOTAL_DURATION]
                jittered_spikes = jittered_spikes[
                    (jittered_spikes >= 0) & (jittered_spikes < total_duration)
                ]
                
                all_times.append(jittered_spikes)
                all_indices.append(np.full(len(jittered_spikes), ch_idx))

# 3. Flatten, Sort, and Enforce ISI
        if len(all_times) > 0:
            times_flat = np.concatenate(all_times)
            indices_flat = np.concatenate(all_indices)
            
            # A. Global Deletion
            if config.deletion_p > 0:
                keep_mask = sample_rng.random(len(times_flat)) > config.deletion_p
                times_flat = times_flat[keep_mask]
                indices_flat = indices_flat[keep_mask]

            # B. Global Injection
            if config.injection_rate > 0:
                times_flat, indices_flat = inject_noise_spikes(
                    times_flat, indices_flat, total_duration, config.channels, config.injection_rate, sample_rng
                )

            # Global sort by time
            sort_idx = np.argsort(times_flat)
            t_sorted = times_flat[sort_idx]
            i_sorted = indices_flat[sort_idx]
            
            # Create a master mask to keep track of which spikes survive
            keep_mask = np.ones(len(t_sorted), dtype=bool)
            
            for ch in range(config.channels):
                # Find where this channel's spikes are in the sorted array
                ch_indices = np.where(i_sorted == ch)[0]
                if len(ch_indices) >= 2:
                    ch_times = t_sorted[ch_indices]
                    
                    # Identify which spikes in this channel violate ISI
                    # We only keep spikes that are far enough from the PREVIOUS kept spike
                    ch_keep = np.ones(len(ch_times), dtype=bool)
                    last_kept_t = ch_times[0]
                    for i in range(1, len(ch_times)):
                        if ch_times[i] < last_kept_t + config.min_isi:
                            ch_keep[i] = False
                        else:
                            last_kept_t = ch_times[i]
                    
                    # Update the master mask
                    keep_mask[ch_indices] = ch_keep
            
            # Apply the mask to both times and indices simultaneously
            t_final = t_sorted[keep_mask]
            i_final = i_sorted[keep_mask]
            
            encoder_spikes = (t_final, i_final)
        sample_metadata = SampleMetadata(sample_id, Split.TRAIN, {"sequence": sequence.tolist()})

        validity_mask = np.ones(len(label_values), dtype=np.float32)
        label = np.vstack([validity_mask, label_values]).T

        record = Record(
            sample_metadata=sample_metadata,
            encoder_spikes=(encoder_spikes, config.channels),
            label=label
        )
        records.append(record)

    if config.chunk is not None:
        records = split_records_into_chunks(global_config, records, config.chunk)
        global_record.duration =  config.chunk
    return records, global_record

def enforce_min_isi_by_shifting(times, min_isi):
    if len(times) < 2: 
        return times
    
    times_fixed = np.copy(times)
    for i in range(1, len(times_fixed)):
        # If this spike is too close to the one before it, shift it forward
        if times_fixed[i] < times_fixed[i-1] + min_isi:
            times_fixed[i] = times_fixed[i-1] + min_isi
            
    return times_fixed

def apply_spike_deletion(times, indices, deletion_prob, rng):
    if len(times) == 0 or deletion_prob <= 0:
        return times, indices
    
    keep_mask = rng.random(len(times)) > deletion_prob
    
    return times[keep_mask], indices[keep_mask]



def inject_noise_spikes(times, indices, total_duration, n_channels, noise_rate, rng):
    if noise_rate <= 0:
        return times, indices
    
    # Calculate how many noise spikes to add across all channels
    # Expected N = Rate * Time * Num_Channels
    n_expected = int(noise_rate * total_duration * n_channels)
    
    # Generate random times and assign them to random channels
    noise_times = rng.uniform(0, total_duration, n_expected)
    noise_indices = rng.integers(0, n_channels, n_expected)
    
    # Concatenate with existing pattern spikes
    new_times = np.concatenate([times, noise_times])
    new_indices = np.concatenate([indices, noise_indices])
    
    return new_times, new_indices

def enforce_min_isi_by_deletion(times, min_isi):
    pprint(times.shape)
    if len(times) < 2: 
        return times
    
    keep = [True] * len(times)
    last_kept_time = times[0]
    
    for i in range(1, len(times)):
        if times[i] < last_kept_time + min_isi:
            keep[i] = False  # Drop the spike instead of pushing it forward
        else:
            last_kept_time = times[i]
            
    return times[keep]

def split_records_into_chunks(config: Config, records: List[Record], chunk_duration: float = 0.250) -> List[Record]:
    """
    Splits a list of long Records into multiple smaller Records of chunk_duration.
    """
    chunked_records = []
    
    for rec in records:
        (t_all, i_all), n_channels = rec.get("encoder_spikes")
        # Assuming dt is stored in your config or can be inferred from the label length
        # total_duration = n_steps * dt
        label_data = rec.get("label") # Shape [N, 2] -> [validity, value]
        n_total_steps = len(label_data)
        
        dt = config.sim_config.analysis_dt
        steps_per_chunk = int(chunk_duration / dt)
        
        num_chunks = n_total_steps // steps_per_chunk
        
        for c_idx in range(num_chunks):
            start_step = c_idx * steps_per_chunk
            end_step = start_step + steps_per_chunk
            
            start_time = start_step * dt
            end_time = end_step * dt
            
            # 1. Slice Labels
            chunk_label = label_data[start_step:end_step]
            
            # 2. Slice Spikes using a mask
            mask = (t_all >= start_time) & (t_all < end_time)
            chunk_t = t_all[mask] - start_time # Normalize time to start at 0.0
            chunk_i = i_all[mask]
            
            # 3. Create new Metadata for the chunk
            # We preserve the original sequence info but note it's a chunk
            chunk_metadata = SampleMetadata(
                sample_id=rec.get("sample_metadata").sample_id,
                split=rec.get("sample_metadata").split,
                metadata=None
            )
            
            # 4. Assemble the new Record
            new_record = Record(
                sample_metadata=chunk_metadata,
                encoder_spikes=((chunk_t, chunk_i), n_channels),
                label=chunk_label
            )
            
            chunked_records.append(new_record)
            
    return chunked_records