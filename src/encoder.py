from datetime import datetime
import glob
import os
import brian2 as b2
import numpy as np

from interfaces import *

def store_cache(
        record: Record,
        cache_dir: str,
    ):

    encoder_spikes = record.get('encoder_spikes')
    label = record.get('label')
    sample_metadata = record.get('sample_metadata')

    cache_dir = os.path.join(cache_dir, f"encoder")
    os.makedirs(cache_dir, exist_ok=True)

    # range_info = record.get('sample_metadata').metadata
    # label_type = range_info['eegphase'].name
    base_name = f"Sample{sample_metadata.sample_id:02d}"
    pattern = os.path.join(cache_dir, f"{base_name}_*.npz")
    for existing_file in glob.glob(pattern):
        os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.npz"
    filepath = os.path.join(cache_dir, filename)

    (spike_times, spike_indices), channel_count = encoder_spikes

    np.savez(
        filepath, 
        times=spike_times, 
        indices=spike_indices,
        channel_count = channel_count,
        label=label,
    )

def load_cache(
        records: list[Record],
        cache_dir
               ):

    cache_dir = os.path.join(cache_dir, f"encoder")

    for record in records:
        sample_metadata = record.get("sample_metadata")
        base_name = f"Sample{sample_metadata.sample_id:02d}"
        pattern = os.path.join(cache_dir, f"{base_name}*.npz")
        matching_files = sorted(glob.glob(pattern)) 
        if not matching_files:
            raise FileNotFoundError(f"No cache file found for {base_name} in {cache_dir}")
        filepath = matching_files[-1]

        with np.load(filepath, allow_pickle=True) as data:
            spike_times = data['times'].copy()
            spike_indices = data['indices'].copy()
            channel_count = data['channel_count'].copy()
            encoder_spikes = ((spike_times, spike_indices), channel_count)
            label = data['label'].copy()            

            record.set('encoder_spikes', encoder_spikes)
            record.set('label', label)


def build_encoding_poisson_layer(
        timedarray: b2.TimedArray, 
        num_par: int,
        ) -> b2.NeuronGroup:
    
    num_channels = timedarray.values.shape[1] // num_par
    poisson_group = b2.PoissonGroup(num_channels * num_par, f'abs(input_array(t, i))*100 * Hz', namespace={'input_array': timedarray})

    return poisson_group # type: ignore


def build_encoding_threshold_layer(
        timedarray: b2.TimedArray, 
        num_par: int,
        threshold: float = 0,
        firing_rate: float = 80.0
        ) -> b2.NeuronGroup:
    
    num_channels = timedarray.values.shape[1] // num_par

    formula = f'int(abs(input_signal(t, i)) > threshold) * {firing_rate} * Hz'

    poisson_group = b2.PoissonGroup(
        num_channels * num_par, 
        formula,
        namespace={
            'input_signal': timedarray,
            'threshold': threshold
        }
    )

    return poisson_group # type: ignore

def build_noise_layer(
        N: int,
        num_par: int,
        dur: float,
        firing_rate: float,
    ):

    formula = f'int(t < dur * second) * firing_rate * Hz'

    poisson_group = b2.PoissonGroup(
        N * num_par, 
        formula,
        namespace={
            'firing_rate': firing_rate,
            'dur': dur
        }
    )

    return poisson_group # type: ignore

def build_encoding_binning_layer(
    timedarray: b2.TimedArray, 
    num_bins: int,
    v_min: float,
    v_max: float,
    firing_rate: float
) -> b2.NeuronGroup:
    
    total_channels = timedarray.values.shape[1] 
    num_neurons = total_channels * num_bins
    bin_width = (v_max - v_min) / num_bins
    isi = 1.0 / firing_rate
    
    model = '''
        channel_idx = i // num_bins : integer 
        bin_idx = i % num_bins : integer

        v_low = v_min + bin_idx * bin_width : 1 
        v_high = v_low + bin_width : 1 

        current_val = input_signal(t, channel_idx) : 1

        is_in_range = (current_val >= v_low) and (current_val < v_high) : boolean
        is_underflow = (current_val < v_min) and (bin_idx == 0) : boolean
        is_overflow = (current_val >= v_max) and (bin_idx == num_bins - 1) : boolean

        in_bin = is_in_range or is_underflow or is_overflow : boolean
        is_clock_tick = (t % isi) < (dt / 2) : boolean
    '''
    
    threshold = 'in_bin and is_clock_tick'
    reset = ''

    encoding_group = b2.NeuronGroup(
        num_neurons, 
        model=model,
        threshold=threshold,
        reset=reset,
        method='exact',
        namespace={
            'input_signal': timedarray,
            'v_min': v_min,
            'v_max': v_max,
            'bin_width': bin_width,
            'num_bins': num_bins,
            'isi': isi * b2.second
        }
    )

    # Initialize last_spike to avoid a massive burst at t=0
    # encoding_group.last_spike = -isi * b2.second

    return encoding_group

def build_encoding_logbinning_layer(
    timedarray: b2.TimedArray, 
    num_bins: int,
    v_min: float,
    v_max: float,
    firing_rate: float,
    k: float = 2.0
) -> b2.NeuronGroup:
    
    if num_bins % 2 != 0:
        raise ValueError("Bin count should be even")

    total_channels = timedarray.values.shape[1] 
    num_neurons = total_channels * num_bins
    isi = 1.0 / firing_rate

    num = num_bins // 2
    indices = np.linspace(0, num, num + 1)
    edges = v_max * (indices / num)**k
    bin_edges = np.concatenate([np.sort(v_min * (indices / num)**k), edges[1:]])

    v_low_vals = np.tile(bin_edges[:-1], total_channels)
    v_high_vals = np.tile(bin_edges[1:], total_channels)

    model = '''
        channel_idx = i // num_bins : integer 
        bin_idx = i % num_bins : integer

        v_low : 1 
        v_high : 1 

        current_val = input_signal(t, channel_idx) : 1

        is_in_range = (current_val >= v_low) and (current_val < v_high) : boolean
        is_underflow = (current_val < v_min) and (bin_idx == 0) : boolean
        is_overflow = (current_val >= v_max) and (bin_idx == num_bins - 1) : boolean

        in_bin = is_in_range or is_underflow or is_overflow : boolean
        is_clock_tick = (t % isi) < (dt / 2) : boolean
    '''
    
    threshold = 'in_bin and is_clock_tick'
    reset = ''

    encoding_group = b2.NeuronGroup(
        num_neurons, 
        model=model,
        threshold=threshold,
        reset=reset,
        method='exact',
        namespace={
            'input_signal': timedarray,
            'v_min': v_min,
            'v_max': v_max,
            'num_bins': num_bins,
            'isi': isi * b2.second
        }
    )

    encoding_group.v_low = v_low_vals
    encoding_group.v_high = v_high_vals

    # Initialize last_spike to avoid a massive burst at t=0
    # encoding_group.last_spike = -isi * b2.second

    return encoding_group

import brian2 as b2

def build_gaussian_binning_layer(
    timedarray: b2.TimedArray, 
    num_bins: int,
    v_min: float,
    v_max: float,
    max_firing_rate: float,
    sigma: float
) -> b2.NeuronGroup:
    
    total_channels = timedarray.values.shape[1] 
    num_neurons = total_channels * num_bins
    bin_width = (v_max - v_min) / num_bins
    
    normalization = 1.0 / (sigma * np.sqrt(2 * np.pi))

    bin_width = (v_max - v_min) / num_bins
    
    if sigma < bin_width * 0.5:
        raise ValueError('Sigma is too small, switch to hard binning')

    model = '''
        channel_idx = i // num_bins : integer
        bin_idx = i % num_bins : integer

        v_center = v_min + (bin_idx + 0.5) * bin_width : 1

        current_val = input_signal(t, channel_idx) : 1

        dist_sq = (current_val - v_center)**2 : 1
        raw_activation = exp(-dist_sq / (2 * sigma_sq)) : 1
        
        rate = max_rate * raw_activation * norm_factor * bin_width : Hz
    '''
    
    threshold = 'rand() < rate * dt'
    reset = ''

    encoding_group = b2.NeuronGroup(
        num_neurons, 
        model=model,
        threshold=threshold,
        reset=reset,
        method='euler', # Required for the stochastic rand() check
        namespace={
            'input_signal': timedarray,
            'v_min': v_min,
            'max_rate': max_firing_rate * b2.Hz,
            'sigma_sq': sigma**2,
            'norm_factor': normalization,
            'bin_width': bin_width,
            'num_bins': num_bins
        }
    )

    return encoding_group


def build_delta_encoding_layer(
    timedarray: b2.TimedArray, 
    delta: float = 0.05,
) -> b2.NeuronGroup:
    total_channels = timedarray.values.shape[1] 
    model = '''
        channel_idx = i: integer
        current_val = input_signal(t, channel_idx) : 1
        last_val : 1
    '''
    threshold = 'abs(current_val-last_val) > delta'
    reset = 'last_val = current_val'

    hybrid_group = b2.NeuronGroup(
        total_channels,
        model=model,
        threshold=threshold,
        reset=reset,
        method='euler',
        namespace={
            'input_signal': timedarray,
            'delta': delta
        }
    )

    return hybrid_group

def build_hybrid_encoding_layer(
    timedarray: b2.TimedArray, 
    delta: float = 0.1,
    base_rate: float = 20.0
) -> b2.NeuronGroup:
    total_channels = timedarray.values.shape[1] 
    model = '''
        channel_idx = i // 2 : integer
        is_delta_sigma = i % 2 : boolean

        # Rate based
        current_val = input_signal(t, channel_idx) : 1
        rate = abs(current_val) * base_rate*10 * Hz : Hz

        # Delta
        last_val : 1
    '''
    threshold = '(is_delta_sigma and abs(current_val-last_val) > delta ) or (not is_delta_sigma) and (rand() < rate * dt)'
    reset = 'last_val = current_val'

    hybrid_group = b2.NeuronGroup(
        total_channels * 2,
        model=model,
        threshold=threshold,
        reset=reset,
        method='euler',
        namespace={
            'total_channels' : total_channels,
            'input_signal': timedarray,
            'base_rate': base_rate,
            'delta': delta
        }
    )

    return hybrid_group