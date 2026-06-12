from datetime import datetime
import glob
import os
from pprint import pprint
import numpy as np
from analysis import get_spectral_radius
from interfaces import Fixed, Gaussian, MaassLIF, MarkramSyn, Neuron, RangeInfo, Record, ReservoirConfig, SampleMetadata, SimpleLIF, SimpleSyn, Synapse, Uniform
import brian2 as b2

from plotting import plot_reservoir
from utils import get_neurongroup, get_synapsegroup

def build_3D_reservoir_layer(
    config: ReservoirConfig,
    encoding_neurons: b2.Group,
    num_par: int,
    noise_seed: int  
    ) -> tuple[b2.NeuronGroup, b2.Synapses, b2.Synapses]:
    local_rng = np.random.RandomState(12)
    noise_rng = np.random.RandomState(noise_seed)

    N_in = encoding_neurons.N // num_par
    N_res = config.N

    root = round(N_res ** (1/3))
    
    if root**3 != N_res:
        raise ValueError(
            f"Invalid r size: N={N_res} is not a perfect cube. "
            f"To use a 3D grid, N must be a value like {root**3} or {(root+1)**3}."
        )

    # Template generation
    # XYZ
    indices = np.arange(N_res)
    x_template = (indices % root)
    y_template = ((indices // root) % root)
    z_template = (indices // (root**2))
    center = (root-1) / 2

    dist_sq = (x_template-center)**2 + (y_template- center)**2 + (z_template - center)**2
    sorted_order = np.argsort(-dist_sq)

    x_template = x_template[sorted_order]
    y_template = y_template[sorted_order]
    z_template = z_template[sorted_order]

    # Inhibitory
    num_inhib = int(config.N * config.factor_inh)
    template_inhib_indices = local_rng.choice(
        np.arange(config.N), 
        size=num_inhib, 
        replace=False
    )
    res_signs_template = np.ones(N_res)
    res_signs_template[template_inhib_indices] = -1

    # Input-reservoir connections
    template_i = []
    template_j = []

    for i in range(N_in):
        targets = local_rng.choice(N_res, size=config.F_in, replace=False)
        
        template_i.extend([i] * config.F_in)
        template_j.extend(targets)

    ij_in_template = np.column_stack((template_i, template_j))

    # Reservoir connections
    dx = x_template[:, np.newaxis] - x_template[np.newaxis, :]
    dy = y_template[:, np.newaxis] - y_template[np.newaxis, :]
    dz = z_template[:, np.newaxis] - z_template[np.newaxis, :]
    dist_sq = dx**2 + dy**2 + dz**2
    p_dist = np.exp(-(dist_sq / (config.lamda**2)))

    res_sign_pre = res_signs_template[:, np.newaxis]
    res_sign_post = res_signs_template[np.newaxis, :]
    p_type = np.zeros((N_res, N_res))
    p_type[(res_sign_pre ==  1) & (res_sign_post ==  1)] = config.C_EE
    p_type[(res_sign_pre ==  1) & (res_sign_post == -1)] = config.C_EI
    p_type[(res_sign_pre == -1) & (res_sign_post ==  1)] = config.C_IE
    p_type[(res_sign_pre == -1) & (res_sign_post == -1)] = config.C_II

    full_p_matrix = p_dist * p_type 
    conn_mask = local_rng.rand(N_res, N_res) < full_p_matrix
    template_res_i, template_res_j = np.where(conn_mask)
    ij_res_template = np.column_stack((template_res_i, template_res_j))

    N_incoming = np.bincount(template_res_j, minlength=N_res)
    N_incoming[N_incoming == 0] = 1

    # Creation of network
    reservoir_group = get_neurongroup(
        config.N, 
        num_par, 
        res_signs_template, 
        config.neuron_res)

    input_reservoir_synapses = get_synapsegroup(
        encoding_neurons, 
        np.ones(N_in),
        reservoir_group, 
        res_signs_template,
        ij_in_template,
        num_par,
        config.synapse_in,
        local_rng,
        noise_rng)

    reservoir_synapses = get_synapsegroup(
        reservoir_group,
        res_signs_template,
        reservoir_group,
        res_signs_template,
        ij_res_template,
        num_par,
        config.synapse_res,
        local_rng,
        noise_rng)

    # print(f"Spectral Radius: {get_spectral_radius(template_res_i, template_res_j, W_vals, N_res)}")
    print(f"Synapses: {len(template_res_i)/(N_res*(N_res-1))*100.0}%")
    plot_reservoir(config, template_res_i, template_res_j, res_signs_template, template_i, template_j,N_in, 2)

    return reservoir_group, input_reservoir_synapses, reservoir_synapses

def store_cache(
        record: Record,
        cache_dir: str,
    ):

    reservoir_spikes = record.get('reservoir_spikes')
    label = record.get('label')
    sample_metadata = record.get('sample_metadata')

    cache_dir = os.path.join(cache_dir, f"reservoir")
    os.makedirs(cache_dir, exist_ok=True)

    base_name = f"Sample{sample_metadata.sample_id:02d}"
    pattern = os.path.join(cache_dir, f"{base_name}_*.npz")
    for existing_file in glob.glob(pattern):
        os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}"
    filepath = os.path.join(cache_dir, filename)

    (spike_times, spike_indices), channel_count = reservoir_spikes

    np.savez(
        filepath, 
        times=spike_times, 
        indices=spike_indices,
        channel_count = channel_count,
        label=label,
    )

def load_cache(
        records: list[Record], 
        cache_dir: str):

    cache_dir = os.path.join(cache_dir, f"reservoir")

    for record in records:
        sample_metadata = record.get('sample_metadata')
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
            reservoir_spikes = ((spike_times, spike_indices), channel_count)
            label = data['label'].copy()        

            record.set('reservoir_spikes', reservoir_spikes)
            record.set('label', label)