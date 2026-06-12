from pprint import pprint

import brian2 as b2
import numpy as np

from interfaces import Neuron, OutputConfig, RangeInfo, Record, SampleMetadata
from utils import get_neurongroup, get_synapsegroup

def build_LI_output_layer(
    config: OutputConfig,
    trained_params: dict,
    reservoir_group: b2.Group,
    num_par: int,
    noise_seed: int
) -> tuple[b2.NeuronGroup, b2.Synapses]:
    local_rng = np.random.RandomState(12)
    noise_rng = np.random.RandomState(noise_seed)
    weights = np.array(trained_params['fc.weight'], dtype=np.float64) * 1e-6 # Magic number
    bias = np.array(trained_params['fc.bias'], dtype=np.float64)

    bias = bias * 0.125 # TODO where does this magic number come from? Its 'randomly' set like this

    num_classes = weights.shape[0]

    N_res = reservoir_group.N // num_par

    i_indices, j_indices = np.meshgrid(np.arange(N_res), np.arange(num_classes), indexing='ij')

    # Flatten them into a (N_res * N_out, 2) array
    ij_out_template = np.column_stack((i_indices.flatten(), j_indices.flatten()))

    output_group = get_neurongroup(
        num_classes,
        num_par,
        np.ones(num_classes),
        Neuron(config.li_config, config.li_config),
    )

    synapses_group = get_synapsegroup(
        reservoir_group,
        np.ones(N_res),
        output_group,
        np.ones(num_classes),
        ij_out_template,
        num_par,
        config.synapse,
        local_rng,
        noise_rng
    )

    output_group.s = np.tile(bias, num_par)
    output_group.v_bias = np.tile(bias, num_par) * b2.volt
    synapses_group.w = np.tile(weights.T.flatten(), num_par)

    return output_group, synapses_group


import os
import glob
import numpy as np
from datetime import datetime

def store_cache(
    # v: np.ndarray,
    # label: np.ndarray, 
    # sample_metadata: SampleMetadata, 
    record: Record,
    cache_dir: str,
):
    
    v = record.get('output_v')
    label = record.get('label')
    sample_metadata = record.get('sample_metadata')
    
    cache_dir = os.path.join(cache_dir, f"output")
    os.makedirs(cache_dir, exist_ok=True)

    base_name = f"Sample{sample_metadata.sample_id:02d}"
    
    pattern = os.path.join(cache_dir, f"{base_name}_v_*.npz")
    for existing_file in glob.glob(pattern):
        os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_v_{timestamp}"
    filepath = os.path.join(cache_dir, filename)

    np.savez(
        filepath, 
        voltages=v,
        label=label,
    )

def load_cache(
        records: list[Record], 
        cache_dir: str):

    cache_dir = os.path.join(cache_dir, f"output")

    for record in records:
        sample_metadata = record.get('sample_metadata')
        base_name = f"Sample{sample_metadata.sample_id:02d}"
        pattern = os.path.join(cache_dir, f"{base_name}*_v_*.npz")
        matching_files = sorted(glob.glob(pattern)) 
        
        if not matching_files:
            raise FileNotFoundError(f"No voltage cache found for {base_name}")
        
        filepath = matching_files[-1]

        with np.load(filepath, allow_pickle=True) as data:
            v_values = data['voltages'].copy()
            label = data['label'].copy()    

            record.set('output_v', v_values)
            record.set('label', label)