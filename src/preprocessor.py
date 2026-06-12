from datetime import datetime
import glob
import re
import os
from typing import cast
import mne
import numpy as np
import random as py_random
import brian2
import matplotlib
import matplotlib.pyplot as plt
import warnings
import chb
import bonn

from interfaces import *
import poisson
import poisson_maass
import whitenoise

brian2.prefs.core.default_float_dtype = brian2.float32

matplotlib.use('TkAgg')
os.environ['PYTHONWARNINGS'] = 'ignore:_get_vc_env is private:UserWarning'

CHB_DATA_PATH = '../data/chb-mit-scalp-eeg-database-1.0.0'
BONN_DATA_PATH = '../bonn-src/data'

def get_records(
        config: Config,
        noise_seed: int,
    ) -> tuple[List[Record], GlobalRecord]:

    match config.preprocessing_config:
        case CHBPreprocessingConfig():
            return chb.generate_chb_data(CHB_DATA_PATH, config)
        
        case BONNPreprocessingConfig():
            return bonn.generate_bonn_data(BONN_DATA_PATH, config)

        case PoissonPreprocessingConfig():
            return poisson.generate_poisson_metadata(config)

        case PoissonMaassPreprocessingConfig():
            return poisson_maass.generate_poisson_maass_metadata(config, noise_seed)

        case WhiteNoiseConfig():
            return whitenoise.generate_whitenoise_data(config)

def get_samples(
        records: List[Record], 
        global_record: GlobalRecord, 
        cache: bool,
        return_array: bool, 
        config: Config,
        working_dir: str | None = None):

    match config.preprocessing_config:
        case CHBPreprocessingConfig():
            chb_config = cast(CHBPreprocessingConfig, config.preprocessing_config)
            if chb_config.channels != 18:
                raise ValueError("Channel count other than 18 is not implemented")
            
            for record in records:
                sample = chb.process_range(record.get('sample_metadata'), global_record.get('metadata'), cache, return_array, config, working_dir)
                
                if sample is not None:
                    input_data, label = sample
                    record.set("input_data", input_data)
                    record.set("label", label)

        case BONNPreprocessingConfig():
            bonn_config = cast(BONNPreprocessingConfig, config.preprocessing_config)
            for record in records:
                sample = bonn.process_bonn_range(record.get('sample_metadata'), config)
                
                if sample is not None:
                    # Split the (Samples, Channels + 2) array back into data and labels
                    # Column -1 is labels, columns 0 to -2 are data + validity
                    input_data = sample[:, :-2] 
                    label = sample[:, -2:] # validity and phase labels
                    
                    record.set("input_data", input_data)
                    record.set("label", label)

        case PoissonPreprocessingConfig():
            raise ValueError("Poisson does not provide continuous input. Consider Encoder_cache as start_location")
        
        case WhiteNoiseConfig():
            raise ValueError("WhiteNoise does not need preprocessing. Consider Preprocessing_out as start_location")


def load_cache(
    records: List[Record], 
    cache_dir: str,
    config: Config
):
    match config.preprocessing_config:
        case CHBPreprocessingConfig():
            chb.load_cache(records, cache_dir, config)

        case _:
            raise ValueError("Load cache not implemented for this preprocessing config")
