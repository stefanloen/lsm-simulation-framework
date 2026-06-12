from typing import List, cast

import numpy as np
from interfaces import Config, GlobalRecord, Record, SampleMetadata, WhiteNoiseConfig
from scipy import signal
from scipy.stats import rankdata

def generate_whitenoise_data(
        config: Config
)-> tuple[List[Record], GlobalRecord]:
    white_noise_config = cast(WhiteNoiseConfig, config.preprocessing_config)
    warmup_samples = int(white_noise_config.duration / config.sim_config.analysis_dt *0.1)
    num_samples = int(white_noise_config.duration / config.sim_config.analysis_dt)

    sos = signal.butter(
        N=4, 
        Wn=white_noise_config.cutoff_freq, 
        btype= 'low', 
        fs=1.0/config.sim_config.analysis_dt, 
        output='sos')

    records = []

    for sample_id in range(white_noise_config.sample_count):
        rng = np.random.default_rng(sample_id)
        raw_noise = rng.uniform(0, 1, num_samples + warmup_samples)

        raw_data = signal.sosfilt(sos, raw_noise)
        raw_data = raw_data[warmup_samples:]

        ranks = rankdata(raw_data)
        uniform_data = (ranks - 1) / (len(ranks) - 1)
        uniform_data = uniform_data[:, np.newaxis]

        sample_metadata = SampleMetadata(sample_id, None)

        # raw_data = np.reshape(raw_data, (-1, 1))

        validity_mask = np.ones((num_samples, 1), dtype=np.float32)
        label = np.hstack([validity_mask, uniform_data])   

        record = Record(input_data=uniform_data, label=label, sample_metadata=sample_metadata)
        records.append(record)

    global_record = GlobalRecord(
        white_noise_config.cutoff_freq, 
        white_noise_config.duration, 
        len(config.train_config.delays), 
        None, 
        None)

    return records, global_record
    