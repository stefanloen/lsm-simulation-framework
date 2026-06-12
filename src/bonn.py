import os
import glob
import numpy as np
from typing import List, Tuple, cast
from mne.filter import filter_data
from scipy.signal import resample


from interfaces import BONNPreprocessingConfig, Config, EEGPhase, GlobalRecord, RangeInfo, Record, SampleMetadata, Split

ORIGINAL_FS = 173.61 
ORIGINAL_SAMPLES = 4097

# --- 1. GENERATE RECORDS ---
def generate_bonn_data(
        data_dir: str, 
        config: Config
    ) -> Tuple[List[Record], GlobalRecord]:
    
    bonn_config = cast(BONNPreprocessingConfig, config.preprocessing_config)
    analysis_dt = config.sim_config.analysis_dt

    records: List[Record] = []
    sample_id = 0

    # Path logic: data_dir/O and data_dir/S
    target_sets = [('O', EEGPhase.INTERICTAL), ('S', EEGPhase.ICTAL)]

    for subfolder, phase in target_sets:
        folder_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(folder_path):
            continue

        files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
        for f in files:
            # We encode the full path into patient_id to bypass RangeInfo limitations
            range_info: RangeInfo = {
                'patient_id': f,  # Store full path here
                'range_id': sample_id,
                'eegphase': phase,
                'range': (0, 4097) 
            }
            
            record = Record(sample_metadata=SampleMetadata(sample_id, Split.TRAIN, range_info))
            records.append(record)
            sample_id += 1

    global_record = GlobalRecord(
        cutoff_freq=0, 
        duration= ORIGINAL_SAMPLES / ORIGINAL_FS, 
        num_classes=4, 
        trained_params=None,
        metadata=None
    )

    return records, global_record

# --- 2. PROCESS RECORD ---
def process_bonn_range(
        sample_metadata: SampleMetadata, 
        config: Config
    ) -> np.ndarray | None:
    
    bonn_config = cast(BONNPreprocessingConfig, config.preprocessing_config)
    analysis_dt = config.sim_config.analysis_dt
    
    range_info = sample_metadata.metadata
    file_path = range_info['patient_id']

    try:
        raw_values = np.loadtxt(file_path).astype(np.float64)
        
        data_to_filter = raw_values.reshape(1, -1)
        filtered_data = filter_data(
            data_to_filter, 
            sfreq=ORIGINAL_FS, 
            l_freq=None, 
            h_freq=bonn_config.cutoff_freq, 
            verbose=False
        )

        duration = ORIGINAL_SAMPLES / ORIGINAL_FS
        n_target_samples = int(duration / analysis_dt)
        
        resampled_data = resample(filtered_data, n_target_samples, axis=1).astype(np.float32)

        data = resampled_data.reshape(1, -1)

        if bonn_config.fixed_normalization:
            median = bonn_config.fixed_median
            scale = bonn_config.fixed_scale
        else:
            q1 = np.percentile(data, 25, axis=1, keepdims=True)
            q3 = np.percentile(data, 75, axis=1, keepdims=True)
            median = np.median(data, axis=1, keepdims=True)
            scale = 1.0 / (q3 - q1 + 1e-9)

        data = (data - median) * scale
        data = np.clip(data, -1, 1)

        n_samples = data.shape[1]
        validity_mask = np.ones((1, n_samples), dtype=np.float32)
        labels = np.ones(n_samples, dtype=np.float32) * range_info['eegphase'].value

        return np.vstack([data, validity_mask, labels.reshape(1, -1)]).T

    except Exception as e:
        print(f"Error in Bonn file {file_path}: {e}")
        return None