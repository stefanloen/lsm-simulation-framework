from datetime import datetime
import glob
from pprint import pprint
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

from interfaces import *

brian2.prefs.core.default_float_dtype = brian2.float32

matplotlib.use('TkAgg')
os.environ['PYTHONWARNINGS'] = 'ignore:_get_vc_env is private:UserWarning'

def generate_chb_data(
        data_dir: str, 
        config: Config,
    ) -> Tuple[List[Record], GlobalRecord]:

    chb_config = cast(CHBPreprocessingConfig, config.preprocessing_config)

    py_random.seed(12)
    summaries: Summaries = {}
    records: List[Record] = []
    patient_ids: List[int] = list(range(1, 24))
    sample_id = 0
    for patient_id in patient_ids:
        patient_name = f"chb{patient_id:02d}"
        patient_summary = get_summary(data_dir, patient_id)
        summaries[patient_name] = patient_summary
        range_infos = get_patient_ranges(patient_summary, chb_config)
        for range_info in range_infos:
            record = Record(sample_metadata=SampleMetadata(sample_id, Split.TRAIN, range_info))
            records.append(record)
            sample_id += 1
        
    global_record = GlobalRecord(
        cutoff_freq=chb_config.cutoff_freq, #type
        duration=chb_config.window_size, 
        num_classes=4, 
        trained_params=None,
        metadata=summaries)

    return records, global_record

def process_range( 
        sample_metadata: SampleMetadata, 
        summaries: Dict[str, PatientSummary], 
        cache: bool,
        return_array: bool, 
        config: Config,
        working_dir: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    
    warnings.filterwarnings("ignore", message="Channel names are not unique")
    warnings.filterwarnings("ignore", message="Scaling factor is not defined")

    cache_dir = None
    if working_dir is not None:
        os.makedirs(working_dir, exist_ok=True)

        cache_dir = os.path.join(working_dir, f"preprocessor")

    if cache and not working_dir:
        raise ValueError(f"output_dir must be specified when cache=True")

    range_info = sample_metadata.metadata

    try:
        summary = summaries[range_info['patient_id']]
        array = load_data(range_info, summary, config)
        if cache and cache_dir:
            store_data(array, range_info, cache_dir, True)

        if return_array:   
            data = array[:, :-2].copy()
            labels = array[:, -2:]
            return data, labels
        else:
            return None
    except Exception as e:
        print(f"Error in range {range_info['patient_id']}: {e}")
        return None

def to_seconds(
        t_str: str
        ) -> int:
    h, m, s = map(int, t_str.split(':'))
    return h * 3600 + m * 60 + s

def store_data(
        array: np.ndarray, 
        range_info: RangeInfo, 
        output_dir: str,
        overwrite: bool
        ):
    os.makedirs(output_dir, exist_ok=True)

    label_type = range_info['eegphase'].name
    base_name = f"{range_info['patient_id']}_{range_info['range_id']:02d}_{label_type}"

    if overwrite:
        pattern = os.path.join(output_dir, f"{base_name}_*.bin")
        for existing_file in glob.glob(pattern):
            os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.bin"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'ab') as f:
        array.tofile(f)

def load_cache(
        records: List[Record], 
        cache_dir: str,
        config: Config,
    ):

    chb_config = cast(CHBPreprocessingConfig, config.preprocessing_config)

    cache_dir = os.path.join(cache_dir, f"preprocessor")

    n_channels = len(COMMON_CHANNELS) + 2 
    expected_samples = int(chb_config.window_size / config.sim_config.analysis_dt)
    expected_total_elements = expected_samples * n_channels

    for record in records:
        range_info = record.get('sample_metadata').metadata
        label_type = range_info['eegphase'].name
        base_name = f"{range_info['patient_id']}_{range_info['range_id']:02d}_{label_type}"
        pattern = os.path.join(cache_dir, f"{base_name}*.bin")
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"No timestamped cache files found for: {base_name}")
        matching_files.sort()
        filepath = matching_files[-1] 
        flat_array = np.fromfile(filepath, dtype=np.float32)

        if flat_array.size != expected_total_elements:
            raise ValueError(
                f"Data integrity error in {base_name}: "
                f"Expected {expected_total_elements} elements ({expected_samples}x{n_channels}), "
                f"but found {flat_array.size}. File may be truncated or corrupted."
            )

        full_array = flat_array.reshape(expected_samples, n_channels)
        data = full_array[:, :-2].copy()
        label = full_array[:, -2:]

        record.set('input_data', data)
        record.set('label', label)

def get_summary(
        dir: str, 
        patient_id: int
        ) -> PatientSummary:
    # Get summary file contents
    patient_name = f"chb{patient_id:02d}"
    summary_filename = f"{patient_name}-summary.txt"
    
    patient_dir = os.path.join(dir, patient_name)
    summary_path = os.path.join(patient_dir, summary_filename)
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Could not find summary file at {summary_path}")

    with open(summary_path, 'r') as f:
        content = f.read()

    # Get Global Sampling Rate
    sr_match = re.search(r"Sampling Rate: (\d+) Hz", content)
    sample_rate = int(sr_match.group(1)) if sr_match else 256

    # Split into file-specific blocks
    file_blocks = content.split("File Name: ")
    patient_summary: PatientSummary = {}

    first_file_seconds = None
    last_file_seconds = 0
    day_offset = 0

    for block in file_blocks[1:]:
        lines = block.split('\n')
        file_name = lines[0].strip()
        
        # Parse Start/End Clock Times
        start_match = re.search(r"File Start Time: ([\d:]+)", block)
        end_match = re.search(r"File End Time: ([\d:]+)", block)
        
        if not start_match or not end_match:
            continue

        start_sec = to_seconds(start_match.group(1))
        end_sec = to_seconds(end_match.group(1))

        if first_file_seconds is None:
            first_file_seconds = start_sec
        elif start_sec < last_file_seconds:
            day_offset += 24 * 3600
        
        last_file_seconds = start_sec

        # Calculate continuous timeline relative to the very first file start
        global_start = (start_sec + day_offset) - first_file_seconds
        global_end = (end_sec + day_offset) - first_file_seconds
        file_duration = global_end - global_start

        # Parse Seizures
        seiz_count_match = re.search(r"Number of Seizures in File: (\d+)", block)
        seizure_count = int(seiz_count_match.group(1)) if seiz_count_match else 0
        
        seizures = []
        if seizure_count > 0:
            # Matches "Seizure Start Time", "Seizure 1 Start Time", etc.
            starts = re.findall(r"Seizure \d* ?Start Time: (\d+) seconds", block)
            ends = re.findall(r"Seizure \d* ?End Time: (\d+) seconds", block)
            for s_string, e_string in zip(starts, ends):
                seizure_start = int(s_string)
                seizure_end = int(e_string)
                seizure_global_start = global_start + seizure_start if seizure_start > 0 else np.nan
                seizure_global_end = global_start + seizure_end if seizure_end < file_duration else np.nan
                
                if np.isnan(seizure_global_start):
                    print("Seizure begins in previous file")

                if np.isnan(seizure_global_end):
                    print("Seizure ends in next file")

                seizures.append({
                    'global_start': seizure_global_start,
                    'global_end': seizure_global_end,
                    'local_start': seizure_start,
                    'local_end': seizure_end
                })

        real_seizure_count = sum([1 for s in seizures if np.isfinite(s['global_start'])])

        # 3. Build Entry
        patient_summary[file_name] = {
            "patient_dir": patient_dir,
            "sample_rate": sample_rate,
            "seizure_count": real_seizure_count,
            "seizures": seizures,
            "global_start": global_start,
            "global_end": global_end
        }

    return patient_summary

def get_onset_timepoints(
        summary: PatientSummary,
        config: CHBPreprocessingConfig
        ) -> List[int]:
    onset_timepoints = []
    
    for file_name in summary:
        file_data = summary[file_name]
        
        if file_data['seizure_count'] > 0:
            for seizure in file_data['seizures']:
                if np.isfinite(seizure['global_start']):
                    onset_timepoints.append(seizure['global_start'])

    return onset_timepoints

def get_ictal_timepoints(
        summary: PatientSummary,
        config: CHBPreprocessingConfig
        ) -> List[int]:
    ictal_timepoints = []
    for file_data in summary.values():
        if file_data['seizure_count'] > 0:
            for seizure in file_data['seizures']:
                if np.isfinite(seizure['global_start']) and np.isfinite(seizure['global_end']):
                    midpoint = (seizure['global_start'] + seizure['global_end']) // 2
                    ictal_timepoints.append(midpoint)
    return ictal_timepoints

def get_interictal_timepoints(
        summary: PatientSummary,
        config: CHBPreprocessingConfig
        ) -> List[int]:
    inter_points = []
    # Time constants
    WINDOW_BUFFER = 0.6 * config.window_size  # Max distance from window start to point (60% of 2h)
    FIVE_HOURS = 5*3600
    THREE_HOURS = 3*3600
    
    all_starts = [f['global_start'] for f in summary.values()]
    all_ends = [f['global_end'] for f in summary.values()]
    if not all_starts: return []
    
    curr = min(all_starts)
    stop = max(all_ends)

    while curr < stop:
        win_start = curr - WINDOW_BUFFER
        win_end = win_start + config.window_size
        
        # Calculate cumulative coverage: sum of all parts of the window that hit a file
        coverage = 0
        for f in summary.values():
            overlap_start = max(win_start, f['global_start'])
            overlap_end = min(win_end, f['global_end'])
            if overlap_start < overlap_end:
                coverage += (overlap_end - overlap_start)
        
        in_file = coverage >= (config.window_size - 30) # Allow 30 seconds missing

        # Safety Check: Is 'curr' 5h away from EVERY seizure in the summary?
        is_safe = True
        for file_data in summary.values():
            for seizure in file_data['seizures']:
                if np.isfinite(seizure['global_start']) and abs(curr - seizure['global_start']) < FIVE_HOURS:
                    is_safe = False
                    break
            if not is_safe: break

        if in_file and is_safe:
            inter_points.append(curr)
            curr += THREE_HOURS
        else:
            curr += 600 # Nudge 10 mins
            
    return inter_points

def get_ranges(
        timepoints: List[int],
        summary: PatientSummary,
        config: CHBPreprocessingConfig
        ) -> List[Tuple[int, int]]:
    ranges = []
    
    for timepoint in timepoints:
        valid_range_found = False
        for _ in range(10): 
            offset = py_random.randint(int(0.4 * config.window_size), int(0.6 * config.window_size))
            start = timepoint - offset
            end = start + config.window_size
            
            if is_window_covered(start, end, summary, config.window_size):
                ranges.append((start, end))
                valid_range_found = True
                break
        
        # if not valid_range_found:
        #     print(f"Warning: Could not find valid coverage at {timepoint}s. Skipping.")
            
    return ranges


def get_patient_ranges(
        summary: PatientSummary,
        config: CHBPreprocessingConfig
        ) -> List[RangeInfo]:
    first_file = list(summary.values())[0]
    patient_id = os.path.basename(first_file['patient_dir'])

    
    
    onset_timepoints = get_onset_timepoints(summary, config)
    preictal_timepoints = [int(t - config.preictal_duration/2) for t in onset_timepoints]
    ictal_timepoints = get_ictal_timepoints(summary, config)
    interictal_timepoints= get_interictal_timepoints(summary, config)



    onset_ranges = get_ranges(onset_timepoints, summary, config)
    preictal_ranges = get_ranges(preictal_timepoints, summary, config)
    ictal_ranges = get_ranges(ictal_timepoints, summary, config)
    interictal_ranges = get_ranges(interictal_timepoints, summary, config)
    
    all_ranges: List[RangeInfo] = []
    idx = 0

    if config.get_onset:
        for r in onset_ranges:
            all_ranges.append({
                'patient_id': patient_id, 
                'range_id': idx,
                'eegphase': EEGPhase.ONSET, 
                'range': r
            })
            idx +=1

    if config.get_preictal:
        for r in preictal_ranges:
            all_ranges.append({
                'patient_id': patient_id, 
                'range_id': idx,
                'eegphase': EEGPhase.PREICTAL, 
                'range': r
            })
            idx +=1
    
    if config.get_ictal:
        for r in ictal_ranges:
            all_ranges.append({
                'patient_id': patient_id, 
                'range_id': idx,
                'eegphase': EEGPhase.ICTAL, 
                'range': r
            })
            idx +=1
        
    if config.get_interictal:
        for r in interictal_ranges:
            all_ranges.append({
                'patient_id': patient_id, 
                'range_id': idx,
                'eegphase': EEGPhase.INTERICTAL, 
                'range': r
            })
            idx+=1
            
    return all_ranges

COMMON_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
    'FZ-CZ', 'CZ-PZ'
]

def load_data(
        range_info: RangeInfo, 
        summary: PatientSummary,
        config: Config) -> np.ndarray:
    
    chb_config = cast(CHBPreprocessingConfig, config.preprocessing_config)

    range_start, range_end = range_info['range']
    n_samples = int(chb_config.window_size / config.sim_config.analysis_dt)

    data = np.zeros((len(COMMON_CHANNELS), n_samples))
    validity_mask = np.zeros((1, n_samples))
    labels = np.ones(n_samples) * EEGPhase.INTERICTAL.value

    pre_ictal_starts = []
    ictal_starts = []
    ictal_ends = []
    post_ictal_ends = []

    overlap = []
    for file_name, file in summary.items():
        if file['global_end'] > range_start and file['global_start'] < range_end:
            overlap.append((file['global_start'], file_name, file))
    overlap.sort()

    # If overlap is empty everything goes wrong

    for file_start, file_name, file in overlap:
        dir = os.path.join(file['patient_dir'], file_name)
        raw = mne.io.read_raw_edf(dir, preload=True, verbose=False)
        
        for seizure in file['seizures']:            
            if np.isfinite(seizure['global_start']):
                pre_ictal_starts.append(max(0,seizure['global_start'] - chb_config.preictal_duration))
                ictal_starts.append(seizure['global_start'])

            if np.isfinite(seizure['global_end']):
                ictal_ends.append(seizure['global_end'])
                post_ictal_ends.append(seizure['global_end'] + chb_config.postictal_duration)

        # Find channels that match COMMON_CHANNELS
        picks_idx = []
        rename_map = {}
        
        raw_names_up = [ch.replace(' ', '').upper() for ch in raw.ch_names]
        
        for standard in COMMON_CHANNELS:
            found = False
            for i, raw_ch in enumerate(raw_names_up):
                if standard in raw_ch:
                    picks_idx.append(i)
                    rename_map[raw.ch_names[i]] = standard
                    found = True
                    break 
            if not found:
                print(f"Warning: {file_name} is missing channel {standard}")

        raw.pick(picks_idx)
        raw.rename_channels(rename_map)
        
        t_min = max(0, range_start - file_start)
        t_max = min(file['global_end'] - file_start, range_end - file_start)
        raw.crop(tmin=t_min, tmax=t_max, include_tmax=False)



        # Filtering of data
        if chb_config.rereference:
            raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

        if chb_config.notch_freqs is not None:
            raw.notch_filter(freqs=chb_config.notch_freqs, verbose=False)

        raw.filter(l_freq=0.5, h_freq=chb_config.cutoff_freq, verbose=False)
        raw.resample(sfreq=1.0/config.sim_config.analysis_dt, verbose=False)

        segment = raw.get_data()
        n_seg = segment.shape[1] # type: ignore

        insert_start = int(max(0, file_start - range_start) / config.sim_config.analysis_dt)
        insert_end = min(n_samples, insert_start + n_seg)
        
        segment_to_insert = segment[:, :insert_end - insert_start] # type: ignore
        data[:, insert_start:insert_end] = segment_to_insert
        validity_mask[0, insert_start:insert_end] = 1

        raw.close()

    if chb_config.fixed_normalization:
        median = chb_config.fixed_median
        scale = chb_config.fixed_scale
    else:
        valid_indices = np.where(validity_mask[0] == 1)[0]
        valid_data = data[:, valid_indices]

        q1 = np.percentile(valid_data, 25, axis=1, keepdims=True)
        q3 = np.percentile(valid_data, 75, axis=1, keepdims=True)
        median = np.median(valid_data, axis=1, keepdims=True)
        iqr = q3 - q1 + 1e-9
        scale = 1/iqr

    data = (data - median) * scale
    data = np.clip(data, -1, 1)

    pre_ictal_starts.sort()
    ictal_starts.sort()
    ictal_ends.sort()
    post_ictal_ends.sort()

    try:
        intervals = list(zip(pre_ictal_starts, ictal_starts, ictal_ends, post_ictal_ends, strict=True))
    except ValueError:
        raise ValueError(
            f"Label mismatch for range {range_start}-{range_end}. "
            f"Starts: {len(ictal_starts)}, Ends: {len(ictal_ends)}. "
            "A seizure likely crosses the boundary of the 2-hour window."
        )

    for pre_start, i_start, i_end, post_end in intervals:
            idx_pre   = int((pre_start - range_start) / config.sim_config.analysis_dt)
            idx_start = int((i_start - range_start) / config.sim_config.analysis_dt)
            idx_end   = int((i_end - range_start) / config.sim_config.analysis_dt)
            idx_post  = int((post_end - range_start) / config.sim_config.analysis_dt)

            idx_pre   = max(0, min(n_samples, idx_pre))
            idx_start = max(0, min(n_samples, idx_start))
            idx_end   = max(0, min(n_samples, idx_end))
            idx_post  = max(0, min(n_samples, idx_post))

            labels[idx_pre:idx_start] = EEGPhase.PREICTAL.value
            labels[idx_end:idx_post]  = EEGPhase.POSTICTAL.value
            labels[idx_start:idx_end] = EEGPhase.ICTAL.value

    data = np.vstack([data, validity_mask, labels.reshape(1, -1)])
    data = data.astype(np.float32)
    return data.T


def is_window_covered(win_start: float, win_end: float, summary: PatientSummary, window_size: float) -> bool:
    coverage = 0
    for f in summary.values():
        overlap_start = max(win_start, f['global_start'])
        overlap_end = min(win_end, f['global_end'])
        if overlap_start < overlap_end:
            coverage += (overlap_end - overlap_start)
    
    return coverage >= (window_size*0.99)

# config = PreprocessingConfig(
#     sampling_frequency=256,    # 256 Hz
#     window_size=7200,         # 4 second window (1024 samples)
#     channels=20,              # 22 EEG electrodes
#     dt=1/256                  # Time step (must be 1/fs)
# )

# summary = get_summary('../data/chb-mit-scalp-eeg-database-1.0.0', 3)
# ictal_timepoints = get_ictal_timepoints(summary)
# ranges = get_patient_ranges(summary)
# timedarray = load_data(ranges[6], summary, config)
# # pprint(ranges)
# plot_timed_array(timedarray)

# # pprint(get_patient_ranges(summary))

# # preprocess('../data/chb-mit-scalp-eeg-database-1.0.0', list(range(1, 25)))
# # preprocess('../data/chb-mit-scalp-eeg-database-1.0.0', '../preprocessed', list(range(1, 2)))
# # pprint(get_summary('../data/chb-mit-scalp-eeg-database-1.0.0', 2))

# pprint(timedarray)