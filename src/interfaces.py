from enum import Enum, auto
from functools import total_ordering
from typing import Any, TypedDict, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass, fields

import numpy as np

@total_ordering
class OrderedEnum(Enum):
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class Split(Enum):
    TRAIN = 0
    VALIDATE = 1

class EEGPhase(Enum):
    INTERICTAL = 0
    PREICTAL = 1
    ICTAL = 2
    POSTICTAL = 3
    ONSET = 4

class ClassNumber(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2

class NetworkLocation(OrderedEnum):
    DATA = auto()
    PREPROCESSOR_OUT = auto()
    ENCODER_OUT = auto()
    RESERVOIR_OUT = auto()
    XY = auto()
    TRAIN = auto()
    MODEL_OUTPUT = auto()
    OUTPUT = auto()

class Mode(Enum):
    RUN_CACHE = auto()
    RUN_CACHE_INBETWEEN = auto()
    RUN_PLOT = auto()
    RUN_CACHE_PLOT = auto()
    RUN_CACHE_INBETWEEN_PLOT = auto()
    PLOT = auto()

class TaskType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()

class SolverType(Enum):
    ADAM = auto()
    ADAMQUANTIZED = auto()
    RIDGE = auto()

@dataclass
class SampleMetadata():
    sample_id: int
    split: Split
    metadata: Any

class RangeInfo(TypedDict):
    patient_id: str
    range_id: int
    eegphase: EEGPhase             
    range: Tuple[int, int]

class FileSummary(TypedDict):
    patient_dir: str
    sample_rate: int
    seizure_count: int
    seizures: List[Dict[str, float]]
    global_start: float
    global_end: float

PatientSummary = Dict[str, FileSummary]
Summaries = Dict[str, PatientSummary]

@dataclass
# @dataclass(frozen=True)
class CHBPreprocessingConfig:
    cutoff_freq: float
    notch_freqs: np.ndarray | None
    rereference: bool
    fixed_normalization: bool 
    fixed_median: float
    fixed_scale: float

    window_size: int
    channels: int

    preictal_duration: float
    postictal_duration: float

    get_interictal: bool
    get_preictal: bool
    get_ictal: bool
    get_onset: bool

@dataclass
class BONNPreprocessingConfig:
    cutoff_freq: float
    fixed_normalization: bool 
    fixed_median: float
    fixed_scale: float

@dataclass
# @dataclass(frozen=True)
class PoissonPreprocessingConfig:
    sample_count: int
    random_duration: float
    controlled_duration: float
    silent_duration: float
    similarity: float
    channels: int
    rate: float
    min_isi: float

@dataclass
class PoissonMaassPreprocessingConfig:
    sample_count: int
    channels: int
    rate: float
    lag: float
    min_isi: float
    n_patterns: int
    n_repeat_min: int
    n_repeat_max: int
    pattern_duration: float
    num_seq_patterns: int
    jitter_std: float
    deletion_p: float
    injection_rate: float

    chunk: float | None

@dataclass
class WhiteNoiseConfig:
    sample_count: int
    duration: float
    cutoff_freq: float

class EncodingType(Enum):
    POISSON = auto()
    THRESHOLD = auto()
    NOISE = auto()
    BINNING = auto()
    BINNING_GAUSSIAN = auto()
    LOGBINNING = auto()
    HYBRID = auto()
    DELTA = auto()

@dataclass
# @dataclass(frozen=True)
class EncodingConfig:
    encoding_type: EncodingType
    binning_bins: int
    binning_vmin: float
    binning_vmax: float
    binning_rate: float
    binning_sigma: float
    binning_k: float
    delta_size: float

@dataclass
class PoissonConfig:
    encoding_type: EncodingType

@dataclass
class SimpleLIF:
    threshold_v: float
    leak_tau: float
    v_rest: float
    refractory_period: float

@dataclass
class MaassLIF:
    leak_tau: float
    threshold_v: float
    v_rest: float
    reset_v: float
    background_I: float
    input_resistance: float
    tau_syn_exc: float
    tau_syn_inh: float
    refractory_period: float

@dataclass
class SimpleLI:
    leak_tau: float
    v_rest: float

@dataclass
class Fixed:
    value: float

@dataclass
class Uniform:
    min: float
    max: float

@dataclass
class Gaussian:
    mean: float
    std: float

@dataclass
class Beta:
    mean: float
    max_dev: float

@dataclass
class SimpleSyn:
    w: Fixed | Uniform | Gaussian | Beta
    syn_delay: float

@dataclass 
class MarkramSyn:
    A: Fixed | Uniform | Gaussian | Beta
    util: Fixed | Uniform | Beta
    tau_rec: Fixed | Uniform | Beta
    tau_fac: Fixed | Uniform | Beta
    syn_delay: float

@dataclass
class Synapse:
    EE: SimpleSyn | MarkramSyn
    EI: SimpleSyn | MarkramSyn
    IE: SimpleSyn | MarkramSyn
    II: SimpleSyn | MarkramSyn
    levels: int
    write_noise: float

@dataclass
class Neuron:
    E: SimpleLIF | MaassLIF | SimpleLI
    I: SimpleLIF | MaassLIF | SimpleLI

@dataclass
# @dataclass(frozen=True)
class ReservoirConfig:
    # Neurons
    N: int
    factor_inh: float

    # Input synapses
    F_in: int 
    synapse_in: Synapse

    # Reservoir synapses
    C_EE: float
    C_EI: float
    C_IE: float
    C_II: float
    lamda: float
    spectral_radius: float
    synapse_res: Synapse

    # Reservoir neurons
    neuron_res: Neuron

@dataclass
class OutputConfig:
    synapse: Synapse
    li_config: SimpleLI

@dataclass
# @dataclass(frozen=True)
class TrainConfig:
    task_type: TaskType
    classes: list
    balance: bool
    solver_type: SolverType
    adam_epochs: int
    adam_learning_rate: float
    ridge_alphas: np.ndarray
    qat_levels: int
    patience: int

    stride: float

    washout_period: float
    delays: np.ndarray

    reservoir_trace_tau: float
    encoder_trace_tau: float
    write_noise: float = 0

# @dataclass(frozen=True)
@dataclass
class SimConfig:
    brian_dt: float
    analysis_dt: float
    spike_threshold: float
    spike_threshold_tau: float

    train: bool

    encoder_spikes: bool
    encoder_trace: bool
    encoder_pca: bool
    reservoir_v: bool
    reservoir_spikes: bool
    reservoir_trace: bool
    reservoir_pca: bool

    get_res_x: bool
    get_true_y: bool
    get_model_y: bool
    get_out_y: bool

    get_model_trace: bool

    cache_preprocessor: bool
    cache_encoder: bool
    cache_reservoir: bool

    cache_res_x: bool
    cache_y_true: bool
    cache_y_model: bool
    cache_y_out: bool

    cache_trained_params: bool
    cache_output: bool

@dataclass
# @dataclass(frozen=True)
class Config:
    sim_config: SimConfig 
    preprocessing_config: CHBPreprocessingConfig | BONNPreprocessingConfig | PoissonPreprocessingConfig | WhiteNoiseConfig | PoissonMaassPreprocessingConfig
    encoding_config: EncodingConfig
    reservoir_config: ReservoirConfig
    output_config: OutputConfig
    train_config: TrainConfig


@dataclass
class Record:
    sample_metadata: SampleMetadata | None = None
    label: np.ndarray | None = None
    input_data: np.ndarray | None = None
    encoder_spikes: tuple[tuple[np.ndarray, np.ndarray], int] | None = None
    reservoir_spikes: tuple[tuple[np.ndarray, np.ndarray], int] | None = None
    reservoir_v: tuple[np.ndarray, np.ndarray] | None = None
    output_v: np.ndarray | None = None
    encoder_trace: np.ndarray | None = None
    reservoir_trace: np.ndarray | None = None
    sample_times: np.ndarray | None = None
    res_x: np.ndarray | None = None
    y_true: np.ndarray | None = None
    y_model: np.ndarray | None = None
    y_res: np.ndarray | None = None
    
    model_trace: np.ndarray | None = None
    encoder_rank: int | None = None
    reservoir_rank: int | None = None
    encoder_rate: float | None = None
    reservoir_rate: float | None = None

    def set(self, field_name: str, value: Any):
        if not hasattr(self, field_name):
            raise AttributeError(f"'{type(self).__name__}' has no field '{field_name}'")
        
        current_val = getattr(self, field_name)
        if current_val is not None:
            raise ValueError(
                f"Field '{field_name}' has already been set and cannot be overwritten. "
                f"Current value exists for Record {self.sample_metadata}."
            )
            
        setattr(self, field_name, value)

    def __repr__(self):
        # Clean repr that shows which fields are populated
        populated = [f.name for f in fields(self) if getattr(self, f.name) is not None]
        return f"Record(populated_fields={populated})"
    
    def get(self, field_name: str):
        val = getattr(self, field_name)
        if val is None:
            raise ValueError(f"Required field '{field_name}' is missing from Record!")
        return val
    
@dataclass
class GlobalRecord:
    cutoff_freq: float | None = None
    duration: float | None = None
    num_classes: int  | None = None
    trained_params: dict | None = None
    metadata: Any | None = None
    performance: dict | None = None

    def set(self, field_name: str, value: Any):
        if not hasattr(self, field_name):
            raise AttributeError(f"'{type(self).__name__}' has no field '{field_name}'")
        
        current_val = getattr(self, field_name)
        if current_val is not None:
            raise ValueError(
                f"Field '{field_name}' has already been set and cannot be overwritten. "
            )
            
        setattr(self, field_name, value)

    def __repr__(self):
        # Clean repr that shows which fields are populated
        populated = [f.name for f in fields(self) if getattr(self, f.name) is not None]
        return f"Record(populated_fields={populated})"
    
    def get(self, field_name: str):
        val = getattr(self, field_name)
        if val is None:
            raise ValueError(f"Required field '{field_name}' is missing from Record!")
        return val