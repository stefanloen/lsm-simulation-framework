from pprint import pprint
import numpy as np
from interfaces import Config, EEGPhase, GlobalRecord, RangeInfo, Record, SampleMetadata, SolverType, Split, TaskType, TrainConfig
from scipy.signal import convolve2d
import multiprocessing
from functools import partial
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, RidgeCV
from collections import defaultdict

# Interictal = 0
# Pre-ictal = 1
# Ictal = 2

import os
import glob
import numpy as np
from datetime import datetime

from plotting import plot_record


def store_xy_cache(
        record: Record,
        field: str,
        cache_dir: str):

    raise NotImplementedError("Storing either X or Y values is not implemented")

    xy_true = record.get('xy_true')
    sample_metadata = record.get('sample_metadata')

    cache_dir = os.path.join(cache_dir, f"xy")
    os.makedirs(cache_dir, exist_ok=True)

    base_name = f"Sample{sample_metadata.sample_id:02d}" 
    pattern = os.path.join(cache_dir, f"{base_name}_*.npz")
    for existing_file in glob.glob(pattern):
        os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(cache_dir, f"{base_name}_{timestamp}.npz")

    X, Y = xy_true
    np.savez(
        filepath, 
        X = X,
        Y = Y 
    )

def load_XY_cache(
        records: list[Record], 
        cache_dir: str):
    raise NotImplementedError("Loading either X or Y values is not implemented")
    cache_dir = os.path.join(cache_dir, f"xy")

    
    for record in records:
        sample_metadata = record.get('sample_metadata')
        base_name = f"Sample{sample_metadata.sample_id:02d}"
        pattern = os.path.join(cache_dir, f"{base_name}*.npz")
        matching_files = sorted(glob.glob(pattern)) 
        
        if not matching_files:
            raise FileNotFoundError(f"No XY cache found for {base_name} in {cache_dir}")
        
        filepath = matching_files[-1] # Get the latest timestamp

        with np.load(filepath, allow_pickle=True) as data:
            # Using .copy() ensures the memory is independent of the file handle
            X = data['X'].copy()
            Y = data['Y'].copy()
            record.set('xy_true', (X,Y))
            
def store_trained_params_cache(
        global_record: GlobalRecord, 
        cache_dir: str):

    trained_params = global_record.get('trained_params')

    cache_dir = os.path.join(cache_dir, f"train")
    os.makedirs(cache_dir, exist_ok=True)

    base_name = f"trained_params"
    
    # Clean up old versions
    pattern = os.path.join(cache_dir, f"{base_name}_*.npz")
    for existing_file in glob.glob(pattern):
        os.remove(existing_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(cache_dir, f"{base_name}_{timestamp}.npz")

    # We unpack the dict so each key becomes an array name in the file
    # If the dict contains non-array types, allow_pickle=True handles them
    np.savez(filepath, **trained_params)

def load_trained_params_cache(
        global_record: GlobalRecord,
        cache_dir: str
        ):
    
    cache_dir = os.path.join(cache_dir, f"train")

    base_name = f"trained_params"
    pattern = os.path.join(cache_dir, f"{base_name}*.npz")
    matching_files = sorted(glob.glob(pattern)) 
    
    if not matching_files:
        raise FileNotFoundError(f"No trained params found for {base_name} in {cache_dir}")
    
    filepath = matching_files[-1]

    with np.load(filepath, allow_pickle=True) as data:
        # Convert the NpzFile object back into a standard dictionary
        # .item() is used if you saved the whole dict as one object, 
        # but since we unpacked, dict(data) is cleaner.
        trained_params = {key: data[key] for key in data.files}
        global_record.set('trained_params', trained_params)

class Readout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Readout, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x) # Output logits

class ReadoutQuantized(nn.Module):
    def __init__(self, input_dim, output_dim, levels):
        super(ReadoutQuantized, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.levels = levels

    def _quantize(self, x):
        if self.levels > 0:
            maxval = 0.1
            total_levels = (2 * self.levels) - 1
            x.data.clamp_(-maxval, maxval)
            step = (2 * maxval) / (total_levels - 1)
            q_x = torch.round(x / step) * step
            return x + (q_x - x).detach()
        else:
            return x

    def forward(self, x):
        # Quantize weights and bias before the linear operation
        q_weight = self._quantize(self.fc.weight)
        # q_bias = self._quantize(self.fc.bias)
        q_bias = self.fc.bias
        return nn.functional.linear(x, q_weight, q_bias)

def model_predict(
        config: TrainConfig,
        trained_params: dict, 
        record: Record):
    
    noise_rng = np.random.RandomState(12)
    trace = record.get('reservoir_trace')

    torch.set_num_threads(1)
    state_dict = {}
    for k, v in trained_params.items():
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v).float()
        else:
            state_dict[k] = v

    if config.write_noise > 0:
        w_tensor = state_dict['fc.weight']
        w_min, w_max = w_tensor.min(), w_tensor.max()
        w_range = w_max - w_min
        sigma = config.write_noise * w_range.item()
        noise = torch.from_numpy(noise_rng.normal(0, sigma, size=w_tensor.shape)).float()
        state_dict['fc.weight'] = w_tensor + noise

    output_dim, input_dim = state_dict['fc.weight'].shape
    model = Readout(input_dim=input_dim, output_dim=output_dim)
    
    # 2. Load the trained weights into the new model instance
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(trace.astype(np.float32))
        logits = model(X_tensor)

        match config.task_type:
            case TaskType.CLASSIFICATION:
                output = torch.softmax(logits, dim=1)
            case TaskType.REGRESSION:
                output = logits

    record.set('model_trace', output.detach().cpu().numpy())

def get_model(
        config: TrainConfig,
        records: list[Record], 
        global_record: GlobalRecord):
    torch.manual_seed(12)

    X_train_stacked = np.concatenate([record.get('res_x') for record in records if record.sample_metadata.split == Split.TRAIN], axis=0) #type: ignore
    Y_train_stacked = np.concatenate([record.get('y_true') for record in records if record.sample_metadata.split == Split.TRAIN], axis=0) #type: ignore

    X_validate_stacked = np.concatenate([record.get('res_x') for record in records if record.sample_metadata.split == Split.VALIDATE], axis=0) #type: ignore
    Y_validate_stacked = np.concatenate([record.get('y_true') for record in records if record.sample_metadata.split == Split.VALIDATE], axis=0) #type: ignore

    train_class_indices = np.argmax(Y_train_stacked, axis=1) if Y_train_stacked.ndim > 1 else Y_train_stacked
    validate_class_indices = np.argmax(Y_validate_stacked, axis=1) if Y_validate_stacked.ndim > 1 else Y_validate_stacked

    # Getting criterion
    match config.task_type:
        case TaskType.CLASSIFICATION:
            classes_col_ids = [c.value for c in config.classes]
            train_mask = np.isin(train_class_indices, classes_col_ids)
            validate_mask = np.isin(validate_class_indices, classes_col_ids)

            X_train_temp = X_train_stacked[train_mask]
            Y_train_temp = Y_train_stacked[train_mask][:, classes_col_ids]

            X_validate_temp = X_validate_stacked[validate_mask]
            Y_validate_temp = Y_validate_stacked[validate_mask][:, classes_col_ids]

            if config.balance:
                X_train_temp, Y_train_temp = balance_dataset(X_train_temp, Y_train_temp)

            output_dim = len(config.classes)

            final_indices = np.argmax(Y_train_temp, axis=1)
            class_counts = np.bincount(final_indices, minlength=output_dim)

            pprint(class_counts)

            if np.any(class_counts == 0):
                missing = [config.classes[i].name for i, count in enumerate(class_counts) if count == 0]
                raise ValueError(f"Training error: No Ys found for classes: {missing}")

            print(f"Task: {config.task_type.name} | Total Ys: {X_train_temp.shape[0]}")

            for i, c in enumerate(config.classes):
                print(f" - {c.name} Ys: {class_counts[i]}")

            X_train = torch.from_numpy(X_train_temp).float()
            Y_train = torch.from_numpy(Y_train_temp).float()

            X_validate = torch.from_numpy(X_validate_temp).float() if len(X_validate_stacked) > 0 else None
            Y_validate = torch.from_numpy(Y_validate_temp).float() if len(Y_validate_stacked) > 0 else None

            weights = len(X_train_temp) / (output_dim * class_counts)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())

        case TaskType.REGRESSION:
            X_train = torch.from_numpy(X_train_stacked).float()
            Y_train = torch.from_numpy(Y_train_stacked).float()

            X_validate = None
            Y_validate = None

            output_dim = Y_train_stacked.shape[1]
            criterion = nn.MSELoss()

    # Getting model
    match config.solver_type:
        case SolverType.ADAM:
            model = Readout(input_dim=X_train.shape[1], output_dim=output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_learning_rate)

            best_val_loss = float('inf')
            best_model_state = None
            patience = 50 
            trigger_times = 0

            report_epochs = config.adam_epochs * 0.1
            for epoch in range(config.adam_epochs):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)

                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                
                val_loss = None
                if X_validate is not None:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_validate)
                        val_loss = criterion(val_outputs, Y_validate)

                        # --- Check for Improvement ---
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            trigger_times = 0
                        else:
                            trigger_times += 1

                if (epoch + 1) % report_epochs == 0:
                    val_log = f", Val Loss: {val_loss.item():.4f}" if val_loss is not None else ""
                    print(f'Epoch [{epoch+1}/{config.adam_epochs}], Loss: {loss.item():.4f}{val_log}')

                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Restoring best weights.")
                    break

            if best_model_state:
                model.load_state_dict(best_model_state)

        case SolverType.ADAMQUANTIZED:
            model = ReadoutQuantized(input_dim=X_train.shape[1], output_dim=output_dim, levels=config.qat_levels)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_learning_rate)

            best_val_loss = float('inf')
            best_model_state = None
            patience = config.patience 
            trigger_times = 0

            report_epochs = config.adam_epochs * 0.01
            for epoch in range(config.adam_epochs):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)

                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                
                val_loss = None
                if X_validate is not None:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_validate)
                        val_loss = criterion(val_outputs, Y_validate)

                        # --- Check for Improvement ---
                        if val_loss < best_val_loss: ############
                            best_val_loss = val_loss
                            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            trigger_times = 0
                        else:
                            trigger_times += 1

                if (epoch + 1) % report_epochs == 0:
                    val_log = f", Val Loss: {val_loss.item():.4f}" if val_loss is not None else ""
                    print(f'Epoch [{epoch+1}/{config.adam_epochs}], Loss: {loss.item():.4f}{val_log}')

                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Restoring best weights.")
                    break

            if best_model_state:
                model.load_state_dict(best_model_state)

                with torch.no_grad():
                    model.fc.weight.copy_(model._quantize(model.fc.weight))
                    # model.fc.bias.copy_(model._quantize(model.fc.bias))

        case SolverType.RIDGE:
            print("Solving with RidgeCV")
            solver = RidgeCV(alphas=config.ridge_alphas)
            solver.fit(X_train.numpy(), Y_train.numpy())
            print(f"Optimal Alpha found: {solver.alpha_}")
            
            trained_weights = torch.from_numpy(solver.coef_).float()
            trained_bias = torch.from_numpy(solver.intercept_).float()

            # q_weights, q_bias = quantize_levels(trained_weights, trained_bias)
        
            # 3. Create model and inject solved parameters
            model = Readout(input_dim=X_train.shape[1], output_dim=output_dim)
            model.load_state_dict({
                'fc.weight': trained_weights,
                'fc.bias': trained_bias
            })

    global_record.trained_params = model.state_dict()

def balance_dataset(X, Y, seed=12):
    class_ids = np.argmax(Y, axis=1)
    unique_classes = np.unique(class_ids)
    
    indices_per_class = [np.where(class_ids == c)[0] for c in unique_classes]
    min_samples = min(len(idx) for idx in indices_per_class)
    
    print(f"balancing dataset to {min_samples} Ys")

    rng = np.random.default_rng(seed)
    balanced_indices = []
    
    for indices in indices_per_class:
        rng.shuffle(indices)
        balanced_indices.append(indices[:min_samples])
    
    final_indices = np.concatenate(balanced_indices)
    rng.shuffle(final_indices)
    
    return X[final_indices], Y[final_indices]


def get_sample_times(
    config: Config,
    record: Record,
    global_record: GlobalRecord
):
    label = record.get('label')

    washout_steps = int(config.train_config.washout_period / config.sim_config.analysis_dt)
    # print(washout_steps)
    stride = int(config.train_config.stride / config.sim_config.analysis_dt)
    
    delays = config.train_config.delays
    delay_steps = [int(delay / config.sim_config.analysis_dt) for delay in delays]
    max_delay = max(delay_steps)

    sample_times = []

    raw_mask = label[:, 0].astype(int)

    # Get valid runs (washout not taken into account)
    padded = np.pad(raw_mask, (1, 1), 'constant')
    runs = np.where(np.diff(padded) != 0)[0]

    # print(runs)
    for start, end in zip(runs[::2], runs[1::2]):
        # print(f"{start}, {end}")
        first_sample = start + washout_steps + max_delay
        # print(first_sample)
        sample_indices = np.arange(first_sample, end, stride)
        # print(sample_indices)
        if sample_indices.size > 0:
            sample_times.append(sample_indices * config.sim_config.analysis_dt)

    # print(sample_times)
    # print(f"Sample_times: {sample_times}")
    # plot_record(config, record, global_record)


    sample_times = np.concatenate(sample_times, axis=0)
    # if len(sample_times) > 0:
        
    # else:
    #     sample_times = np.array([])

    record.set('sample_times', sample_times)

def get_res_x(
    config: Config,
    record: Record
):
    res_trace = record.get('reservoir_trace')
    sample_indices = np.round(record.get('sample_times') / config.sim_config.analysis_dt).astype(int)
    res_x = res_trace[sample_indices]
    record.set('res_x', res_x)

def get_true_y(
    config: Config,
    global_record: GlobalRecord,
    record: Record
):
    label = record.get('label')

    delays = config.train_config.delays
    delay_steps = [int(delay / config.sim_config.analysis_dt) for delay in delays]

    sample_indices = np.round(record.get('sample_times') / config.sim_config.analysis_dt).astype(int)

    match config.train_config.task_type: 
        case TaskType.CLASSIFICATION:
            actual_labels = label[:, 1].astype(int)
            Y_raw = actual_labels[sample_indices]
            y_true = np.eye(global_record.get('num_classes'))[Y_raw.astype(int)]
        case TaskType.REGRESSION:
            actual_labels = label[:, 1]
            y_true = np.column_stack([actual_labels[sample_indices - steps] for steps in delay_steps])

    record.set('y_true', y_true)

def get_model_y(
    config: Config,
    record: Record
):
    model_trace = record.get('model_trace')
    sample_indices = np.round(record.get('sample_times') / config.sim_config.analysis_dt).astype(int)
    model_y = model_trace[sample_indices]
    record.set('y_model', model_y)

def get_res_y(
    config: Config,
    record: Record
):
    output_v = record.get('output_v')
    sample_indices = np.round(record.get('sample_times') / config.sim_config.analysis_dt).astype(int)
    output_y = output_v.T[sample_indices]
    record.set('y_res', output_y)


def split_records(records: list, train_ratio: float = 0.7, seed: int = 12):
    rng = np.random.default_rng(seed)
    phase_to_ids = defaultdict(list)
    all_extracted_ids = [] 

    for rec in records:
        if rec.sample_metadata is None:
            continue
            
        phase = rec.sample_metadata.metadata['eegphase']
        if not isinstance(phase, EEGPhase):
            phase = EEGPhase(phase)
            
        s_id = rec.sample_metadata.sample_id
        phase_to_ids[phase].append(s_id)
        all_extracted_ids.append(s_id)

    train_ids = []
    val_ids = []

    for phase, ids in phase_to_ids.items():
        rng.shuffle(ids)
        
        n_total = len(ids)
        n_train = int(n_total * train_ratio)
        
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:])

    return (
        np.sort(np.array(all_extracted_ids)),
        np.sort(np.array(train_ids)), 
        np.sort(np.array(val_ids)), 
    )

from collections import defaultdict

def split_records_labelled(records: list):
    phase_to_ids = defaultdict(list)

    # Collect IDs and their corresponding phases
    for rec in records:
        if rec.sample_metadata is None:
            continue
            
        phase = rec.sample_metadata.metadata['eegphase'].value
    
        s_id = rec.sample_metadata.sample_id
        phase_to_ids[phase].append((s_id, phase))

    all_ids = []
    all_labels = []

    # Flatten the dict into sorted lists
    # We sort by s_id to ensure a deterministic order
    combined_data = []
    for phase, pairs in phase_to_ids.items():
        combined_data.extend(pairs)
    
    # Sort by the ID (the first element of the pair)
    combined_data.sort(key=lambda x: x[0])
    
    # Unzip into separate arrays
    all_ids = [item[0] for item in combined_data]
    all_labels = [item[1] for item in combined_data]

    return np.array(all_ids), np.array(all_labels)

def quantize_levels(weights, bias, levels=256):
    """
    Quantizes weights for a symmetric differential memristor pair.
    
    levels: Total discrete states per memristor (e.g., 16 for 4-bit).
    """
    # 1. Symmetric Scaling: 0 is always the midpoint
    w_max = torch.max(torch.abs(weights))
    scale = 1.0 / (w_max + 1e-8)
    
    w_scaled = weights * scale
    b_scaled = bias * scale

    # 2. Split into positive and negative conductance arrays
    g_pos = torch.clamp(w_scaled, min=0)
    g_neg = torch.clamp(-w_scaled, min=0)

    # 3. Quantize based on the number of gaps (levels - 1)
    # This ensures 0.0 is the first level and 1.0 is the last
    step = 1.0 / (levels - 1)

    g_pos_q = torch.round(g_pos / step) * step
    g_neg_q = torch.round(g_neg / step) * step

    # 4. Reconstruct effective weights
    q_weights = g_pos_q - g_neg_q
    
    # 5. Quantize bias using the same grid
    # We treat bias symmetrically as well
    b_pos = torch.clamp(b_scaled, min=0)
    b_neg = torch.clamp(-b_scaled, min=0)
    q_bias = (torch.round(b_pos / step) - torch.round(b_neg / step)) * step

    return q_weights, q_bias