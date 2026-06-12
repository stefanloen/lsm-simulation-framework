from pprint import pprint
import numpy as np
from scipy.signal import lfilter

from interfaces import Config, EEGPhase, Record, SimConfig, Split, TaskType, TrainConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, r2_score, accuracy_score
from scipy import stats

def get_trace(
        config: SimConfig,
        spike_tuple: tuple[tuple[np.ndarray, np.ndarray], int], 
        duration: float,
        tau: float
    ) -> np.ndarray:

    (times, indices), channels = spike_tuple
    num_bins = int(np.ceil(duration / config.analysis_dt))
    
    binned_spikes = np.zeros((num_bins, channels), dtype=np.float32)
    bin_indices = np.clip((times / config.analysis_dt).astype(int), 0, num_bins - 1)
    np.add.at(binned_spikes, (bin_indices, indices), 1)
    
    # Equation: y[n] = x[n] + alpha * y[n-1]
    alpha = np.exp(-config.analysis_dt / tau)
    
    # lfilter calculates: a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1]... - a[1]*y[n-1]...
    b = [1.0]
    a = [1.0, -alpha]
    
    # Apply filter along the time axis (axis 0)
    trace = lfilter(b, a, binned_spikes, axis=0)
    
    # # Normalize (optional, to match your kernel normalization)
    # # The steady-state gain of this IIR filter is 1 / (1 - alpha)
    # trace /= (1.0 / (1.0 - alpha)) 
    
    return trace

def analyze_trace_rank(
    config: SimConfig,
    trace: np.ndarray, 
    tau: float, 
    variance_threshold: float = 0.95
) -> tuple[int, PCA]:
    num_bins, num_channels = trace.shape
    
    # 1. Slice warm-up
    warmup_bins = int((5 * tau) / config.analysis_dt)
    trace_steady = trace[warmup_bins:, :]

    # 2. Filter inactive neurons
    active_mask = np.var(trace_steady, axis=0) > 1e-9
    active_count = np.sum(active_mask)
    filtered_trace = trace_steady[:, active_mask]

    # 3. PCA Calculation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_trace)
    pca = PCA().fit(scaled_data)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    true_rank = int((np.argmax(cumulative_variance >= variance_threshold) + 1))
    
    # --- Interesting Prints ---
    # print(f"\n" + "="*40)
    # print(f"📊 PCA ANALYSIS: {label.upper()}")
    # print(f"="*40)
    # print(f"🔹 Physical Dimensions: {num_channels} neurons over {num_bins * dt:.2f}s")
    # print(f"🔹 Active Neurons:     {active_count}/{num_channels} ({(active_count/num_channels)*100:.1f}%)")
    
    if active_count > 0:
        # print(f"🔹 Effective Rank:     {true_rank} dimensions")
        # print(f"🔹 Compression Ratio:  {num_channels / true_rank:.2f}x")
        
        # Checking how much the top 3 components dominate
        top_3 = sum(pca.explained_variance_ratio_[:3]) * 100
        # print(f"🔹 Variance in PC1-3:  {top_3:.1f}%")
        
        if top_3 > 90:
            pass
            # print("⚠️  Warning: High redundancy! Most activity is trapped in a 3D subspace.")
        else:
            pass
            # print("✅ Rich Dynamics: Information is well-distributed across the population.")
    else:
        pass
        # print("❌ Error: No neural activity detected in trace.")
    
    # print("="*40)
    
    return true_rank, pca



def calculate_pca_distance(
    trace_A: np.ndarray, 
    trace_B: np.ndarray, 
    variance_threshold: float = 0.95
) -> tuple[np.ndarray, int]:
    """
    Projects two traces into a shared PCA space and calculates 
    the Euclidean distance between them at every time step.
    """
    # 1. Combine traces to define the shared state space
    # (Vertical stack: rows are time steps, columns are neurons)
    combined_data = np.vstack([trace_A, trace_B])
    
    # 2. Standardize (Crucial for PCA so high-rate neurons don't dominate)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_data)
    
    # 3. Fit PCA on the combined space
    pca = PCA(n_components=variance_threshold)
    pca.fit(combined_scaled)
    
    # 4. Project Trace A and Trace B individually into the new space
    # We must scale them using the SAME scaler used for fitting
    space_A = pca.transform(scaler.transform(trace_A))
    space_B = pca.transform(scaler.transform(trace_B))
    
    # 5. Calculate Pointwise Euclidean Distance
    # np.linalg.norm with axis=1 gives distance at every time bin
    distances = np.linalg.norm(space_A - space_B, axis=1)
    
    # Returns the distance array and the number of components (Rank)
    return distances, pca.n_components_

def get_separation_factor(
          in_a: np.ndarray,
          in_b: np.ndarray,
          out_a: np.ndarray,
          out_b: np.ndarray,
          last_n: int
) -> float:
        in_distances, _ = calculate_pca_distance(in_a, in_b)
        out_distances, _ = calculate_pca_distance(out_a, out_b)

        in_segment = in_distances[-last_n:]
        out_segment = out_distances[-last_n:]

        avg_in_dist = np.mean(in_segment)
        avg_out_dist = np.mean(out_segment)

        if avg_in_dist < 1e-9:
             return 0.0

        separation_factor = avg_out_dist / avg_in_dist

        return float(separation_factor)

def get_convergence_factor(
          run1: np.ndarray,
          run2: np.ndarray,
          last_n: int
) -> tuple[float, np.ndarray]:
        distances, _ = calculate_pca_distance(run1, run2)
        midpoint = len(distances) //2

        last_n_random = distances[midpoint - last_n : midpoint]
        last_n_controlled = distances[-last_n:]

        random_dist = np.mean(last_n_random)
        controlled_dist = np.mean(last_n_controlled)

        convergence_factor = random_dist/(controlled_dist + 1e-9)

        return float(convergence_factor), distances

def get_samples_to_converge(
    run1: np.ndarray,
    run2: np.ndarray,
    from_N: int,
    threshold: float,
) -> int:
    distances, _ = calculate_pca_distance(run1, run2)
    
    start_idx = from_N
    search_space = distances[start_idx:]
    
    above_threshold = np.where(search_space > threshold)[0]
    
    if len(above_threshold) == 0:
        return 0
        
    last_unconverged_idx = above_threshold[-1]
    
    converged_idx = start_idx + last_unconverged_idx + 1
    if converged_idx >= len(distances):
        return -1 # Never converged
        
    return converged_idx - from_N
    

def get_performance_metrics(
          task_type: TaskType,
          classes: list,
          records: list[Record],
          output_type: str,
          balance: bool = False) -> dict:

    pprint(len(records))
    records = [r for r in records if r.sample_metadata.split == Split.VALIDATE]
    pprint(len(records))

    y_true = [record.get('y_true') for record in records]
    
    match output_type:
        case 'model':
            y_pred = [record.get('y_model') for record in records]

        case 'res':
            y_pred = [record.get('y_res') for record in records]

        case _ :
            raise ValueError('Invalid output type')

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    performance = {}

    match task_type:
        case TaskType.CLASSIFICATION:
            target_col_ids = [c.value for c in classes]
            y_true_indices_full = np.argmax(y_true, axis=1)
            mask = np.isin(y_true_indices_full, target_col_ids)
            
            y_true = y_true[mask][:, target_col_ids]
            
            y_pred = y_pred[mask]


            if balance:
                y_true, y_pred = balance_dataset(y_true, y_pred)

            y_true_idx = np.argmax(y_true, axis=1)
            y_pred_idx = np.argmax(y_pred, axis=1)

            target_names = [c.name for c in classes]

            performance['accuracy'] = accuracy_score(y_true_idx, y_pred_idx)
            performance['f1_macro'] = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
            performance['classification_report'] = classification_report(
                y_true_idx, 
                y_pred_idx, 
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )

        case TaskType.REGRESSION:
            r2_scores = r2_score(y_true, y_pred, multioutput='raw_values')
            performance['r2_scores'] = r2_scores
            performance['sum_r2_scores'] = np.sum(np.maximum(0, r2_scores)) 

    return performance

def balance_dataset(y_true, y_predict, seed=12):
    class_ids = np.argmax(y_true, axis=1)
    
    num_classes = y_true.shape[1] 
    
    indices_per_class = [np.where(class_ids == c)[0] for c in range(num_classes)]
    
    counts = [len(idx) for idx in indices_per_class]
    print(f"Detected class distribution: {counts}")
    
    active_counts = [c for c in counts if c > 0]
    if not active_counts:
        return y_true, y_predict
        
    min_samples = min(active_counts)
    print(f"Balancing dataset to {min_samples} Ys per class")

    rng = np.random.default_rng(seed)
    balanced_indices = []
    
    for indices in indices_per_class:
        if len(indices) > 0:
            shuffled_idx = indices.copy()
            rng.shuffle(shuffled_idx)
            balanced_indices.append(shuffled_idx[:min_samples])
    
    final_indices = np.concatenate(balanced_indices)
    rng.shuffle(final_indices)
    
    return y_true[final_indices], y_predict[final_indices]

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

def get_spectral_radius(template_res_i, template_res_j, W_vals, N_res):
    """
    Calculates the spectral radius of the effective weight matrix.
    
    Args:
        template_res_i: Source neuron indices (from conn_mask)
        template_res_j: Target neuron indices (from conn_mask)
        W_vals: The normalized reservoir weight matrix (N_res x N_res)
        N_res: Total number of reservoir neurons
    """
    # 1. Extract the active weights from the template
    active_weights = W_vals[template_res_i, template_res_j]
    
    # 2. Build the effective weight matrix in sparse format
    # This matches the normalized 'W_vals' used in your synapses
    W_eff = csr_matrix((active_weights, (template_res_i, template_res_j)), 
                       shape=(N_res, N_res))
    
    # 3. Compute the largest eigenvalue (magnitude)
    # 'LM' stands for Largest Magnitude
    eigenvalues = eigs(W_eff, k=1, which='LM', return_eigenvectors=False)
    
    spectral_radius = np.abs(eigenvalues[0])
    return spectral_radius

from scipy import stats
import numpy as np

from scipy import stats
import numpy as np

def calculate_separation_metrics(results_matrix: np.ndarray) -> tuple[float, float, float, float, float]:
    # results_matrix columns: [dist_u, dist_x, label_i, label_j]
    in_dist = results_matrix[:, 0]   # dist_u
    out_dist = results_matrix[:, 1]  # dist_x
    label_i = results_matrix[:, 2]
    label_j = results_matrix[:, 3]

    slope_val, intercept_val, _, _, _ = stats.linregress(in_dist, out_dist)
    
    slope = float(slope_val)
    intercept = float(intercept_val)

    intra_mask = (label_i == label_j)
    inter_mask = (label_i != label_j)

    cv = float(np.mean(out_dist[intra_mask])) if np.any(intra_mask) else 0.0
    cd = float(np.mean(out_dist[inter_mask])) if np.any(inter_mask) else 0.0
    
    sep = cd / cv if cv != 0.0 else 0.0
    
    return cv, cd, sep, slope, intercept