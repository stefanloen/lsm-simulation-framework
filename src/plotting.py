from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

from interfaces import Config, GlobalRecord, Record, TaskType

def plot_spikes(spikes, title="Neural Spike Raster"):
    t, i = spikes

    # Safety check for empty data
    if t is None or len(t) == 0:
        print("No spikes to plot.")
        return

    plt.figure(figsize=(12, 6))
    
    # Using marker='|' makes the dots look like real spikes
    # s=2 ensures markers don't overlap too much
    plt.scatter(t, i, s=2, marker='|', color='black', linewidths=0.5)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    
    # Standard neuroscience styling
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()

def plot_array_batches(array_batches, config, title="Input Data Batches"):
    """
    Plots each batch in array_batches as a separate subplot.
    Assumes array_batches is a list of [samples, channels].
    """
    num_batches = len(array_batches)
    if num_batches == 0:
        print("No data to plot.")
        return

    # Calculate time axis using config
    fs = config.preprocessing_config.sampling_frequency
    dt = 1.0 / fs
    
    fig, axes = plt.subplots(num_batches, 1, figsize=(12, 2 * num_batches), sharex=True)
    
    # Ensure axes is always an array even for 1 batch
    if num_batches == 1:
        axes = [axes]

    for i, batch in enumerate(array_batches):
        num_samples = batch.shape[0]
        time_axis = np.arange(num_samples) * dt
        
        # Plot each channel in the batch
        # We add an offset to each channel to separate them visually (Stacked plot)
        offset = np.max(np.abs(batch)) * 1.5
        for ch in range(batch.shape[1]):
            axes[i].plot(time_axis, batch[:, ch] + (ch * offset), linewidth=0.7)
        
        axes[i].set_ylabel(f"Batch {i}")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_record(
        config: Config,
        record: Record, 
        global_record: GlobalRecord, 
        title="Record Visualization"):
    """
    Plots all available data from a Record object.
    Uses getattr for discovery to avoid the ValueError from record.get().
    """
    # 1. Configuration Discovery (Safe Access)
    # Using getattr here because global_metadata might be partially filled
    n_classes = getattr(global_record, 'num_classes', 3)
    n_classes = 1
    cmap = plt.cm.get_cmap('tab10')
    
    shade_map = {1: 'yellow', 2: 'red'}
    class_labels = {}# {0: 'Interictal', 1: 'Preictal', 2: 'Ictal'}

    # Simplified Discovery: Just check for existence
    plots_config = {
        "Input EEG": getattr(record, 'input_data') is not None,
        "Encoder Spikes": getattr(record, 'encoder_spikes') is not None,
        "Encoder Trace": getattr(record, 'encoder_trace') is not None,
        "Reservoir Spikes": getattr(record, 'reservoir_spikes') is not None,
        "Reservoir Voltage": getattr(record, 'reservoir_v') is not None,
        "Output Activity": getattr(record, 'output_v') is not None,
        "Reservoir Trace": getattr(record, 'reservoir_trace') is not None,
        "Probabilities": getattr(record, 'model_trace') is not None
    }
    
    active_plots = [name for name, active in plots_config.items() if active]
    if not active_plots:
        print(f"Nothing to plot: {record}") # Uses your new __repr__
        return

    # 2. Setup Time Axis
    if plots_config["Input EEG"]:
        # Now safe to use .get() because we verified it's not None
        input_data = record.get('input_data')
        data = input_data if input_data.shape[0] < input_data.shape[1] else input_data.T
        time_steps = data.shape[1]
    else:
        label = getattr(record, 'label')
        time_steps = len(label) if label is not None else 1000
    
    x_time = np.arange(time_steps) * config.sim_config.analysis_dt

    # 3. Initialize Figure
    fig, axes = plt.subplots(len(active_plots), 1, figsize=(15, 4 * len(active_plots)), sharex=True)
    if len(active_plots) == 1: axes = [axes]
    
    def add_shading(ax, ymin, ymax):
        label = getattr(record, 'label')
        if config.train_config.task_type is TaskType.CLASSIFICATION and label is not None:
            for c_id in range(1, 10):
                mask = (label[:, 1] == c_id) if label.ndim > 1 else (label == c_id)
                ax.fill_between(x_time, ymin, ymax, where=mask[:len(x_time)], 
                                color=shade_map.get(c_id, 'gray'), alpha=0.2, zorder=0)

    curr = 0

    # 4. Plotting Panels (Using .get() safely now)
    if plots_config["Input EEG"]:
        ax = axes[curr]
        offset = np.percentile(np.abs(data), 99) * 2.0
        for i in range(data.shape[0]):
            ax.plot(x_time, data[i, :] + i*2, lw=0.7, color='#2c3e50', alpha=0.8)
        add_shading(ax, -1, data.shape[0] * 2)
        ax.set_title(f"{title} - Raw EEG")
        curr += 1

    if plots_config["Encoder Spikes"]:
        ax = axes[curr]
        spikes, _ = record.get('encoder_spikes') # Unpacking the spikes tuple
        ax.scatter(spikes[0], spikes[1], s=5, c='black', marker='|')
        add_shading(ax, 0, np.max(spikes[1]) if spikes[1].size else 1)
        ax.set_title(f"{title} -Encoder Spikes (Indices vs Time)")
        curr += 1

    if plots_config["Encoder Trace"]:
        ax = axes[curr]
        trace = record.get('encoder_trace')
        im = ax.imshow(trace.T, aspect='auto', cmap='viridis', origin='lower',
                        extent=[x_time[0], x_time[-1], 0, trace.shape[1]])
        plt.colorbar(im, ax=ax, label="Activity")
        ax.set_title(f"{title} - Encoder Rank/Activity Trace")
        curr += 1

    if plots_config["Reservoir Spikes"]:
        ax = axes[curr]
        res_spikes, _ = record.get('reservoir_spikes')
        ax.scatter(res_spikes[0], res_spikes[1], s=2, c='black', marker='|')
        add_shading(ax, 0, np.max(res_spikes[1]) if res_spikes[1].size else 1)
        ax.set_title(f"{title} -Reservoir Spikes")
        curr += 1

    if plots_config["Reservoir Voltage"]:
        ax = axes[curr]
        v_time, v_val = record.get('reservoir_v')
        ax.plot(v_time, v_val, color='#e67e22')
        add_shading(ax, np.min(v_val), np.max(v_val))
        ax.set_title(f"{title} -Reservoir Membrane Potential (Sample)")
        curr += 1

    if plots_config["Reservoir Trace"]:
        ax = axes[curr]
        res_trace = record.get('reservoir_trace')
        im = ax.imshow(res_trace.T, aspect='auto', cmap='viridis', origin='lower',
                        extent=[x_time[0], x_time[-1], 0, res_trace.shape[1]])
        if record.sample_times is not None:
            sample_times = record.get('sample_times')
            ax.vlines(sample_times, 0, res_trace.shape[1], 
                      colors='red', linestyles='--', alpha=0.5, lw=1)
        ax.set_title(f"{title} -Reservoir Activity Trace")
        curr += 1

    if plots_config["Output Activity"]:
        ax = axes[curr]
        out_v = record.get('output_v')
        plot_data = out_v[:, :len(x_time)] 
        for i in range(plot_data.shape[0]):
            ax.plot(x_time, plot_data[i, :], label=class_labels.get(i, f"Class {i}"), color=cmap(i))
        add_shading(ax, np.min(out_v), np.max(out_v))
        ax.set_title(f"{title} - Readout Layer Integration")
        curr += 1

    if plots_config["Probabilities"]:
        ax = axes[curr]
        output = record.get('model_trace')
        plot_probs = output[:len(x_time), :]
        for i in range(plot_probs.shape[1]):
            ax.plot(x_time, plot_probs[:, i], label=class_labels.get(i, f"Class {i}"), color=cmap(i), lw=2)

        if config.train_config.task_type is TaskType.REGRESSION and getattr(record, 'label') is not None:
            ax.plot(x_time, getattr(record, 'label'), label='label', color='black', lw=1)

        add_shading(ax, 0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        ax.set_title("Final Classification Confidence")
        curr += 1

    # Cleanup
    for ax in axes: 
        ax.set_xlabel("Time (Seconds)")
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    plt.show()

def visualise_connectivity(S):
    print("Lekker plotten")
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    
    # 1. Check for is_inh variable safely
    # S.source refers to the NeuronGroup. We check if it has 'is_inh'
    source_is_inh = getattr(S.source, 'is_inh', None)
    target_is_inh = getattr(S.target, 'is_inh', None)

    # Subplot 1: Schematic of connections
    plt.subplot(121)
    
    # Function to get color array: Blue if Exc (1), Red if Inh (0), Black if unknown
    def get_colors(is_inh_array, N):
        if is_inh_array is not None:
            return ['red' if val else 'black' for val in is_inh_array]
        return ['black'] * N

    # Plot source and target nodes with colors
    plt.scatter(np.zeros(Ns), np.arange(Ns), c=get_colors(source_is_inh, Ns), s=20, zorder=3)
    plt.scatter(np.ones(Nt), np.arange(Nt), c=get_colors(target_is_inh, Nt), s=20, zorder=3)

    # Plot connection lines (colored by source neuron type)
    for idx in range(len(S.i)):
        pre_idx = S.i[idx]
        post_idx = S.j[idx]
        line_color = 'black'
        if source_is_inh is not None:
            line_color = 'red' if source_is_inh[pre_idx] else 'black'
        
        plt.plot([0, 1], [pre_idx, post_idx], color=line_color, lw=0.5, alpha=0.3)

    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    # plt.ylim(-1, max(Ns, Nt))

    # Subplot 2: Adjacency Matrix
    plt.subplot(122)
    
    # Color the dots in the matrix by the source (i) neuron type
    matrix_colors = 'black'
    if source_is_inh is not None:
        # S.i is the array of source indices for every existing synapse
        matrix_colors = ['red' if source_is_inh[i] else 'black' for i in S.i]

    plt.scatter(S.i, S.j, c=matrix_colors, s=2, alpha=0.8)
    
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.title('Connectivity Matrix')
    plt.tight_layout()

    plt.show()


def print_active_percentage(synapses: b2.Synapses, N_res, num_par):
    # Number of synapses actually created
    num_active = len(synapses)
    total_possible = num_par * (N_res ** 2) 
    connectivity_ratio = num_active / total_possible
    percentage = connectivity_ratio * 100

    print(f"Reservoir Connectivity: {percentage:.2f}% ({num_active}/{total_possible})")


def plot_reservoir_convergence(dist_profile, labels, sampling_freq, threshold_t = 0.0, pair_id=0):
    """
    Plots the Euclidean distance between a pair of traces over time.
    """
    time_vec = np.arange(len(dist_profile)) / sampling_freq
    
    plt.figure(figsize=(12, 5))
    
    # 1. Plot the Distance Curve
    plt.plot(time_vec, dist_profile, color='#2c3e50', lw=1.5, label='PCA Distance')
    
    # 2. Color the background based on labels
    # We find where the label changes from 0 to 1/2
    phase_change_idx = np.where(labels[:, 1] > 1)[0][0]
    switch_time = phase_change_idx / sampling_freq
    
    plt.axvspan(0, switch_time, color='red', alpha=0.1, label='Prefix (Divergence)')
    plt.axvspan(switch_time, time_vec[-1], color='green', alpha=0.1, label='Suffix (Convergence)')
    
    # 2. Add Vertical Line for Convergence Time
    if threshold_t > 0:
        plt.axvline(x=threshold_t, color='red', linestyle='--', lw=2, 
                    label=f'Convergence Time ({threshold_t:.2f}s)')

    # 3. Aesthetics
    plt.title(f"Reservoir Convergence Test: Pair {pair_id}", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("PCA Euclidean Distance", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    # Annotate the final distance
    final_dist = np.mean(dist_profile[-int(0.5*sampling_freq):])
    plt.annotate(f'Final Steady State: {final_dist:.4f}', 
                 xy=(time_vec[-1], final_dist), 
                 xytext=(time_vec[-1]-1.5, final_dist + np.max(dist_profile)*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_memory_curve(delays, r2_scores, title="Reservoir Memory Profile"):
    """
    Plots R2 scores against time delays to visualize fading memory.
    
    Args:
        delays (np.array): The time delays used (e.g., np.arange(0.0, 10.0, 0.5))
        r2_scores (list/np.array): The R2 score calculated for each delay
        title (str): Plot title
    """
    # Calculate the sum for the legend (to match your Optuna attribute)
    total_r2 = np.sum(r2_scores)
    
    plt.figure(figsize=(10, 5))
    
    # Plotting the curve
    plt.plot(delays, r2_scores, marker='o', linestyle='-', color='#2ca02c', label=f'Total $R^2$ Sum: {total_r2:.3f}')
    
    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel("Delay (ms or time steps)", fontsize=12)
    plt.ylabel("$R^2$ Score", fontsize=12)
    plt.ylim(-0.1, 1.1)  # R2 is capped at 1.0; showing slightly below 0 for perspective
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Optional: Add a "Fading Memory" threshold line
    plt.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Assuming your TrainConfig delays were: np.arange(0.0, 10.0, 0.5)
# And your r2_scores is an array of 20 values
# plot_memory_curve(config.train_config.delays, model_performance['r2_scores'])

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def plot_reservoir(config, template_res_i, template_res_j, signs_template, template_i, template_j, num_in, num_out):
    N_res = config.N
    root = round(N_res ** (1/3))
    
    indices = np.arange(N_res)
    x = (indices % root)
    y = ((indices // root) % root)
    z = (indices // (root**2))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_proj_type('persp', focal_length=0.2)

    # 1. Plot Reservoir Neurons
    colors = ['red' if s == -1 else 'blue' for s in signs_template]
    ax.scatter(x, y, z, c=colors, s=50, alpha=0.8, label='Reservoir')

    # 2. Plot Input Tower (Vertical column at x = -2)
    in_x = np.full(num_in, -2)      # Offset from the reservoir
    in_y = np.full(num_in, root/2)  # Keep it in the middle of the Y-plane
    in_z = np.linspace(0, root - 1, num_in) # Spread it vertically along Z
    ax.scatter(in_x, in_y, in_z, c='green', s=60, marker='^', label='Input')

    # 3. Plot Reservoir Connections
    for i in range(len(template_res_i)):
        u, v = template_res_i[i], template_res_j[i]
        if u == v:
            ax.scatter(x[u], y[u], z[u], c='gold', s=150, marker='o', edgecolors='black', zorder=10)
        else:
            line_color = 'red' if signs_template[u] == -1 else 'blue'
            ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], c=line_color, alpha=0.2, linewidth=1)

    # 4. Plot Input-to-Reservoir Connections
    for i in range(len(template_i)):
        u, v = template_i[i], template_j[i]
        ax.plot([in_x[u], x[v]], [in_y[u], y[v]], [in_z[u], z[v]], c='green', alpha=0.2, linewidth=1)

    # 5. Plot Output Tower (Vertical column at x = root + 1)
    # Assume num_out is the number of readout neurons you have
    out_x = np.full(num_out, root + 1)
    out_y = np.full(num_out, root / 2)
    out_z = np.linspace(0, root - 1, num_out)
    
    ax.scatter(out_x, out_y, out_z, c='orange', s=60, marker='s', label='Output')

    ax.set_title(f"Reservoir Connectivity with Input Tower")
    ax.legend()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_separation_ratio(results_matrix):
    results_matrix = np.array(results_matrix)
    # results_matrix shape: [in_dist, out_dist, label_i, label_j]
    in_dist = results_matrix[:, 0]
    out_dist = results_matrix[:, 1]
    label_i = results_matrix[:, 2]
    label_j = results_matrix[:, 3]

    plt.figure(figsize=(12, 8))

    # --- 1. Plot Intra-class pairs per class type ---
    # Find unique labels to iterate through them
    unique_labels = np.unique(np.concatenate([label_i, label_j]))
    
    # We'll use a colormap or a list of colors for different classes
    # 'cool' or 'Blues' works well to stay in the 'stability' color family
    intra_colors = ['#1f77b4', '#9467bd', '#2ca02c'] # Blue, Purple, Green
    
    for idx, val in enumerate(unique_labels):
        # Mask where both samples belong to the same specific class
        specific_intra_mask = (label_i == val) & (label_j == val)
        
        plt.scatter(in_dist[specific_intra_mask], out_dist[specific_intra_mask], 
                    alpha=1.0, c=intra_colors[idx % len(intra_colors)], 
                    label=f'Intra-class (Class {int(val)})', s=10)

    # --- 2. Plot Inter-class pairs (Cross-class separation) ---
    inter_mask = (label_i != label_j)
    plt.scatter(in_dist[inter_mask], out_dist[inter_mask], 
                alpha=1.0, c='orange', label='Inter-class (Separation)', s=10)

    # --- 3. Add Linear Regression for the overall trend ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(in_dist, out_dist)
    line = slope * in_dist + intercept
    plt.plot(in_dist, line, color='red', linewidth=2, 
             label=f'Trend: y={slope:.2f}x + {intercept:.2f}')

    # --- 4. Add the Target Line (y = x) for reference ---
    max_val = max(np.max(in_dist), np.max(out_dist))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Target Ratio (1.0)')

    # Formatting
    plt.xlabel('Input Separation: ||u_i(t) - u_j(t)||')
    plt.ylabel('Reservoir Separation: ||x_i(t) - x_j(t)||')
    plt.title('Separation Ratio Graph: Multi-Class Intra-class Analysis')
    
    # Increase alpha for legend visibility
    leg = plt.legend()
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
        
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Zone Analysis
    print(f"Calculated Slope (m): {slope:.2f}")
    print(f"Calculated Intercept (b): {intercept:.2f}")
    
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_readout_analysis(params_list):
    num_folds = len(params_list)
    # Extract weights for Class 0 (since Class 1 is just the mirror image)
    # Shape will be [num_folds, num_neurons]
    all_weights = np.array([p['fc.weight'][0].cpu().numpy() for p in params_list])
    # Shape will be [num_folds, 2]
    all_biases = np.array([p['fc.bias'].cpu().numpy() for p in params_list])
    
    num_neurons = all_weights.shape[1]
    neuron_indices = np.arange(num_neurons)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 1, 0.8]})
    plt.subplots_adjust(hspace=0.4)

    # --- 1. OVERLAY WEIGHT PROFILE (Stability View) ---
    for i in range(num_folds):
        axes[0].plot(neuron_indices, all_weights[i], alpha=0.6, label=f'Fold {i+1}')
    
    axes[0].set_title('Readout Weight Profile Across Folds', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Neuron Index (Reservoir)')
    axes[0].set_ylabel('Weight Magnitude')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- 2. MEAN WEIGHTS WITH VARIANCE (Sparsity View) ---
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)
    
    axes[1].bar(neuron_indices, mean_weights, yerr=std_weights, color='teal', alpha=0.7, capsize=2)
    axes[1].set_title('Mean Weight per Neuron (Error bars = Std Dev)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Neuron Index')
    axes[1].set_ylabel('Mean Weight')
    
    # Annotate the strongest neuron
    max_idx = np.argmax(np.abs(mean_weights))
    axes[1].annotate(f'Anchor Neuron {max_idx}\nVal: {mean_weights[max_idx]:.2f}',
                     xy=(max_idx, mean_weights[max_idx]), xytext=(max_idx+10, mean_weights[max_idx]),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    # --- 3. BIAS COMPARISON ---
    # Plotting biases as a grouped bar chart
    x_bias = np.arange(num_folds)
    width = 0.35
    axes[2].bar(x_bias - width/2, all_biases[:, 0], width, label='Bias Class 0', color='#1f77b4')
    axes[2].bar(x_bias + width/2, all_biases[:, 1], width, label='Bias Class 1', color='#ff7f0e')
    
    axes[2].set_title('Bias Distribution per Fold', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_bias)
    axes[2].set_xticklabels([f'Fold {i+1}' for i in range(num_folds)])
    axes[2].set_ylabel('Bias Value')
    axes[2].legend()
    axes[2].axhline(0, color='black', linewidth=0.8)

    plt.show()

# To use it:
# plot_readout_analysis(params_list)