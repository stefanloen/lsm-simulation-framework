import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import copy
from multiprocessing import Pool
import os
from pprint import pprint
import shutil

import numpy as np
import optuna
from analysis import get_performance_metrics
from cfgs import cfg_dataonly
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from framework import run
from interfaces import ClassNumber, EEGPhase, Fixed, Gaussian, MaassLIF, MarkramSyn, NetworkLocation, Neuron, ReservoirConfig, SimpleLIF, SimpleSyn, Split, Synapse, TaskType, Uniform
from plotting import plot_readout_analysis, plot_record
from train import split_records, split_records_labelled    


# --- 2. DATA PROCESSING FUNCTIONS ---

def exponential_filter(spike_tuple, n_channels, total_duration, dt, tau=0.300):
    """
    Converts (times, indices) into a dense (time_steps, channels) trace
    using an exponential decay kernel.
    """
    t_sorted, i_sorted = spike_tuple
    time_steps = int(total_duration / dt)
    # Initialize dense feature matrix (Time, Channels)
    trace = np.zeros((time_steps, n_channels))
    
    # Convert spike times to bin indices
    spike_bins = (t_sorted / dt).astype(int)
    
    # Simulation loop for exponential decay
    # We iterate through channels to apply the decay efficiently
    for ch in range(n_channels):
        ch_mask = (i_sorted == ch)
        ch_spike_bins = spike_bins[ch_mask]
        
        # Only process if there are spikes in this channel
        if len(ch_spike_bins) > 0:
            current_v = 0.0
            decay = np.exp(-dt / tau)
            
            # Walk through time
            for t in range(time_steps):
                current_v *= decay
                if t in ch_spike_bins:
                    # Increment for every spike in this bin
                    current_v += np.sum(ch_spike_bins == t)
                trace[t, ch] = current_v
                
    return trace

def create_windows_and_filter(record_list, window_size=50, keep_labels=[0.0, 2.0], use_spikes = False, dt = 0.01, tau = 0.3):
    """
    Processes a list of records: Windows first, then filters by target label.
    Maps labels to 0 and 1.
    """
    X_list, y_list = [], []
    
    for rec in record_list:
        if not use_spikes:
            features = rec.input_data # Expected (Time, Channels)
        else:
            # Unpack the tuple: ( (times, indices), channel_count )
            spike_data, n_channels = rec.encoder_spikes
            # Calculate duration from labels if not explicitly in record
            dt = 0.01
            duration = len(rec.label) * dt
            
            features = exponential_filter(spike_data, n_channels, duration, dt)

        # Assuming label is 2D, we take the target column
        labels = rec.label[:, 1]  
        
        for i in range(len(labels) - window_size + 1):
            target_label = labels[i + window_size - 1]
            
            if target_label in keep_labels:
                X_list.append(features[i : i + window_size])
                # Map: First in keep_labels -> 0, Second -> 1
                y_list.append(0 if target_label == keep_labels[0] else 1)
                
    return np.array(X_list), np.array(y_list)

import numpy as np

from scipy.signal import lfilter

def create_windows_and_filter_no_leak(record_list, window_size=50, keep_labels=[0.0, 2.0], use_spikes = True, dt=0.01, tau=0.3):
    X_list, y_list = [], []
    
    # Define the IIR filter coefficients for exponential decay:
    # y[t] = x[t] + decay * y[t-1]
    decay = np.exp(-dt / tau)
    b = [1.0]       # Numerator
    a = [1.0, -decay] # Denominator (this creates the recursive decay)

    for rec in record_list:
        (t_sorted, i_sorted), n_channels = rec.encoder_spikes
        labels = rec.label[:, 1]
        
        for i in range(len(labels) - window_size + 1):
            target_label = labels[i + window_size - 1]
            
            if target_label in keep_labels:
                window_start_t = i * dt
                window_end_t = (i + window_size) * dt
                
                # 1. Grab spikes in window
                mask = (t_sorted >= window_start_t) & (t_sorted < window_end_t)
                win_times = t_sorted[mask] - window_start_t 
                win_indices = i_sorted[mask]
                
                # 2. Create binary spike counts (the input 'x')
                counts_grid = np.zeros((window_size, n_channels))
                bin_idx = (win_times / dt).astype(int)
                bin_idx = np.clip(bin_idx, 0, window_size - 1)
                
                for ch in range(n_channels):
                    ch_bins = bin_idx[win_indices == ch]
                    if ch_bins.size > 0:
                        counts_grid[:, ch] = np.bincount(ch_bins, minlength=window_size)

                window_features = lfilter(b, a, counts_grid, axis=0)
                
                X_list.append(window_features)
                y_list.append(0 if target_label == keep_labels[0] else 1)
                
    return np.array(X_list), np.array(y_list)

def scale_data(X_train, X_val):
    """ Scales 3D data using a 2D scaler fit on training data only. """
    num_train, steps, feats = X_train.shape
    num_val = X_val.shape[0]
    
    scaler = StandardScaler()
    # Fit on training features flattened
    X_train_reshaped = X_train.reshape(-1, feats)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_train, steps, feats)
    
    # Transform validation features using training scaler
    X_val_reshaped = X_val.reshape(-1, feats)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(num_val, steps, feats)
    
    return X_train_scaled, X_val_scaled

# --- 3. MODEL ARCHITECTURE ---

def build_binary_1d_cnn(window_size, num_features):
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.Conv1D(32, kernel_size=3, dilation_rate=5, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def build_simple_1d_cnn(window_size, num_features):
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.Conv1D(8, kernel_size=7, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),  # Reduced dropout for a smaller network
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    WINDOW_SIZE = 200
    NUM_FEATURES = 18
    KEEP_LABELS = [0.0, 2.0]  # Binary Classification targets

    # --- 4. EXECUTION FLOW ---
    records, global_record = run(
    start_loc=NetworkLocation.DATA, 
    end_loc=NetworkLocation.PREPROCESSOR_OUT, 
    train_filter={EEGPhase.INTERICTAL: 140, EEGPhase.ICTAL: 140, 'patients': [i for i in range(1, 25) if i != 12]} ,
    validate_filter=None,
    max_processes=5,
    config=cfg_dataonly.config,
    noise_seed=12,
    brian_dir='../tmp/main/brian',
    cache_dir='../tmp/main/cache')



    # A. Split records based on metadata Enum (Split.TRAIN=0, Split.VALIDATE=1)
    record_labels = [r.sample_metadata.metadata['eegphase'].value for r in records]

    # 2. Define your own split (e.g., 80% Train, 20% Validation)
    # random_state ensures your results are reproducible
    train_recs, val_recs = train_test_split(
        records, 
        test_size=0.20, 
        stratify=record_labels, 
        random_state=42
    )


    print("train:")
    print(f"len: {len(train_recs)}")
    for rec in train_recs:
        print(rec.sample_metadata.sample_id)

    print("validate:")
    print(f"len: {len(val_recs)}")
    for rec in val_recs:
        print(rec.sample_metadata.sample_id)

    # B. Window and Filter (Leak-free: split happens BEFORE windowing)
    X_train_raw, y_train = create_windows_and_filter(train_recs, WINDOW_SIZE, KEEP_LABELS)
    X_val_raw, y_val = create_windows_and_filter(val_recs, WINDOW_SIZE, KEEP_LABELS)

    # C. Scale features
    X_train, X_val = scale_data(X_train_raw, X_val_raw)

    # D. Shuffle Training Data (Validation order doesn't matter)
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"Final Dataset: Train={X_train.shape}, Val={X_val.shape}")

    # E. Build and Train
    model = build_binary_1d_cnn(WINDOW_SIZE, NUM_FEATURES)

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    )

    print("Starting training on NVIDIA GPU...")
    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
    )

    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1

    print("\n" + "="*30)
    print(f"HIGHEST VALIDATION ACCURACY: {best_val_acc * 100:.2f}%")
    print(f"ACHIEVED AT EPOCH: {best_epoch}")
    print("="*30)

    # --- 5. VISUALIZATION ---

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Binary Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Binary Cross-Entropy Loss')
    plt.legend()
    plt.show()