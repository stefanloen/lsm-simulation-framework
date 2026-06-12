from datetime import datetime
import glob
import os
from pprint import pprint
import numpy as np
from analysis import get_spectral_radius
from interfaces import Beta, Fixed, Gaussian, MaassLIF, MarkramSyn, Neuron, RangeInfo, Record, ReservoirConfig, SampleMetadata, SimpleLI, SimpleLIF, SimpleSyn, Synapse, Uniform
import brian2 as b2

LIMODEL = """
ds/dt = (-(s - (v_rest + v_bias) / volt )) / tau : 1
v = s * volt : volt

s_exc : 1
s_inh : 1

v_rest : volt
tau : second
v_bias : volt
"""

SIMPLELIF_MODEL = """
ds/dt= (-(s-(v_rest / volt))) / tau : 1 (unless refractory)
v = s * volt : volt

s_exc : 1
s_inh : 1

refractory_t : second
threshold_v : volt
v_rest : volt
tau : second
"""

MAASSLIF_MODEL = """
dv/dt = (-(v - v_rest) + Ri * (I_bg + I_syn)) / tau : volt (unless refractory)  

ds_exc/dt = -s_exc / tau_syn_exc : 1 
ds_inh/dt = -s_inh / tau_syn_inh : 1

I_exc = s_exc * amp : amp
I_inh = s_inh * amp : amp

I_syn = I_exc + I_inh : amp

s: 1
reset_v : volt
v_rest : volt
threshold_v : volt
tau : second
tau_syn_exc : second
tau_syn_inh : second
refractory_t : second
Ri : ohm
I_bg : amp
"""

SIMPLESYN_MODEL = """
w: 1
"""

SIMPLESYN_ON_PRE = """
s_post += w / 30e-9 * 4e-3
s_exc_post += w * int(w > 0)
s_inh_post += w * int(w < 0)
"""

MARKRAMSYN_MODEL = """
dx/dt = (1-x)/tau_d : 1 (event-driven)
du/dt = -u/tau_f : 1 (event-driven)

w : 1               
U_param : 1         
tau_d : second       
tau_f : second      
"""

MARKRAM_ON_PRE = """
s_post += (w * x * u) / 30e-9 * 4e-3
s_exc_post += (w * x * u) * int(w > 0)
s_inh_post += (w * x * u) * int(w < 0)

x -= x * u
u += U_param * (1 - u)
"""

def get_synapsegroup(
    source: b2.Group,
    source_sign: np.ndarray,
    target: b2.Group,
    target_sign: np.ndarray,
    ij_template: np.ndarray,
    num_par: int,
    syn_conf: Synapse,
    local_rng: np.random.RandomState,
    noise_rng: np.random.RandomState
) -> b2.Synapses:
    source_N = source.N // num_par
    target_N = target.N // num_par

    match syn_conf.EE, syn_conf.EI, syn_conf.IE, syn_conf.II:
        case (SimpleSyn() as ee, SimpleSyn() as ei, SimpleSyn() as ie, SimpleSyn() as ii):
            W_vals = np.zeros((source_N, target_N))
            delay_vals = np.zeros((source_N, target_N))
            
            source_sign_2d = source_sign[:, np.newaxis]
            target_sign_2d = target_sign[np.newaxis,:]

            mask_ee = (source_sign_2d == 1)  & (target_sign_2d == 1)
            mask_ei = (source_sign_2d == 1)  & (target_sign_2d == -1)
            mask_ie = (source_sign_2d == -1) & (target_sign_2d == 1)
            mask_ii = (source_sign_2d == -1) & (target_sign_2d == -1)

            W_vals[mask_ee] = sample_dist(ee.w, local_rng, size=np.sum(mask_ee))
            W_vals[mask_ei] = sample_dist(ei.w, local_rng, size=np.sum(mask_ei))
            W_vals[mask_ie] = sample_dist(ie.w, local_rng, size=np.sum(mask_ie))
            W_vals[mask_ii] = sample_dist(ii.w, local_rng, size=np.sum(mask_ii))

            W_vals = add_nonideal(
                W_vals, 
                levels=syn_conf.levels, 
                write_noise=syn_conf.write_noise, 
                rng=noise_rng
            )

            delay_vals[mask_ee] = ee.syn_delay
            delay_vals[mask_ei] = ei.syn_delay
            delay_vals[mask_ie] = ie.syn_delay
            delay_vals[mask_ii] = ii.syn_delay

            # current_spectral_radius = get_spectral_radius(template_res_i, template_res_j, W_vals, N_res)
            # # W_vals *= (config.spectral_radius / current_spectral_radius)
            # # new_spectral_radius = get_spectral_radius(template_res_i, template_res_j, W_vals, N_res)

            # # print(f"SR was: {current_spectral_radius}, now: {new_spectral_radius}")
            # print(f"Mean abs weights: {np.mean(np.abs(W_vals))}")
            # print(f"Max weight: {np.max(W_vals)}")

            synapses = b2.Synapses(
                source, target, 
                model=SIMPLESYN_MODEL, 
                on_pre=SIMPLESYN_ON_PRE,
            )

            i_offsets = np.arange(num_par) * source_N
            j_offsets = np.arange(num_par) * target_N

            template_i = ij_template[:, 0]
            template_j = ij_template[:, 1]

            all_res_i = (template_i[:, np.newaxis] + i_offsets).flatten(order='F')
            all_res_j = (template_j[:, np.newaxis] + j_offsets).flatten(order='F')
            
            synapses.connect(i=all_res_i, j=all_res_j)
            synapses.w = np.tile(W_vals[template_i, template_j], num_par)
            synapses.delay =  np.tile(delay_vals[template_i, template_j], num_par) * b2.second

            return synapses

        case (MarkramSyn() as ee, MarkramSyn() as ei, MarkramSyn() as ie, MarkramSyn() as ii):
            W_vals = np.zeros((source_N, target_N))
            delay_vals = np.zeros((source_N, target_N))
            U_vals = np.zeros((source_N, target_N))
            T_rec_vals = np.zeros((source_N, target_N))
            T_fac_vals = np.zeros((source_N, target_N))
            
            source_sign_2d = source_sign[:, np.newaxis]
            target_sign_2d = target_sign[np.newaxis, :]

            mask_ee = (source_sign_2d == 1)  & (target_sign_2d == 1)
            mask_ei = (source_sign_2d == 1)  & (target_sign_2d == -1)
            mask_ie = (source_sign_2d == -1) & (target_sign_2d == 1)
            mask_ii = (source_sign_2d == -1) & (target_sign_2d == -1)

            # Sample Weights (A)
            W_vals[mask_ee] = sample_dist(ee.A, local_rng, size=np.sum(mask_ee))
            W_vals[mask_ei] = sample_dist(ei.A, local_rng, size=np.sum(mask_ei))
            W_vals[mask_ie] = sample_dist(ie.A, local_rng, size=np.sum(mask_ie))
            W_vals[mask_ii] = sample_dist(ii.A, local_rng, size=np.sum(mask_ii))

            W_vals = add_nonideal(
                W_vals, 
                levels=syn_conf.levels, 
                write_noise=syn_conf.write_noise, 
                rng=noise_rng
            )

            # Sample Utilization (U)
            U_vals[mask_ee] = sample_dist(ee.util, local_rng, size=np.sum(mask_ee))
            U_vals[mask_ei] = sample_dist(ei.util, local_rng, size=np.sum(mask_ei))
            U_vals[mask_ie] = sample_dist(ie.util, local_rng, size=np.sum(mask_ie))
            U_vals[mask_ii] = sample_dist(ii.util, local_rng, size=np.sum(mask_ii))

            # Sample Tau Recovery
            T_rec_vals[mask_ee] = sample_dist(ee.tau_rec, local_rng, size=np.sum(mask_ee))
            T_rec_vals[mask_ei] = sample_dist(ei.tau_rec, local_rng, size=np.sum(mask_ei))
            T_rec_vals[mask_ie] = sample_dist(ie.tau_rec, local_rng, size=np.sum(mask_ie))
            T_rec_vals[mask_ii] = sample_dist(ii.tau_rec, local_rng, size=np.sum(mask_ii))

            # Sample Tau Facilitation
            T_fac_vals[mask_ee] = sample_dist(ee.tau_fac, local_rng, size=np.sum(mask_ee))
            T_fac_vals[mask_ei] = sample_dist(ei.tau_fac, local_rng, size=np.sum(mask_ei))
            T_fac_vals[mask_ie] = sample_dist(ie.tau_fac, local_rng, size=np.sum(mask_ie))
            T_fac_vals[mask_ii] = sample_dist(ii.tau_fac, local_rng, size=np.sum(mask_ii))

            # Delays
            delay_vals[mask_ee] = ee.syn_delay
            delay_vals[mask_ei] = ei.syn_delay
            delay_vals[mask_ie] = ie.syn_delay
            delay_vals[mask_ii] = ii.syn_delay

            synapses = b2.Synapses(
                source, target, 
                model=MARKRAMSYN_MODEL, 
                on_pre=MARKRAM_ON_PRE,
            )

            i_offsets = np.arange(num_par) * source_N
            j_offsets = np.arange(num_par) * target_N

            template_i = ij_template[:, 0]
            template_j = ij_template[:, 1]

            all_i = (template_i[:, np.newaxis] + i_offsets).flatten(order='F')
            all_j = (template_j[:, np.newaxis] + j_offsets).flatten(order='F')
            
            synapses.connect(i=all_i, j=all_j)
            
            # Map and Tile variables
            synapses.w = np.tile(W_vals[template_i, template_j], num_par)
            synapses.delay = np.tile(delay_vals[template_i, template_j], num_par) * b2.second
            synapses.U_param = np.tile(U_vals[template_i, template_j], num_par)
            synapses.tau_d = np.tile(T_rec_vals[template_i, template_j], num_par) * b2.second
            synapses.tau_f = np.tile(T_fac_vals[template_i, template_j], num_par) * b2.second
            
            # Initialize states
            synapses.x = 1.0
            synapses.u = 0.0
            return synapses
        
        case _:
            raise ValueError(f"Unsupported synapse type ")

def get_neurongroup(
    N: int,
    num_par: int,
    sign: np.ndarray,
    neuron_conf: Neuron
) -> b2.NeuronGroup:
    
    total_N = N*num_par
    full_signs = np.tile(sign, num_par)

    match neuron_conf.E, neuron_conf.I:
        case SimpleLIF() as E, SimpleLIF() as I:
        
            group = b2.NeuronGroup(
                total_N, 
                model=SIMPLELIF_MODEL, 
                threshold='v > threshold_v', 
                reset='s = v_rest / volt', 
                refractory='refractory_t', #type: ignore
                method='euler'
            )

            rest_vals = np.where(full_signs == 1, E.v_rest, I.v_rest)
            group.v_rest = rest_vals * b2.volt
            group.s = rest_vals
            
            thresh_vals = np.where(full_signs == 1, E.threshold_v, I.threshold_v)
            group.threshold_v = thresh_vals * b2.volt
            
            tau_vals = np.where(full_signs == 1, E.leak_tau, I.leak_tau)
            group.tau = tau_vals * b2.second
            
            ref_vals = np.where(full_signs == 1, E.refractory_period, I.refractory_period)
            group.refractory_t = ref_vals * b2.second

            return group
        
        case MaassLIF() as E, MaassLIF() as I:
            group = b2.NeuronGroup(
                total_N, 
                model=MAASSLIF_MODEL, 
                threshold='v > threshold_v', 
                reset='v = reset_v', 
                refractory='refractory_t', #type: ignore
                method='euler'
            )
            
            I_bg_vals = np.where(full_signs == 1, E.background_I, I.background_I)
            group.I_bg = I_bg_vals * b2.amp

            Ri_vals = np.where(full_signs == 1, E.input_resistance, I.input_resistance)
            group.Ri = Ri_vals * b2.ohm

            rest_vals = np.where(full_signs == 1, E.v_rest, I.v_rest)
            group.v_rest = rest_vals * b2.volt

            tau_vals = np.where(full_signs == 1, E.leak_tau, I.leak_tau)
            group.tau = tau_vals * b2.second

            tau_syn_exc_vals = np.where(full_signs == 1, E.tau_syn_exc, I.tau_syn_exc)
            group.tau_syn_exc = tau_syn_exc_vals * b2.second

            tau_syn_inh_vals = np.where(full_signs == 1, E.tau_syn_inh, I.tau_syn_inh)
            group.tau_syn_inh = tau_syn_inh_vals * b2.second

            thresh_vals = np.where(full_signs == 1, E.threshold_v, I.threshold_v)
            group.threshold_v = thresh_vals * b2.volt
            
            reset_vals = np.where(full_signs == 1, E.reset_v, I.reset_v)
            group.reset_v = reset_vals * b2.volt
            group.v = reset_vals * b2.volt
            
            ref_vals = np.where(full_signs == 1, E.refractory_period, I.refractory_period)
            group.refractory_t = ref_vals * b2.second
            
            return group

        case SimpleLI() as E, SimpleLI() as I:
            group = b2.NeuronGroup(
                total_N,
                model=LIMODEL,
                method='euler'
            )
                
            rest_vals = np.where(full_signs == 1, E.v_rest, I.v_rest)
            group.v_rest = rest_vals * b2.volt
            group.s = rest_vals
            
            tau_vals = np.where(full_signs == 1, E.leak_tau, I.leak_tau)
            group.tau = tau_vals * b2.second
            
            return group

        case _:
            raise ValueError(f"Unsupported neuron configuration type: {type(neuron_conf)}")


def sample_dist(dist, rng, size=1):
    match dist:
        case Fixed(value):
            return np.full(size, value)
        
        case Uniform(low, high):
            return rng.uniform(low, high, size=size)
        
        case Gaussian(mean, std):
            return rng.normal(mean, std, size=size)
            
        case Beta(mean, max_dev):
            return (rng.beta(2, 2, size=size) - 0.5) * (2 * max_dev) + mean

        case _:
            raise ValueError(f"Unknown distribution type: {type(dist)}")
        
def add_nonideal(W_vals: np.ndarray, levels: int, write_noise: float, rng: np.random.RandomState) -> np.ndarray:
    if W_vals.size == 0:
        return W_vals

    w_min, w_max = np.min(W_vals), np.max(W_vals)
    w_range = w_max - w_min
    
    if w_range == 0:
        return W_vals

    if levels > 0:
        W_norm = (W_vals - w_min) / w_range
        W_vals = np.round(W_norm * (levels - 1)) / (levels - 1)
        W_vals = W_vals * w_range + w_min

    if write_noise > 0:
        # write_noise is treated as a percentage of the total weight range
        # e.g., 0.05 = 5% of the range as standard deviation
        sigma = write_noise * w_range
        noise = rng.normal(0, sigma, size=W_vals.shape)
        W_vals += noise

    return W_vals