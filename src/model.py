from brian2 import *
import numpy as np
prefs.codegen.target = "numpy"

def init_g(N: int, excitatory_ratio: float, g_max_ex: float, g_max_in: float) -> tuple:
    """
    Initialize conductance values for excitatory and inhibitory neurons.

    Parameters:
    N (int): Total number of neurons.
    excitatory_ratio (float): Ratio of excitatory neurons to total neurons.
    g_max_ex (float): Maximum conductance for excitatory neurons.
    g_max_in (float): Maximum conductance for inhibitory neurons.

    Returns:
    tuple: A tuple containing:
        - excitatory_idx (np.ndarray): Array indicating which neurons are excitatory.
        - g_a_init (np.ndarray): Initial conductance values for all neurons.
        - g_max_ex_init (np.ndarray): Maximum conductance values for excitatory neurons.
        - g_max_in_init (np.ndarray): Maximum conductance values for inhibitory neurons.
    """
    excitatory_count = int(N * excitatory_ratio)
    inhibitory_count = N - excitatory_count
    excitatory_idx = np.array([1] * excitatory_count + [0] * inhibitory_count)
    np.random.shuffle(excitatory_idx)

    g_a_init = np.zeros(N)
    g_a_init[excitatory_idx != 0] = g_max_ex * 0.2

    g_max_ex_init = np.zeros(N)
    g_max_ex_init[excitatory_idx != 0] = g_max_ex

    g_max_in_init = np.zeros(N)
    g_max_in_init[excitatory_idx == 0] = g_max_in
    
    return excitatory_idx, g_a_init, g_max_ex_init, g_max_in_init 


def run_experiment(pre_synapse: NeuronGroup, params: dict, num_repeat: int, episode_duration: float) -> tuple:
    """
    Run the experiment with the given parameters.

    Parameters:
    pre_synapse (NeuronGroup): The presynaptic neuron group.
    params (dict): Dictionary containing the parameters for the experiment.
        - N (int): Total number of neurons.
        - excitatory_ratio (float): Ratio of excitatory neurons to total neurons.
        - tau_m (float): Membrane time constant.
        - V_rest (float): Resting membrane potential.
        - E_ex (float): Excitatory reversal potential.
        - tau_ex (float): Excitatory synaptic time constant.
        - g_max_ex (float): Maximum conductance for excitatory neurons.
        - tau_in (float): Inhibitory synaptic time constant.
        - E_in (float): Inhibitory reversal potential.
        - g_max_in (float): Maximum conductance for inhibitory neurons.
        - A_pre (float): Amplitude of presynaptic change.
        - tau_pre (float): Time constant of presynaptic change.
        - tau_post (float): Time constant of postsynaptic change.
        - A_post (float): Amplitude of postsynaptic change.
    num_repeat (int): Number of times the experiment is repeated.
    episode_duration (float): Duration of each episode in the experiment.

    Returns:
    tuple: A tuple containing:
        - (pre_spike_mon, post_spike_mon) (tuple): Spike monitors for pre and post synaptic neurons.
        - S (Synapses): Synapses object.
        - post_state_mon (StateMonitor): State monitor for the postsynaptic neuron.
        - excitatory_idx (np.ndarray): Array indicating which neurons are excitatory.
    """
    N = params['N']
    excitatory_ratio = params['excitatory_ratio']
    tau_m = params['tau_m']
    V_rest = params['V_rest']
    E_ex = params['E_ex']
    tau_ex = params['tau_ex']
    g_max_ex = params['g_max_ex']
    tau_in = params['tau_in']
    E_in = params['E_in']
    g_max_in = params['g_max_in']
    A_pre = params['A_pre']
    tau_pre = params['tau_pre']
    tau_post = params['tau_post']
    A_post = params['A_post']
    V_threshold = params['V_threshold']
    V_reset = params['V_reset']

    LIF = '''dV/dt = ( V_rest - V + g_ex * (E_ex - V) + g_in * (E_in - V) ) / tau_m : volt
             d g_ex/dt = - g_ex / tau_ex : 1
             d g_in/dt = - g_in / tau_in : 1
            '''
    post_synapse = NeuronGroup(1, model=LIF, method='euler', threshold=f'V>V_threshold', reset='V=V_reset')
    post_synapse.V = V_rest
    
    synapse_eqs = '''
                g_a : 1
                g_max_ex_ : 1
                g_max_in_ : 1
                dPa/dt = -Pa/tau_pre : 1 (event-driven)
                dM/dt = -M/tau_post : 1 (event-driven)
                '''
    S = Synapses(pre_synapse, post_synapse,
                model=synapse_eqs,
                on_pre = '''
                    g_ex += g_a
                    g_in += g_max_in_
                    Pa += A_pre
                    g_a = clip(g_a + M * g_max_ex_, 0, g_max_ex_)
                    ''',
                on_post = '''
                    M += A_post
                    g_a = clip(g_a + Pa * g_max_ex_ , 0, g_max_ex_ )
                    ''',
                method='exact')
    S.connect()
    
    excitatory_idx, S.g_a, S.g_max_ex_, S.g_max_in_ = init_g(N, excitatory_ratio, g_max_ex, g_max_in)

    pre_spike_mon = SpikeMonitor(pre_synapse)
    post_spike_mon = SpikeMonitor(post_synapse)
    post_state_mon = StateMonitor(post_synapse, 'V', record=True)

    net = Network(pre_synapse, post_synapse, S, pre_spike_mon, post_spike_mon, post_state_mon)
    net.run(episode_duration * num_repeat)

    return (pre_spike_mon, post_spike_mon), S, post_state_mon, excitatory_idx