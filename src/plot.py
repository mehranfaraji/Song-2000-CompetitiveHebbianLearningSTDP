# import numpy as np
# import matplotlib.pyplot as plt
from brian2 import *
prefs.codegen.target = "numpy"

def plot_experiment(file_name: str, display: bool, params: dict) -> None:
    """
    Plot the results of the experiment.

    Parameters:
    file_name (str): The name of the file to save the plot.
    display (bool): Whether to display the plot.
    params (dict): Dictionary containing the parameters for the experiment.
        - g_max_ex (float): Maximum conductance for excitatory neurons.
        - g_max_in (float): Maximum conductance for inhibitory neurons.
        - A_pre (float): Amplitude of presynaptic change.
        - A_post (float): Amplitude of postsynaptic change.
        - episode_duration (float): Duration of each episode in the experiment.
        - num_repeat (int): Number of times the experiment is repeated.
        - tau_pre (float): Time constant of presynaptic change.
        - tau_post (float): Time constant of postsynaptic change.
        - B (float): *** (Please specify what B represents)
        - post_spike_mon (SpikeMonitor): Spike monitor for the postsynaptic neurons.
        - post_state_mon (StateMonitor): State monitor for the postsynaptic neuron.
        - input_stimuli_monitor (SpikeMonitor): Spike monitor for the input stimuli.
        - first_spike_time (np.ndarray): Array of first spike times for each neuron.
        - S (Synapses): Synapses object.
        - excitatory_idx (np.ndarray): Array indicating which neurons are excitatory.

    Returns:
    None
    """
    g_max_ex = params['g_max_ex']
    g_max_in = params['g_max_in']
    A_pre = params['A_pre']
    A_post = params['A_post']
    episode_duration = params['episode_duration']
    num_repeat = params['num_repeat']
    tau_pre = params['tau_pre']
    # tau_post = params['tau_post']
    B = params['B']
    post_spike_mon = params['post_spike_mon']
    post_state_mon = params['post_state_mon']
    input_stimuli_monitor = params['input_stimuli_monitor']
    first_spike_time = params['first_spike_time']
    S = params['S']
    excitatory_idx = params['excitatory_idx']
    
    episode_span = int(episode_duration/ms*10)
    num_spikes_last_episode = len(post_spike_mon.t[post_spike_mon.t/ms > post_state_mon.t[-episode_span]/ms])
    num_spikes_first_episode = len(post_spike_mon.t[post_spike_mon.t/ms < post_state_mon.t[episode_span]/ms])
    description =  f'''num_repeat={num_repeat}, g_max_ex={g_max_ex}, g_max_in={g_max_in}, taupre=taupost={tau_pre}, A_pre={A_pre:.5f}, A_post={A_post:.5f}
B = {B}, number of spikes during first episode = {num_spikes_first_episode}, number of spikes during last episode = {num_spikes_last_episode}'''

    f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(9, 6), height_ratios=[0.7, 0.3])

    # only_excitatory = np.where(excitatory_idx != 0)[0]
    only_excitatory = where(excitatory_idx != 0)[0]
    
    a0.plot(input_stimuli_monitor.t/ms, input_stimuli_monitor.i, '.k')
    a0.set_xlabel('Time (ms)')
    a0.set_ylabel('Neuron index')
    a0.set_title('Input stimuli')

    a1.plot(first_spike_time[only_excitatory], S.g_a[only_excitatory]/g_max_ex , '.k')
    a1.set_xlabel('First spike time [relative latency] (ms)')
    a1.set_ylabel(r'g / $g_{max}$')
    a1.set_title(r'g/$g_{max}$ ratio as a function of relative latency')
    a1.set_ylim([-0.1, 1.05])
    a1.set_xlim([-3, None])

    w_init = array([g_max_ex * 0.2] * len(first_spike_time[only_excitatory]))
    a2.plot(first_spike_time[only_excitatory], w_init / g_max_ex, '.k')
    a2.set_xlabel('First spike time [relative latency] (ms)')
    a2.set_ylabel(r'g / $g_max$')
    a2.set_title(r'initial g/$g_{max}$ ratio as a function of relative latency')
    a2.set_ylim([0, 1])
    a2.set_xlim([-3, None])

    a3.plot(S.i[only_excitatory], S.g_a[only_excitatory]/g_max_ex, '.k')
    a3.set_xlabel('Synapse index')
    a3.set_ylabel(r'g / $g_max$')
    a3.set_title(r'final g/$g_{max}$ ratio of each neuron')
    a3.set_ylim([-0.1, 1.05])

    plt.figtext(0.5, 1, description, ha='center', va='top')
    plt.tight_layout(pad=2.5)
    if file_name:
        plt.savefig(file_name)
    if not display:
        plt.close()