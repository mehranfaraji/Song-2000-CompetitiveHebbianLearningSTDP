from brian2 import *
prefs.codegen.target = "numpy"

# The `generate_input` function uses code adapted from the [Neuro4ML](https://github.com/neuro4ml/neuro4ml.github.io/blob/main/materials/w4/w4-figures.ipynb) course
#  to produce input stimuli. While this implementation slightly differs from the method described in the paper, it does not affect the final results of the experiment.

def generate_input(N: int, groupsize: int, num_repeat: int, episode_duration: float) -> tuple:
    """
    Generate input for the neural network experiment.

    Parameters:
    N (int): Total number of neurons.
    groupsize (int): Size of each group of neurons.
    num_repeat (int): Number of times the experiment is repeated.
    episode_duration (float): Duration of each episode in the experiment.

    Returns:
    tuple: A tuple containing:
        - pre_synapse (SpikeGeneratorGroup): The presynaptic neuron group.
        - first_spike_time (np.ndarray): Array of first spike times for each neuron.
        - pre_spike_mon (SpikeMonitor): Spike monitor for the presynaptic neurons.
    """
    start_scope()
    eqs = '''
    starttime : second
    rate = int(t > starttime and t < starttime + 20*ms) * 100*Hz : Hz
    '''
    pre_synapse = NeuronGroup(N, eqs, threshold='rand() < rate * dt')
    pre_synapse.starttime = repeat(rand(N // groupsize) * 30, groupsize) * ms
    pre_spike_mon = SpikeMonitor(pre_synapse)

    net = Network(pre_synapse, pre_spike_mon)
    net.run(60 * ms)

    first_spike_time = array([pre_spike_mon.spike_trains()[i][0] / ms if len(pre_spike_mon.spike_trains()[i]) != 0 else -10 for i in range(N)])

    spikes_i, spikes_t = pre_spike_mon.i, pre_spike_mon.t
    times_added = [(i) * episode_duration for i in range(num_repeat)]
    time_stack = [spikes_t + t for t in times_added]
    spike_times = np.concatenate(time_stack, axis=0) * second
    spike_idx = np.tile(spikes_i, reps=num_repeat)
    pre_synapse = SpikeGeneratorGroup(N, spike_idx, spike_times)

    return pre_synapse, first_spike_time, pre_spike_mon