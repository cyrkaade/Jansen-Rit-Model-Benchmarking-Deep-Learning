import numpy as np
import matplotlib.pyplot as plt

def simulate_jr_model(F, stim_time, Tbase, Ttot, dt,
                      C=135.0, A=3.25, B=22, a=100.0, b=50.0, ka=0.1, kA=0.1,
                      vm=6.0, r=0.56, v0=6.0, seed=None):
    """
    Simulate a 2-node Jansen-Rit neural mass model using difference equations.
    
    Parameters:
        F : list or array
            List of stimulation frequencies (Hz) to simulate.
        stim_time : float
            Duration (in ms) during which stimulation is applied.
        Tbase : float
            Baseline period (in ms) before the stimulation onset.
        Ttot : float
            Total simulation duration (in ms).
        dt : float
            Time step (in ms) for simulation.
        C, A, B, a, b, ka, kA : float
            Model parameters for Node 1.
        vm : float
            Maximum firing rate or scaling factor.
        r : float
            Slope parameter used in the sigmoid function.
        v0 : float
            Threshold parameter in the sigmoid function.
        seed : int, optional
            Seed for random number generation (for reproducibility).
    
    Returns:
        sim_lfp : dict
            A dictionary where each key is a stimulation frequency and the value
            is a tuple (node1_output, node2_output) representing the simulated LFPs.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define coupling parameters for Node 1 and Node 2
    C11 = C
    C12 = 0.8 * C11
    C13 = 0.25 * C11
    C14 = 0.25 * C11

    C21 = C
    C22 = 0.8 * C21
    C23 = 0.25 * C21
    C24 = 0.25 * C21

    # Node 1 intrinsic parameters (using the passed values)
    A1 = A
    B1 = B
    a1 = a
    b1 = b
    ka1 = ka
    kA1 = kA

    # Node 2 parameters (hard-coded values)
    A2 = 5
    B2 = 26
    a2 = 50
    b2 = 10
    ka2 = 0.2
    kA2 = 0.4

    # External drive parameters for each node
    P1 = 50  
    P2 = 100

    # Inter-node coupling parameters
    ad = 10
    K1 = 1000
    K2 = 800

    # Determine number of simulation steps (assumes Tbase, Ttot, stim_time are given in ms)
    L = int(Ttot / dt)
    Tbase_steps = int(Tbase / dt)
    
    # Dictionary to store simulation outputs for each frequency
    sim_lfp = {}

    # Loop over each stimulation frequency
    for freq in F:
        # Create a stimulation cycle of length floor(1000/freq) samples.
        cycle_length = int(np.floor(1000 / freq))
        cycle = np.zeros(cycle_length)
        if cycle_length >= 2:
            cycle[0] = 1
            cycle[1] = -1
        else:
            cycle[0] = 1

        # Repeat the cycle for the duration of stimulation.
        # The number of cycles is floor((stim_time * freq) / 1000)
        n_cycles = int(np.floor(stim_time * freq / 1000))
        stim = np.tile(cycle, n_cycles)

        # Construct the complete input signal:
        #   - Zeros during baseline,
        #   - followed by the stimulation train,
        #   - followed by zeros to complete the simulation.
        I = np.concatenate([np.zeros(Tbase_steps),
                            stim,
                            np.zeros(L - Tbase_steps - len(stim))])
        
        # Define the input to each node.
        # For node 1: 60*I, for node 2: 0.
        Ip = np.vstack([60 * I, np.zeros_like(I)])
        Ii = r * Ip  # Scaled input for each node (element-wise)
        Is = 0.4 * Ip[0, :]  # Not used further in the simulation but computed as in Matlab
        
        # Initialize state variables for both nodes (all arrays have L samples)
        y0  = np.zeros(L)
        y1  = np.zeros(L)
        y2  = np.zeros(L)
        y3  = np.zeros(L)
        y4  = np.zeros(L)
        y5  = np.zeros(L)
        y6  = np.zeros(L)
        y7  = np.zeros(L)
        y8  = np.zeros(L)
        y9  = np.zeros(L)
        y10 = np.zeros(L)
        y11 = np.zeros(L)
        y12 = np.zeros(L)
        y13 = np.zeros(L)
        y14 = np.zeros(L)
        y15 = np.zeros(L)
        
        # Generate random noise inputs for each node (p1 for Node 1, p2 for Node 2)
        p1 = P1 + 0.1 * np.random.randn(L)
        p2 = P2 + 0.1 * np.random.randn(L)
        
        # Simulation loop using Euler's method for difference equations.
        for ii in range(1, L):
            # -------- Node 1 equations --------
            # Update y0 and its derivative y3
            y0[ii] = y0[ii-1] + y3[ii-1] * dt
            y3[ii] = y3[ii-1] + dt * (
                        A1 * a1 * (Ii[0, ii-1] + (vm / (1 + np.exp(r * (v0 - y1[ii-1] - y13[ii-1] + y2[ii-1]))))) -
                        2 * a1 * y3[ii-1] - a1**2 * y0[ii-1]
                     )
            
            # Update y1 and its derivative y4
            y1[ii] = y1[ii-1] + y4[ii-1] * dt
            y4[ii] = y4[ii-1] + dt * (
                        kA1 * A1 * ka1 * a1 * (p1[ii-1] + Ip[0, ii-1] + (C12 * vm / (1 + np.exp(r * (v0 - C11 * y0[ii-1]))))) -
                        2 * ka1 * a1 * y4[ii-1] - ka1**2 * a1**2 * y1[ii-1]
                     )
            
            # Update y13 and its derivative y15 (coupling from Node 2)
            y13[ii] = y13[ii-1] + dt * y15[ii-1]
            y15[ii] = y15[ii-1] + dt * (
                        A2 * ad * K2 * (vm / (1 + np.exp(r * (v0 - (y7[ii-1] + y12[ii-1] - y8[ii-1]))))) -
                        2 * ad * y15[ii-1] - ad**2 * y13[ii-1]
                     )
            
            # Update y2 and its derivative y5
            y2[ii] = y2[ii-1] + y5[ii-1] * dt
            y5[ii] = y5[ii-1] + dt * (
                        B1 * b1 * (Ip[0, ii-1] + (C14 * vm / (1 + np.exp(r * (v0 - C13 * y0[ii-1]))))) -
                        2 * b1 * y5[ii-1] - b1**2 * y2[ii-1]
                     )
            
            # -------- Node 2 equations --------
            # Update y6 and its derivative y9
            y6[ii] = y6[ii-1] + y9[ii-1] * dt
            y9[ii] = y9[ii-1] + dt * (
                        A2 * a2 * (Ii[1, ii-1] + (vm / (1 + np.exp(r * (v0 - y7[ii-1] - y12[ii-1] + y8[ii-1]))))) -
                        2 * a2 * y9[ii-1] - a2**2 * y6[ii-1]
                     )
            
            # Update y7 and its derivative y10
            y7[ii] = y7[ii-1] + y10[ii-1] * dt
            y10[ii] = y10[ii-1] + dt * (
                        kA2 * A2 * ka2 * a2 * (p2[ii-1] + Ip[1, ii-1] + (K1 * y12[ii-1]) + (C22 * vm / (1 + np.exp(r * (v0 - C21 * y6[ii-1]))))) -
                        2 * a2 * ka2 * y10[ii-1] - (a2 * ka2)**2 * y7[ii-1]
                     )
            
            # Update y12 and its derivative y14
            y12[ii] = y12[ii-1] + dt * y14[ii-1]
            y14[ii] = y14[ii-1] + dt * (
                        A1 * ad * (vm / (1 + np.exp(r * (v0 - (y1[ii-1] - y2[ii-1]))))) -
                        2 * ad * y14[ii-1] - ad**2 * y12[ii-1]
                     )
            
            # Update y8 and its derivative y11
            y8[ii] = y8[ii-1] + y11[ii-1] * dt
            y11[ii] = y11[ii-1] + dt * (
                        B2 * b2 * (Ip[1, ii-1] + (C24 * vm / (1 + np.exp(r * (v0 - C23 * y6[ii-1]))))) -
                        2 * b2 * y11[ii-1] - b2**2 * y8[ii-1]
                     )
        
        # Compute the local field potential outputs for each node.
        # Here we take a segment starting from (Tbase - 1000 ms) to the end.
        start_idx = max(0, Tbase_steps - int(1000/dt))
        outp1 = y1[start_idx:] + y13[start_idx:] - y2[start_idx:]
        outp2 = y7[start_idx:] + y12[start_idx:] - y8[start_idx:]
        
        # Center the outputs by removing the mean.
        outp1 = outp1 - np.mean(outp1)
        outp2 = outp2 - np.mean(outp2)
        
        # Save outputs in dictionary with the stimulation frequency as the key.
        sim_lfp[freq] = (outp1, outp2)
    
    return sim_lfp

def plot_simulation(sim_lfp, dt=1.0):
    """
    Plot the simulation outputs for each stimulation frequency.
    
    Parameters:
        sim_lfp : dict
            Dictionary with keys as stimulation frequencies and values as tuples (node1, node2).
        dt : float
            Time step (in ms) used for the simulation (for proper time axis scaling).
    """
    num_freq = len(sim_lfp)
    # Create a subplot for each frequency (2 plots per frequency: one for each node)
    fig, axes = plt.subplots(num_freq, 2, figsize=(12, 3 * num_freq))
    
    # Ensure axes is 2D even if only one frequency is simulated.
    if num_freq == 1:
        axes = np.array([axes])
    
    # Plot each frequency's results.
    for i, (freq, (outp1, outp2)) in enumerate(sim_lfp.items()):
        t1 = np.arange(len(outp1)) * dt
        t2 = np.arange(len(outp2)) * dt
        
        axes[i, 0].plot(t1, outp1, color='b')
        axes[i, 0].set_title(f'Node 1 Output (Freq = {freq} Hz)')
        axes[i, 0].set_xlabel('Time (ms)')
        axes[i, 0].set_ylabel('Amplitude')
        
        axes[i, 1].plot(t2, outp2, color='r')
        axes[i, 1].set_title(f'Node 2 Output (Freq = {freq} Hz)')
        axes[i, 1].set_xlabel('Time (ms)')
        axes[i, 1].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
"""
# Example usage:
if __name__ == '__main__':
    # Define simulation parameters
    F = [10, 20]      # List of stimulation frequencies (Hz)
    stim_time = 100   # Duration of stimulation in ms
    Tbase = 1000      # Baseline duration in ms
    Ttot = 3000       # Total simulation time in ms
    dt = 1            # Time step in ms

    # Run the simulation
    sim_lfp = simulate_jr_model(F, stim_time, Tbase, Ttot, dt, seed=42)
    
    # Plot the results
    plot_simulation(sim_lfp, dt=dt)

"""

