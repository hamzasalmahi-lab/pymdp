import numpy as np
from pymdp import utils
from pymdp.agent import Agent

# --- STEP A: DEFINE THE STATES ---
num_states = [2] 
num_obs = [2] 

# --- STEP B: THE 'A' MATRIX ---
A = utils.random_A_matrix(num_obs, num_states)
A[0][0, 0] = 0.9  
A[0][1, 0] = 0.1
A[0][0, 1] = 0.3  
A[0][1, 1] = 0.7

# --- STEP C: THE 'C' MATRIX ---
# Increased precision: stronger preference for stable state to spike free energy harder on mismatches
C = utils.obj_array(1)
C[0] = np.array([4.0, -4.0])

# --- STEP B2: THE 'B' MATRIX ---
B = utils.random_B_matrix(num_states, [1])
B[0][:, :, 0] = np.eye(2)

# --- STEP D: INITIALIZE THE AGENT ---
# Lower learning rate (lr_pA=0.01) increases 'Surprise' for unstable events, preventing over-adaptation to pathological noise
my_hrit_agent = Agent(A=A, B=B, C=C, lr_pA=0.01)

# --- ADD VPM BUFFER AND RSI PROPERTY ---
import collections
my_hrit_agent.vpm_buffer = collections.deque(maxlen=30)

@property
def rsi(my_hrit_agent):
    """Calculate Relative Stability Index as exp(-VFE)"""
    # Assuming latest VFE is available; in pymdp, this might need adjustment
    try:
        vfe = my_hrit_agent.F  # Free energy from last inference
        return np.exp(-vfe)
    except AttributeError:
        return 1.0  # Default if not available

my_hrit_agent.rsi = rsi

def calculate_vpm(my_hrit_agent):
    """Calculate VPM (Volatility of Manifold Precision) as std dev of k-score buffer"""
    if len(my_hrit_agent.vpm_buffer) < 2:
        return 0.0
    return np.std(list(my_hrit_agent.vpm_buffer))

my_hrit_agent.calculate_vpm = calculate_vpm

# --- NEW STEP: THE ENGINE INTERFACE ---
# This is the 'Glue' that main_engine.py needs to work!
def process_neural_signal(observation_index, agent):
    """
    Takes a 0 or 1 from the sensor and returns the agent's belief.
    """
    # Wrap the observation in a list as pymdp expects
    observation = [observation_index]
    
    # The Agent 'Infers' the hidden state (Stable vs Unstable)
    qs = agent.infer_states(observation)
    
    return qs[0] # Returns the probability array [Presence, Snap]

# Optional: Keep your print statement for local testing
if __name__ == "__main__":
    print("HRIT Agent initialized and ready for engine integration.")
