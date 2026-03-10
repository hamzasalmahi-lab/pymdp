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
C = utils.obj_array(1)
C[0] = np.array([2.0, -2.0])

# --- STEP D: INITIALIZE THE AGENT ---
my_hrit_agent = Agent(A=A, C=C)

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
