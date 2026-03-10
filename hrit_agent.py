import numpy as np
from pymdp import utils
from pymdp.agent import Agent

# --- STEP A: DEFINE THE STATES ---
# Our brain has 2 possible internal states: 0 = Stable (Presence), 1 = Unstable (The Snap)
num_states = [2] 

# Our brain can see 2 things: 0 = High Precision Signal, 1 = Noisy Signal
num_obs = [2] 

# --- STEP B: THE 'A' MATRIX (Likelihood) ---
# This tells the AI: "If I am Stable, I usually see High Precision signals."
A = utils.random_A_matrix(num_obs, num_states)
A[0][0, 0] = 0.9  # Stable state -> 90% chance of High Precision
A[0][1, 0] = 0.1
A[0][0, 1] = 0.3  # Unstable state -> 70% chance of Noisy Signal
A[0][1, 1] = 0.7

# --- STEP C: THE 'C' MATRIX (Preferences) ---
# This is your HRIT 'Homeostatic Drive' (Phi).
# We tell the AI: "You strongly prefer being in a High Precision state."
C = utils.obj_array(1)
C[0] = np.array([2.0, -2.0]) # High reward for state 0, Penalty for state 1

# --- STEP D: INITIALIZE THE AGENT ---
my_hrit_agent = Agent(A=A, C=C)

# --- STEP E: RUN A MINI-SIMULATION ---
print("--- Starting HRIT Homeostatic Simulation ---")
# Let's pretend the brain receives a 'Noisy Signal' (Observation index 1)
observation = [1] 

# The Agent 'Infers' what is happening to its manifold
qs = my_hrit_agent.infer_states(observation)

print(f"Observation Received: {'Noisy Signal' if observation[0]==1 else 'Clear Signal'}")
print(f"Agent's Belief (Stable vs Unstable): {qs[0].round(2)}")

if qs[0][1] > 0.5:
    print("DIAGNOSIS: The agent senses a drop in Phi. Manifold is flattening!")
else:
    print("DIAGNOSIS: Agent maintains Waking Presence.")
