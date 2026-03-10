from hrit_agent import process_neural_signal, my_hrit_agent
from sensor_interface import get_observation_from_eeg
from manifold_geometry import calculate_k_score
import time

# 1. Mock 'Live' EEG Data (Just random numbers for now)
mock_eeg_stream = [np.random.random(10) for _ in range(20)]

print("--- HRIT-GIFT ENGINE STARTING ---")

for chunk in mock_eeg_stream:
    # STEP 1: SENSOR (Eyes)
    obs = get_observation_from_eeg(chunk)
    
    # STEP 2: AGENT (Brain)
    belief = process_neural_signal(obs, my_hrit_agent)
    
    # STEP 3: GEOMETRY (Map)
    k_score = calculate_k_score(belief)
    
    print(f"Neural Input: {obs} | Belief: {belief.round(2)} | GIFT-Score: {k_score:.2f}")
    time.sleep(0.5)
