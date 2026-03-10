import numpy as np
import time
from hrit_agent import my_hrit_agent  # This imports the agent you built

# --- CONFIGURATION ---
VPM_THRESHOLD = 0.05  # How much 'shaking' we allow before the alarm
BUFFER_SIZE = 5       # We look at the last 5 seconds of data
k_history = []        # This stores our GIFT-Scores for VPM analysis

def calculate_k_score(qs):
    """
    GIFT Field Equation: Turns the agent's belief into Curvature (k).
    Certainty (Presence) = High k | Uncertainty (Dissociation) = Low k.
    """
    # Distance from 0.5 (maximum uncertainty)
    certainty = np.abs(qs[0] - 0.5) * 2
    return round(certainty, 3)

def run_diagnostic_engine(simulated_minutes=20):
    print("🚀 [GIFT ENGINE]: Initializing Neural Relativity Suite...")
    print("📡 [STATUS]: Monitoring HRIT Density Tensor (H, S, A, R)...")
    print("-" * 50)

    for i in range(simulated_minutes):
        # 1. SIMULATE SENSOR (MNE-Python Interface)
        # We simulate a 'Noisy' signal (1) increasing as time goes on
        noise_chance = i / simulated_minutes 
        observation = [1] if np.random.random() < noise_chance else [0]
        
        # 2. UPDATE BRAIN (pymdp)
        # Your agent 'thinks' about the noise
        qs = my_hrit_agent.infer_states(observation)
        
        # 3. CALCULATE GEOMETRY (GIFT)
        # We turn that thought into a Curvature Score (k)
        k_score = calculate_k_score(qs[0])
        
        # 4. VPM ALARM (The Predictive Part)
        k_history.append(k_score)
        vpm_status = "STABLE"
        
        if len(k_history) > BUFFER_SIZE:
            k_history.pop(0)
            variance = np.var(k_history)
            
            if variance > VPM_THRESHOLD:
                vpm_status = "⚠️  VPM ALERT: MANIFOLD INSTABILITY"
            elif k_score < 0.2:
                vpm_status = "❌ CRITICAL: PHENOMENAL SNAP"

        # OUTPUT DISPLAY
        print(f"Time: {i}s | Input: {observation} | k-Score: {k_score:.2f} | {vpm_status}")
        
        time.sleep(0.3) # Slow it down so you can watch it run

if __name__ == "__main__":
    run_diagnostic_engine()
