import numpy as np
import time

# --- 1. INTERNAL IMPORTS ---
# These connect your 'Brain', 'Sensor', and 'Geometry' files
try:
    from hrit_agent import my_hrit_agent, process_neural_signal 
    from sensor_interface import get_observation_from_eeg
    from manifold_geometry import calculate_k_score
except ImportError as e:
    print(f"❌ ERROR: Could not find a support file. Make sure all .py files are in the same folder.\n{e}")
    exit()

# --- 2. VPM SENTINEL SETTINGS ---
k_history = []
VPM_THRESHOLD = 0.045 # From GIFT.pdf: The 'Wobble' limit
WINDOW_SIZE = 5       # We analyze the last 5 seconds of data

def run_diagnostic_loop(iterations=30):
    print("🧠 [HRIT-GIFT ENGINE]: ONLINE")
    print("📡 [MONITOR]: Watching for VPM Instability & Metric Collapse...")
    print("-" * 65)

    for t in range(iterations):
        # A. SENSOR STEP (Mocking MNE-Python)
        # We simulate a brain signal that gets noisier as time passes
        simulated_eeg_chunk = np.random.normal(0, 1, 100) 
        if t > 15: # Simulate a stressor starting at 15 seconds
            simulated_eeg_chunk += np.random.normal(0, 2, 100)
            
        obs = get_observation_from_eeg(simulated_eeg_chunk)

        # B. INFERENCE STEP (The 'Brain' thinks)
        belief = process_neural_signal(obs, my_hrit_agent)

        # C. GEOMETRY STEP (The 'Map' calculates curvature)
        k = calculate_k_score(belief)
        k_history.append(k)

        # D. VPM DIAGNOSTIC (The 'Alarm')
        status = "✅ STABLE"
        if len(k_history) >= WINDOW_SIZE:
            k_history.pop(0)
            variance = np.var(k_history)
            
            if variance > VPM_THRESHOLD:
                status = "⚠️  VPM ALERT: CRITICAL WOBBLE"
            elif k < 0.2:
                status = "❌ CRITICAL: PHENOMENAL SNAP"

        # E. CONSOLE DASHBOARD
        print(f"[{t:02d}s] | Input: {obs} | k-Score: {k:.2f} | {status}")
        
        time.sleep(0.5) # Real-time simulation speed

if __name__ == "__main__":
    run_diagnostic_loop()
