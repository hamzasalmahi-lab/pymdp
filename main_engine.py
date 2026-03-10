import numpy as np
import time
# Ensure hrit_agent.py is in the same folder
from hrit_agent import my_hrit_agent, process_neural_signal 

# --- CONSTANTS (HRIT/GIFT PARAMETERS) ---
VPM_THRESHOLD = 0.045  # Variance threshold for 'Metric Instability'
WINDOW_SIZE = 6        # How many seconds of history to analyze
k_history = []         # Buffer for the GIFT-Score

def calculate_k_score(qs):
    """
    Calculates the GIFT-Score (k) using Information Geometry.
    k = 1.0 (Manifold Depth/Presence) | k = 0.0 (Metric Collapse/Snap)
    """
    # Measure distance from the 'Uncertainty Center' (0.5)
    # We use the absolute difference to quantify 'Belief Precision'
    certainty = np.abs(qs[0] - 0.5) * 2
    return round(float(certainty), 4)

def run_engine():
    print("🧠 [HRIT-GIFT ENGINE]: ACTIVATED")
    print("🛠️  [DIAGNOSTICS]: Monitoring VPM Variance & Phenomenal Snap Risk...")
    print("-" * 60)

    try:
        # Simulate a 30-second neural stream
        for t in range(30):
            # 1. SENSOR MOCK (Simulating MNE-Python output)
            # As time increases, we simulate a 'stressor' increasing noise
            noise_level = 1 if (t > 15 and np.random.random() > 0.4) else 0
            
            # 2. BRAIN (pymdp Inference)
            # The agent 'thinks' about the noise signal
            belief = process_neural_signal(noise_level, my_hrit_agent)
            
            # 3. GEOMETRY (GIFT Field Equation)
            # We map the belief to a curvature score (k)
            k = calculate_k_score(belief)
            k_history.append(k)
            
            # 4. SENTINEL (VPM Early Warning)
            status = "STABLE"
            if len(k_history) >= WINDOW_SIZE:
                k_history.pop(0)
                # Calculate the 'Wobble' (Variance)
                current_variance = np.var(k_history)
                
                if current_variance > VPM_THRESHOLD:
                    status = "⚠️  VPM ALERT: INSTABILITY DETECTED"
                elif k < 0.2:
                    status = "❌ CRITICAL: PHENOMENAL SNAP"

            # CONSOLE DASHBOARD
            print(f"[{t:02d}s] | Input: {noise_level} | k-Score: {k:.3f} | {status}")
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\n[ENGINE]: Manual Override. Shutting down...")

if __name__ == "__main__":
    run_engine()
