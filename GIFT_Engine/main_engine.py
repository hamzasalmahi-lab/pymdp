import numpy as np
import mne
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
    
    # Load real EEG data from MNE sample dataset
    print("Loading sample EEG data...")
    data_path = mne.datasets.sample.data_path()
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    eeg_data = raw.get_data(picks='eeg')[0]  # Use first EEG channel
    print(f"Loaded {len(eeg_data)} samples of EEG data.")

    for t in range(iterations):
        # A. SENSOR STEP (Using real EEG data)
        start_idx = t * 100
        end_idx = start_idx + 100
        if end_idx > len(eeg_data):
            end_idx = len(eeg_data)
        simulated_eeg_chunk = eeg_data[start_idx:end_idx]
        if len(simulated_eeg_chunk) < 100:
            simulated_eeg_chunk = np.pad(simulated_eeg_chunk, (0, 100 - len(simulated_eeg_chunk)), mode='constant')
            
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
