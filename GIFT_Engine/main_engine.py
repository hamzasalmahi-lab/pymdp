import numpy as np
import mne
import time

# --- 1. INTERNAL IMPORTS ---
# These connect your 'Brain', 'Sensor', and 'Geometry' files
try:
    from hrit_agent import my_hrit_agent, process_neural_signal 
    from sensor_interface import get_observation_from_eeg
    from manifold_geometry import calculate_k_score
    from plot_learning import plot_agent_learning
except ImportError as e:
    print(f"❌ ERROR: Could not find a support file. Make sure all .py files are in the same folder.\n{e}")
    exit()

# --- 2. VPM SENTINEL SETTINGS ---
k_history = []
VPM_THRESHOLD = 0.045           # From GIFT.pdf: The 'Wobble' limit
WINDOW_SIZE = 5                 # We analyze the last 5 seconds of data
BASELINE_DURATION = 10          # First 10 seconds to establish baseline
SNAP_THRESHOLD = 0.2            # k-score below this indicates Phenomenal Snap
BASELINE_K_THRESHOLD = 0.05     # Variance threshold for baseline stability

# --- 3. DIAGNOSTIC CLASSIFIER STATE ---
class DiagnosticClassifier:
    def __init__(self):
        self.baseline_k_values = []
        self.baseline_established = False
        self.baseline_mean = None
        self.k_values_all = []
        self.diagnoses = []
        self.state_counts = {
            'stable': 0,
            'pre_dissociative': 0,
            'phenomenal_snap': 0
        }
        self.baseline_seconds_completed = 0
        self.snap_duration = 0  # Track total seconds of dissociative snap
        
    def update(self, t, k, obs, belief):
        """Update classifier with current neuroscience data"""
        self.k_values_all.append(k)
        
        # Phase 1: Baseline Tracking (first 10 seconds of stable data)
        if not self.baseline_established and t < BASELINE_DURATION:
            if len(self.k_values_all) >= WINDOW_SIZE:
                recent_variance = np.var(self.k_values_all[-WINDOW_SIZE:])
                if recent_variance < BASELINE_K_THRESHOLD:
                    self.baseline_k_values.append(k)
            
            if t == BASELINE_DURATION - 1 and len(self.baseline_k_values) >= 3:
                self.baseline_mean = np.mean(self.baseline_k_values)
                self.baseline_established = True
                self.baseline_seconds_completed = BASELINE_DURATION
        
        # Phase 2: Real-time Diagnosis
        if len(self.k_values_all) >= WINDOW_SIZE:
            recent_k_hist = self.k_values_all[-WINDOW_SIZE:]
            variance = np.var(recent_k_hist)
            
            # Diagnosis 1: Phenomenal Snap (Metric Collapse)
            if k < SNAP_THRESHOLD:
                diagnosis = "❌ DIAGNOSIS: PHENOMENAL SNAP"
                self.state_counts['phenomenal_snap'] += 1
                self.snap_duration += 1  # Track duration of dissociative snap
                
            # Diagnosis 2: Pre-Dissociative Instability
            elif variance > VPM_THRESHOLD:
                diagnosis = "⚠️  DIAGNOSIS: PRE-DISSOCIATIVE INSTABILITY"
                self.state_counts['pre_dissociative'] += 1
                
            # Baseline: Stable
            else:
                diagnosis = "✅ STABLE"
                self.state_counts['stable'] += 1
            
            self.diagnoses.append({
                'time_s': t,
                'k_score': k,
                'variance': variance,
                'diagnosis': diagnosis
            })
        else:
            self.state_counts['stable'] += 1
        
        return diagnosis if len(self.diagnoses) > 0 else "✅ STABLE"
    
    def generate_report(self, total_time_s):
        """Generate diagnostic summary report"""
        total_states = sum(self.state_counts.values())
        
        report = []
        report.append("\n" + "=" * 75)
        report.append("🧬 GIFT ENGINE DIAGNOSTIC SUMMARY REPORT 🧬")
        report.append("=" * 75)
        
        # Baseline Information
        if self.baseline_established:
            report.append(f"\n📊 BASELINE CALIBRATION (First {self.baseline_seconds_completed}s):")
            report.append(f"   • Mean k-score (Baseline): {self.baseline_mean:.4f}")
            report.append(f"   • Samples used for baseline: {len(self.baseline_k_values)}")
        else:
            report.append(f"\n⚠️  BASELINE: Not established (insufficient stable data)")
        
        # State Distribution
        report.append(f"\n📈 STATE DISTRIBUTION (Total Time: {total_time_s}s):")
        if total_states > 0:
            stable_pct = (self.state_counts['stable'] / total_states) * 100
            pre_diss_pct = (self.state_counts['pre_dissociative'] / total_states) * 100
            snap_pct = (self.state_counts['phenomenal_snap'] / total_states) * 100
            
            report.append(f"   • ✅ STABLE: {self.state_counts['stable']} frames ({stable_pct:.1f}%)")
            report.append(f"   • ⚠️  PRE-DISSOCIATIVE: {self.state_counts['pre_dissociative']} frames ({pre_diss_pct:.1f}%)")
            report.append(f"   • ❌ PHENOMENAL SNAP: {self.state_counts['phenomenal_snap']} frames ({snap_pct:.1f}%)")
        
        # Statistical Summary
        if self.k_values_all:
            report.append(f"\n📐 K-SCORE STATISTICS:")
            report.append(f"   • Mean k-score: {np.mean(self.k_values_all):.4f}")
            report.append(f"   • Std Dev: {np.std(self.k_values_all):.4f}")
            report.append(f"   • Min k-score: {np.min(self.k_values_all):.4f}")
            report.append(f"   • Max k-score: {np.max(self.k_values_all):.4f}")
        
        # VPM Threshold Summary
        report.append(f"\n⚙️  VPM SENTINEL THRESHOLDS:")
        report.append(f"   • Variance Threshold (Pre-Dissociative): {VPM_THRESHOLD}")
        report.append(f"   • Metric Collapse Threshold (Snap): {SNAP_THRESHOLD}")
        report.append(f"   • Window Size: {WINDOW_SIZE}s")
        
        # Clinical Assessment
        report.append(f"\n🔬 CLINICAL ASSESSMENT:")
        if self.state_counts['phenomenal_snap'] > 0:
            report.append(f"   ⚠️  ALERT: Phenomenal Snap detected ({self.state_counts['phenomenal_snap']} events)")
        if self.state_counts['pre_dissociative'] > 0:
            report.append(f"   ⚠️  ALERT: Pre-Dissociative Instability detected ({self.state_counts['pre_dissociative']} events)")
        if self.state_counts['stable'] > (total_states * 0.7):
            report.append(f"   ✅ System remained stable for most of the session")
        
        # Doctor's Note
        report.append(f"\n👨‍⚕️ DOCTOR'S NOTE:")
        if self.snap_duration > 0:
            report.append(f"   • Total duration of Dissociative Snap: {self.snap_duration} seconds")
            report.append(f"   • Clinical significance: Prolonged dissociative episodes may indicate severe neural instability")
            report.append(f"   • Recommendation: Immediate clinical intervention and monitoring required")
        else:
            report.append(f"   • No dissociative snaps detected during monitoring period")
            report.append(f"   • Patient maintained stable neural manifold throughout session")
            report.append(f"   • Status: Neural integrity preserved")
        
        report.append("\n" + "=" * 75)
        
        return "\n".join(report)

def run_diagnostic_loop(iterations=30):
    print("🧠 [HRIT-GIFT ENGINE]: ONLINE")
    print("📡 [MONITOR]: Watching for VPM Instability & Metric Collapse...")
    print("-" * 80)
    
    # Initialize diagnostic classifier
    classifier = DiagnosticClassifier()
    
    # Load real EEG data from MNE sample dataset
    print("Loading sample EEG data...")
    data_path = mne.datasets.sample.data_path()
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    eeg_data = raw.get_data(picks='eeg')[0]  # Use first EEG channel
    print(f"Loaded {len(eeg_data)} samples of EEG data.")
    print("-" * 80)

    for t in range(iterations):
        # A. SENSOR STEP (Using real EEG data)
        start_idx = t * 100
        end_idx = start_idx + 100
        if end_idx > len(eeg_data):
            end_idx = len(eeg_data)
        simulated_eeg_chunk = eeg_data[start_idx:end_idx]
        if len(simulated_eeg_chunk) < 100:
            simulated_eeg_chunk = np.pad(simulated_eeg_chunk, (0, 100 - len(simulated_eeg_chunk)), mode='constant')
            
        # Extract observation using new preprocessing pipeline
        obs, alpha_normalized, alpha_power = get_observation_from_eeg(simulated_eeg_chunk)

        # B. INFERENCE STEP (The 'Brain' thinks)
        belief = process_neural_signal(obs, my_hrit_agent)

        # C. GEOMETRY STEP (The 'Map' calculates curvature)
        k = calculate_k_score(belief)

        # D. DIAGNOSTIC CLASSIFICATION
        diagnosis = classifier.update(t, k, obs, belief)

        # E. CLINICAL DASHBOARD
        # Determine manifold stability
        if "PHENOMENAL SNAP" in diagnosis:
            manifold_stability = "Collapsed"
            vpm_risk = "Critical"
        elif "PRE-DISSOCIATIVE" in diagnosis:
            manifold_stability = "Unstable"
            vpm_risk = "Elevated"
        else:
            manifold_stability = "Stable"
            vpm_risk = "Low"
        
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                     CLINICAL DASHBOARD                      ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║ Neural Precision:          {k:.4f}                           ║")
        print(f"║ Manifold Stability:        {manifold_stability:<12}                     ║")
        print(f"║ VPM Risk Level:           {vpm_risk:<12}                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        
        time.sleep(1)  # Update every 1 second
    
    # F. GENERATE DIAGNOSTIC REPORT
    report = classifier.generate_report(iterations)
    print(report)
    
    # G. PRINT AGENT LEARNING (A MATRIX CHANGES)
    print("\n" + "=" * 80)
    print("🧠 AGENT GENERATIVE MODEL (A MATRIX) - Belief Updates")
    print("=" * 80)
    print("\n📊 INITIAL A MATRIX (Prior Beliefs):")
    print("   Observation → State Mapping (obs | state)")
    print("   " + "-" * 70)
    print(f"   State 0 (Stable):      Obs=0: {0.9:.2f}  |  Obs=1: {0.1:.2f}")
    print(f"   State 1 (Unstable):    Obs=0: {0.3:.2f}  |  Obs=1: {0.7:.2f}")
    print("\n   Interpretation:")
    print("   • When agent believes state=STABLE: expects obs=0 (90% confidence)")
    print("   • When agent believes state=UNSTABLE: expects obs=1 (70% confidence)")
    
    # Get final A matrix from agent
    final_A = my_hrit_agent.A[0]
    print(f"\n📊 FINAL A MATRIX (After {iterations} observations):")
    print("   " + "-" * 70)
    print(f"   State 0 (Stable):      Obs=0: {final_A[0, 0]:.4f}  |  Obs=1: {final_A[1, 0]:.4f}")
    print(f"   State 1 (Unstable):    Obs=0: {final_A[0, 1]:.4f}  |  Obs=1: {final_A[1, 1]:.4f}")
    
    # Calculate changes
    change_00 = final_A[0, 0] - 0.9
    change_10 = final_A[1, 0] - 0.1
    change_01 = final_A[0, 1] - 0.3
    change_11 = final_A[1, 1] - 0.7
    
    print(f"\n📈 BELIEF UPDATES (Δ = Final - Initial):")
    print("   " + "-" * 70)
    print(f"   Stable state, obs=0:     {change_00:+.4f}  (was 0.9000, now {final_A[0, 0]:.4f})")
    print(f"   Stable state, obs=1:     {change_10:+.4f}  (was 0.1000, now {final_A[1, 0]:.4f})")
    print(f"   Unstable state, obs=0:   {change_01:+.4f}  (was 0.3000, now {final_A[0, 1]:.4f})")
    print(f"   Unstable state, obs=1:   {change_11:+.4f}  (was 0.7000, now {final_A[1, 1]:.4f})")
    
    print(f"\n🔬 Clinical Interpretation:")
    if abs(change_00) > 0.01 or abs(change_11) > 0.01:
        print(f"   ✅ Agent LEARNED from EEG observations")
        print(f"   • Updated stability beliefs by up to {max(abs(change_00), abs(change_11)):.4f}")
        print(f"   • Model confidence improved through active inference")
    else:
        print(f"   ⚠️  Minimal learning detected")
        print(f"   • Agent prior beliefs well-matched to data")
        print(f"   • No significant Bayesian updates required")
    
    print("=" * 80)
    
    # H. SAVE LEARNING VISUALIZATION
    print("\n📊 Generating learning visualization...")
    initial_A = np.array([[0.9, 0.3], [0.1, 0.7]])
    plot_path = plot_agent_learning(initial_A, final_A, patient_id="GIFT_Engine_Single_Patient")
    print(f"✅ Learning plot saved to: {plot_path}")

if __name__ == "__main__":
    run_diagnostic_loop()
