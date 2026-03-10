"""
BATCH PROCESSOR FOR GIFT ENGINE - Multiple Patient Analysis
Processes EEG data from multiple patients and generates cohort-level diagnostics
"""

import numpy as np
import mne
import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from hrit_agent import my_hrit_agent, process_neural_signal
from sensor_interface import get_observation_from_eeg
from manifold_geometry import calculate_k_score

# --- CONFIGURATION ---
VPM_THRESHOLD = 0.045
WINDOW_SIZE = 5
BASELINE_DURATION = 10
SNAP_THRESHOLD = 0.2
BASELINE_K_THRESHOLD = 0.05

class DiagnosticClassifier:
    """Single patient diagnostic classifier"""
    def __init__(self, patient_id):
        self.patient_id = patient_id
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
        
    def update(self, t, k, obs, belief):
        """Update classifier with current data"""
        self.k_values_all.append(k)
        diagnosis = "stable"  # Default diagnosis
        
        # Baseline Tracking
        if not self.baseline_established and t < BASELINE_DURATION:
            if len(self.k_values_all) >= WINDOW_SIZE:
                recent_variance = np.var(self.k_values_all[-WINDOW_SIZE:])
                if recent_variance < BASELINE_K_THRESHOLD:
                    self.baseline_k_values.append(k)
            
            if t == BASELINE_DURATION - 1 and len(self.baseline_k_values) >= 3:
                self.baseline_mean = np.mean(self.baseline_k_values)
                self.baseline_established = True
        
        # Real-time Diagnosis
        if len(self.k_values_all) >= WINDOW_SIZE:
            recent_k_hist = self.k_values_all[-WINDOW_SIZE:]
            variance = np.var(recent_k_hist)
            
            if k < SNAP_THRESHOLD:
                diagnosis = "phenomenal_snap"
                self.state_counts['phenomenal_snap'] += 1
            elif variance > VPM_THRESHOLD:
                diagnosis = "pre_dissociative"
                self.state_counts['pre_dissociative'] += 1
            else:
                diagnosis = "stable"
                self.state_counts['stable'] += 1
            
            self.diagnoses.append({
                'time_s': t,
                'k_score': k,
                'variance': variance,
                'diagnosis': diagnosis
            })
        else:
            self.state_counts['stable'] += 1
        
        return diagnosis
    
    def get_summary(self):
        """Return diagnostic summary as dict"""
        total_states = sum(self.state_counts.values())
        summary = {
            'patient_id': self.patient_id,
            'total_frames': total_states,
            'baseline_mean': self.baseline_mean,
            'baseline_established': self.baseline_established,
        }
        
        if total_states > 0:
            summary['stable_pct'] = (self.state_counts['stable'] / total_states) * 100
            summary['pre_dissociative_pct'] = (self.state_counts['pre_dissociative'] / total_states) * 100
            summary['phenomenal_snap_pct'] = (self.state_counts['phenomenal_snap'] / total_states) * 100
        
        if self.k_values_all:
            summary['k_mean'] = float(np.mean(self.k_values_all))
            summary['k_std'] = float(np.std(self.k_values_all))
            summary['k_min'] = float(np.min(self.k_values_all))
            summary['k_max'] = float(np.max(self.k_values_all))
        
        return summary


class BatchProcessor:
    """Processes multiple patient EEG files"""
    def __init__(self, output_dir="batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.patient_summaries = []
        self.processed_patients = 0
        
    def process_eeg_file(self, eeg_path, patient_id, max_seconds=30):
        """Process a single EEG file"""
        try:
            print(f"\n🔬 Processing Patient {patient_id}...", end=" ")
            
            # Load EEG data
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=0)
            eeg_data = raw.get_data(picks='eeg')[0]
            
            # Initialize classifier
            classifier = DiagnosticClassifier(patient_id)
            
            # Process in chunks (1 second each at 100 samples/second)
            for t in range(min(max_seconds, len(eeg_data) // 100)):
                start_idx = t * 100
                end_idx = start_idx + 100
                if end_idx > len(eeg_data):
                    break
                
                eeg_chunk = eeg_data[start_idx:end_idx]
                obs = get_observation_from_eeg(eeg_chunk)
                belief = process_neural_signal(obs, my_hrit_agent)
                k = calculate_k_score(belief)
                classifier.update(t, k, obs, belief)
            
            # Store summary
            summary = classifier.get_summary()
            self.patient_summaries.append(summary)
            self.processed_patients += 1
            
            print(f"✅ ({classifier.state_counts['stable']} stable, "
                  f"{classifier.state_counts['pre_dissociative']} unstable, "
                  f"{classifier.state_counts['phenomenal_snap']} snap)")
            
            return summary
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return None
    
    def process_dataset(self, file_list):
        """Process multiple EEG files"""
        print("="*80)
        print("🧠 GIFT ENGINE: BATCH PROCESSING MODE")
        print("="*80)
        
        for idx, eeg_file in enumerate(file_list, 1):
            patient_id = f"P{idx:03d}"
            self.process_eeg_file(eeg_file, patient_id)
        
        return self.get_cohort_statistics()
    
    def get_cohort_statistics(self):
        """Compute statistics across all patients"""
        if not self.patient_summaries:
            return None
        
        # Extract arrays for vectorized computation
        k_means = np.array([p['k_mean'] for p in self.patient_summaries if 'k_mean' in p])
        stable_pcts = np.array([p['stable_pct'] for p in self.patient_summaries if 'stable_pct' in p])
        snap_pcts = np.array([p['phenomenal_snap_pct'] for p in self.patient_summaries if 'phenomenal_snap_pct' in p])
        pre_diss_pcts = np.array([p['pre_dissociative_pct'] for p in self.patient_summaries if 'pre_dissociative_pct' in p])
        
        cohort_stats = {
            'total_patients': len(self.patient_summaries),
            'processed_patients': self.processed_patients,
            'k_score': {
                'mean': float(np.mean(k_means)) if len(k_means) > 0 else 0,
                'std': float(np.std(k_means)) if len(k_means) > 0 else 0,
                'min': float(np.min(k_means)) if len(k_means) > 0 else 0,
                'max': float(np.max(k_means)) if len(k_means) > 0 else 0,
            },
            'state_distribution': {
                'stable_mean_pct': float(np.mean(stable_pcts)) if len(stable_pcts) > 0 else 0,
                'pre_dissociative_mean_pct': float(np.mean(pre_diss_pcts)) if len(pre_diss_pcts) > 0 else 0,
                'phenomenal_snap_mean_pct': float(np.mean(snap_pcts)) if len(snap_pcts) > 0 else 0,
            },
            'risk_assessment': {
                'patients_with_snap': len([p for p in self.patient_summaries if p.get('phenomenal_snap_pct', 0) > 0]),
                'patients_with_instability': len([p for p in self.patient_summaries if p.get('pre_dissociative_pct', 0) > 0]),
            }
        }
        
        return cohort_stats
    
    def save_results(self):
        """Save detailed results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'patient_summaries': self.patient_summaries,
            'cohort_statistics': self.get_cohort_statistics()
        }
        
        output_file = self.output_dir / 'batch_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        return results


def download_public_eeg_dataset(dataset_name="temp_uva", num_samples=30):
    """Download or fetch public EEG datasets"""
    print(f"📡 Downloading {dataset_name} dataset...")
    
    # Using Temple University Hospital EEG Seizure Corpus
    # This is a publicly available dataset with many subjects
    datasets_dir = Path.home() / "eeg_datasets" / dataset_name
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, use MNE sample data replicated to simulate multiple patients
    # In production, you'd download from OpenNeuro, PhysioNet, etc.
    print(f"⚠️  Generating simulated patient dataset ({num_samples} samples)...")
    
    file_list = []
    data_path = mne.datasets.sample.data_path()
    base_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    
    # Create symbolic links to simulate multiple patients
    for i in range(num_samples):
        patient_file = datasets_dir / f"patient_{i+1:03d}_eeg.fif"
        if not patient_file.exists():
            # Copy the same file for demo (in production, would be real different patient data)
            import shutil
            shutil.copy(base_fname, patient_file)
        file_list.append(str(patient_file))
    
    print(f"✅ Dataset ready: {len(file_list)} patient files")
    return file_list


if __name__ == "__main__":
    # Download public EEG dataset (~30 patients)
    patient_files = download_public_eeg_dataset("temple_uva", num_samples=30)
    
    # Process all patients
    processor = BatchProcessor(output_dir="/workspaces/pymdp/GIFT_Engine/batch_results")
    cohort_stats = processor.process_dataset(patient_files)
    
    # Save results
    processor.save_results()
    
    # Display cohort summary
    print("\n" + "="*80)
    print("🧬 COHORT-LEVEL ANALYSIS")
    print("="*80)
    if cohort_stats:
        print(f"\n📊 Total Patients Processed: {cohort_stats['processed_patients']}/{cohort_stats['total_patients']}")
        print(f"\n📐 K-Score (Population):")
        print(f"   Mean: {cohort_stats['k_score']['mean']:.4f} ± {cohort_stats['k_score']['std']:.4f}")
        print(f"   Range: [{cohort_stats['k_score']['min']:.4f}, {cohort_stats['k_score']['max']:.4f}]")
        print(f"\n📈 State Distribution (Mean % per patient):")
        print(f"   ✅ Stable: {cohort_stats['state_distribution']['stable_mean_pct']:.1f}%")
        print(f"   ⚠️  Pre-Dissociative: {cohort_stats['state_distribution']['pre_dissociative_mean_pct']:.1f}%")
        print(f"   ❌ Phenomenal Snap: {cohort_stats['state_distribution']['phenomenal_snap_mean_pct']:.1f}%")
        print(f"\n🔬 Risk Assessment:")
        print(f"   Patients with Snap Events: {cohort_stats['risk_assessment']['patients_with_snap']}")
        print(f"   Patients with Instability: {cohort_stats['risk_assessment']['patients_with_instability']}")
    print("="*80)
