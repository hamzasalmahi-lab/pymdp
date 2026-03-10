"""
study_manager.py

Performs a retrospective study using the GIFT Engine by comparing a
"Control" cohort against a "Pathological" cohort.  The script walks through
all EEG files stored in the two data folders, runs the engine diagnostic
loop with learning enabled, collects summary metrics and then performs a
simple statistics comparison on the resulting k-scores.

Usage:
    python study_manager.py

Requirements:
    - pandas
    - scipy
    - mne
    - matplotlib
    - copy

The output CSV is written to /exports/retrospective_study_results.csv and a
k-score distribution figure is saved to the same directory.
"""

import os
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind
import logging
import importlib
import copy

from hrit_agent import process_neural_signal
import hrit_agent
from sensor_interface import get_observation_from_eeg
from manifold_geometry import calculate_k_score

# reuse constants defined in main_engine
from main_engine import WINDOW_SIZE, BASELINE_DURATION

# ensure exports folder exists
EXPORT_DIR = Path("/workspaces/pymdp/exports")
EXPORT_DIR.mkdir(exist_ok=True)

# study directories (relative to workspace root)
ROOT = Path(__file__).resolve().parent.parent
CONTROL_DIR = ROOT / "data" / "control"
PATHOLOGICAL_DIR = ROOT / "data" / "pathological"

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


class FileMetrics:
    def __init__(self):
        self.ks = []
        self.variances = []
        self.snap_duration = 0


def run_single_file(filepath):
    """Run the diagnostic loop on a single EEG file and return metrics."""
    # reload hrit_agent module to ensure fresh A/B/C and agent state
    importlib.reload(hrit_agent)
    # create a fresh agent instance (learning enabled)
    agent = hrit_agent.Agent(A=copy.deepcopy(hrit_agent.A),
                             B=copy.deepcopy(hrit_agent.B),
                             C=copy.deepcopy(hrit_agent.C))
    agent.learn_A = True

    # prepare classifier using same class as main_engine
    from main_engine import DiagnosticClassifier
    classifier = DiagnosticClassifier()

    # load file with mne
    try:
        raw = mne.io.read_raw(filepath, preload=True, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load {filepath}: {e}")
        return None

    # use first EEG channel available
    try:
        eeg_data = raw.get_data(picks='eeg')[0]
    except Exception:
        # fallback to first channel if eeg pick fails
        eeg_data = raw.get_data()[0]

    sfreq = int(np.round(raw.info.get('sfreq', 1)))
    total_secs = int(np.ceil(len(eeg_data) / sfreq))

    metrics = FileMetrics()

    # iterate one-second windows
    for t in range(total_secs):
        start = t * sfreq
        end = start + sfreq
        chunk = eeg_data[start:end]
        if len(chunk) < sfreq:
            chunk = np.pad(chunk, (0, sfreq - len(chunk)), mode='constant')
        obs, alpha_norm, alpha_power = get_observation_from_eeg(chunk)
        belief = process_neural_signal(obs, agent)
        k = calculate_k_score(belief)

        # update classifier
        diagnosis = classifier.update(t, k, obs, belief)

        metrics.ks.append(k)
        # compute variance over last window.
        if len(classifier.k_values_all) >= WINDOW_SIZE:
            metrics.variances.append(np.var(classifier.k_values_all[-WINDOW_SIZE:]))
        else:
            metrics.variances.append(0.0)

    # compute summary numbers
    # Minimum k-score to catch 'Snap' events
    min_k = np.min(metrics.ks) if metrics.ks else np.nan
    # 90th percentile of variance to detect instability spikes
    percentile_90_var = np.percentile(metrics.variances, 90) if metrics.variances else 0.0
    snap_count = classifier.snap_duration
    # learning delta: sum abs change in A
    initial_A = hrit_agent.A
    final_A = agent.A
    # A is an object array; compute total absolute change across all entries
    learning_delta = 0.0
    try:
        for idx in range(len(initial_A)):
            learning_delta += np.sum(np.abs(final_A[idx] - initial_A[idx]))
    except Exception:
        # fallback if not iterable
        learning_delta = np.sum(np.abs(final_A - initial_A))

    return {
        'min_k': min_k,
        'percentile_90_var': percentile_90_var,
        'snap_count': snap_count,
        'learning_delta': learning_delta,
        'ks_distribution': metrics.ks,
    }


def gather_group(directory, group_name):
    results = []
    min_ks = []

    for fname in os.listdir(directory):
        path = directory / fname
        if not path.is_file():
            continue
        logging.info(f"Processing file {path} ({group_name})")
        metrics = run_single_file(str(path))
        if metrics is None:
            continue
        results.append({
            'group': group_name,
            'file': fname,
            'min_k': metrics['min_k'],
            'percentile_90_var': metrics['percentile_90_var'],
            'snap_count': metrics['snap_count'],
            'learning_delta': metrics['learning_delta'],
        })
        min_ks.append(metrics['min_k'])
    return results, min_ks


def main():
    control_results, control_min_ks = gather_group(CONTROL_DIR, 'Control')
    path_results, path_min_ks = gather_group(PATHOLOGICAL_DIR, 'Pathological')

    df = pd.DataFrame(control_results + path_results)
    csv_path = EXPORT_DIR / 'retrospective_study_results.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"Results written to {csv_path}")

    # statistical comparison of min_k
    control_mins = df[df.group == 'Control']['min_k'].dropna()
    path_mins = df[df.group == 'Pathological']['min_k'].dropna()
    tstat, pval = ttest_ind(control_mins, path_mins, equal_var=False)

    print("\n===== RETROSPECTIVE STUDY SUMMARY =====\n")
    print(df[['group', 'file', 'min_k', 'percentile_90_var', 'snap_count', 'learning_delta']])
    
    # Table for pathological minimum k-scores
    print("\n===== PATHOLOGICAL GROUP MINIMUM K-SCORES =====\n")
    path_df = df[df.group == 'Pathological'][['file', 'min_k']]
    print(path_df)
    
    # Check if any below 0.4
    if (path_df['min_k'] < 0.4).any():
        print("\n✅ Some pathological files reached the Risk Zone (k < 0.4)")
    else:
        print("\n⚠️  No pathological files reached the Risk Zone (k < 0.4)")
        print("   Recommendation: Adjust sensor_interface.py to increase sensitivity to high-frequency noise")
    
    print(f"\np-value for Min_K difference (Control vs Pathological): {pval:.4e}\n")

    # plot distributions of minimum k-scores
    plt.figure(figsize=(10,6))
    
    # For small datasets, use histogram; for larger, KDE
    if len(control_min_ks) > 1:
        kde_control = stats.gaussian_kde(control_min_ks)
        x_control = np.linspace(min(control_min_ks), max(control_min_ks), 1000)
        plt.plot(x_control, kde_control(x_control), label='Control', color='blue', linewidth=2)
        plt.fill_between(x_control, kde_control(x_control), alpha=0.3, color='blue')
    else:
        plt.hist(control_min_ks, bins=10, alpha=0.6, label='Control', color='blue', density=True)
    
    if len(path_min_ks) > 1:
        kde_path = stats.gaussian_kde(path_min_ks)
        x_path = np.linspace(min(path_min_ks), max(path_min_ks), 1000)
        plt.plot(x_path, kde_path(x_path), label='Pathological', color='red', linewidth=2)
        plt.fill_between(x_path, kde_path(x_path), alpha=0.3, color='red')
    else:
        plt.hist(path_min_ks, bins=10, alpha=0.6, label='Pathological', color='red', density=True)
    
    # Vertical line at k=0.4
    plt.axvline(x=0.4, color='black', linestyle='--', linewidth=2, label='k = 0.4 Threshold')
    
    # Shade Clinical Risk Zone (below 0.4)
    plt.axvspan(plt.xlim()[0], 0.4, alpha=0.2, color='red', label='Clinical Risk Zone')
    
    plt.xlabel('Minimum k-score (Stability)')
    plt.ylabel('Density')
    plt.title('Distribution of Minimum k-score (Stability) per File')
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = EXPORT_DIR / 'group_comparison_k_score.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Group comparison plot saved to {plot_path}")


if __name__ == '__main__':
    main()
