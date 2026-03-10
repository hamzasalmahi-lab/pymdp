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
    # mean k after calibration (10s)
    if len(metrics.ks) > BASELINE_DURATION:
        mean_k = np.mean(metrics.ks[BASELINE_DURATION:])
    else:
        mean_k = np.mean(metrics.ks) if metrics.ks else np.nan

    peak_variance = np.max(metrics.variances) if metrics.variances else 0.0
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
        'mean_k': mean_k,
        'peak_variance': peak_variance,
        'snap_count': snap_count,
        'learning_delta': learning_delta,
        'ks_distribution': metrics.ks,
    }


def gather_group(directory, group_name):
    results = []
    all_ks = []

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
            'mean_k': metrics['mean_k'],
            'peak_variance': metrics['peak_variance'],
            'snap_count': metrics['snap_count'],
            'learning_delta': metrics['learning_delta'],
        })
        all_ks.extend(metrics['ks_distribution'])
    return results, all_ks


def main():
    control_results, control_ks = gather_group(CONTROL_DIR, 'Control')
    path_results, path_ks = gather_group(PATHOLOGICAL_DIR, 'Pathological')

    df = pd.DataFrame(control_results + path_results)
    csv_path = EXPORT_DIR / 'retrospective_study_results.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"Results written to {csv_path}")

    # statistical comparison of mean_k
    control_means = df[df.group == 'Control']['mean_k'].dropna()
    path_means = df[df.group == 'Pathological']['mean_k'].dropna()
    tstat, pval = ttest_ind(control_means, path_means, equal_var=False)

    print("\n===== RETROSPECTIVE STUDY SUMMARY =====\n")
    print(df[['group', 'file', 'mean_k', 'peak_variance', 'snap_count', 'learning_delta']])
    print(f"\np-value for Mean_K difference (Control vs Pathological): {pval:.4e}\n")

    # plot distributions
    plt.figure(figsize=(8,6))
    plt.hist(control_ks, bins=30, alpha=0.6, label='Control')
    plt.hist(path_ks, bins=30, alpha=0.6, label='Pathological')
    plt.xlabel('k-score')
    plt.ylabel('Frequency')
    plt.title('Group Comparison of k-score Distributions')
    plt.legend()
    plot_path = EXPORT_DIR / 'group_comparison_k_score.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Group comparison plot saved to {plot_path}")


if __name__ == '__main__':
    main()
