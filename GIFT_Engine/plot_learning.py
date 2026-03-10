"""
PLOT LEARNING - GIFT Engine Agent Learning Visualization
Creates and saves diagnostic plots showing A matrix belief updates
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import os

def plot_agent_learning(initial_A, final_A, patient_id="Single_Patient", save_path=None):
    """
    Create diagnostic plot showing agent learning (A matrix changes)

    Parameters:
    -----------
    initial_A : array-like
        Initial A matrix (prior beliefs)
    final_A : array-like
        Final A matrix (after learning)
    patient_id : str
        Identifier for the patient/session
    save_path : str or Path
        Path to save the PNG file (if None, uses default)

    Returns:
    --------
    str : Path to saved PNG file
    """

    # Calculate belief updates
    delta_A = final_A - initial_A

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create exports folder if it doesn't exist
    exports_dir = Path("/workspaces/pymdp/exports")
    exports_dir.mkdir(exist_ok=True)

    # Default save path
    if save_path is None:
        filename = f"diagnostic_report_{timestamp}.png"
        save_path = exports_dir / filename

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'GIFT Engine - Agent Learning Visualization\nPatient: {patient_id}', fontsize=14, weight='bold')

    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Plot 1: Initial A Matrix
    im1 = ax1.imshow(initial_A, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('Initial A Matrix (Prior Beliefs)', fontsize=12, weight='bold')
    ax1.set_xlabel('Observations')
    ax1.set_ylabel('States')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Obs=0\n(Stable)', 'Obs=1\n(Unstable)'])
    ax1.set_yticklabels(['State 0\n(Stable)', 'State 1\n(Unstable)'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{initial_A[i, j]:.3f}',
                          ha="center", va="center", color="white", weight='bold')

    # Plot 2: Final A Matrix
    im2 = ax2.imshow(final_A, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Final A Matrix (After Learning)', fontsize=12, weight='bold')
    ax2.set_xlabel('Observations')
    ax2.set_ylabel('States')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Obs=0\n(Stable)', 'Obs=1\n(Unstable)'])
    ax2.set_yticklabels(['State 0\n(Stable)', 'State 1\n(Unstable)'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{final_A[i, j]:.4f}',
                          ha="center", va="center", color="white", weight='bold')

    # Plot 3: Belief Updates (Delta)
    im3 = ax3.imshow(delta_A, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
    ax3.set_title('Belief Updates (Δ = Final - Initial)', fontsize=12, weight='bold')
    ax3.set_xlabel('Observations')
    ax3.set_ylabel('States')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Obs=0\n(Stable)', 'Obs=1\n(Unstable)'])
    ax3.set_yticklabels(['State 0\n(Stable)', 'State 1\n(Unstable)'])

    # Add text annotations for delta
    for i in range(2):
        for j in range(2):
            color = "black" if abs(delta_A[i, j]) < 0.1 else "white"
            text = ax3.text(j, i, f'{delta_A[i, j]:+.4f}',
                          ha="center", va="center", color=color, weight='bold')

    # Plot 4: Learning Summary Statistics
    ax4.axis('off')
    ax4.set_title('Learning Summary', fontsize=12, weight='bold')

    # Calculate statistics
    max_change = np.max(np.abs(delta_A))
    mean_change = np.mean(np.abs(delta_A))
    learning_detected = max_change > 0.01

    summary_text = f"""
LEARNING STATISTICS

Maximum Belief Change: {max_change:.4f}
Mean Absolute Change: {mean_change:.4f}

Learning Assessment:
• {'✅ SIGNIFICANT LEARNING' if learning_detected else '⚠️  MINIMAL LEARNING'}

Clinical Interpretation:
"""

    if learning_detected:
        summary_text += """
• Agent updated beliefs based on EEG observations
• Generative model adapted to patient data
• Active inference learning successful"""
    else:
        summary_text += """
• Agent priors well-matched to observations
• No significant belief updates required
• Stable generative model maintained"""

    # Add belief interpretation
    summary_text += f"""

BELIEF INTERPRETATION

State 0 → Obs 0: {final_A[0, 0]:.3f} ({'+' if delta_A[0, 0] > 0 else ''}{delta_A[0, 0]:.3f})
  "Stable state produces stable obs"

State 0 → Obs 1: {final_A[1, 0]:.3f} ({'+' if delta_A[1, 0] > 0 else ''}{delta_A[1, 0]:.3f})
  "Stable state produces unstable obs"

State 1 → Obs 0: {final_A[0, 1]:.3f} ({'+' if delta_A[0, 1] > 0 else ''}{delta_A[0, 1]:.3f})
  "Unstable state produces stable obs"

State 1 → Obs 1: {final_A[1, 1]:.3f} ({'+' if delta_A[1, 1] > 0 else ''}{delta_A[1, 1]:.3f})
  "Unstable state produces unstable obs"
"""

    ax4.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
             family='monospace', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Add colorbars
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Probability')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Probability')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Change in Probability')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Learning visualization saved: {save_path}")
    return str(save_path)

def plot_batch_learning(batch_results_json, save_path=None):
    """
    Create cohort-level learning visualization from batch results

    Parameters:
    -----------
    batch_results_json : str or Path
        Path to batch results JSON file
    save_path : str or Path
        Path to save the PNG file

    Returns:
    --------
    str : Path to saved PNG file
    """

    # Load batch results
    with open(batch_results_json, 'r') as f:
        results = json.load(f)

    patient_summaries = results['patient_summaries']

    # Extract learning metrics (assuming minimal learning for demo)
    # In practice, you'd need to store initial/final A matrices per patient
    learning_scores = []
    for patient in patient_summaries:
        # Placeholder: calculate some learning metric from available data
        k_std = patient.get('k_std', 0)
        learning_scores.append(k_std)  # Higher variability might indicate learning

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create exports folder
    exports_dir = Path("/workspaces/pymdp/exports")
    exports_dir.mkdir(exist_ok=True)

    if save_path is None:
        filename = f"cohort_learning_report_{timestamp}.png"
        save_path = exports_dir / filename

    # Create cohort learning plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('GIFT Engine - Cohort Learning Analysis', fontsize=14, weight='bold')

    # Plot 1: Learning distribution across patients
    ax1.hist(learning_scores, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Learning Score (k-score Variability)')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Patient Learning Distribution')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Learning vs Stability correlation
    stable_pcts = [p.get('stable_pct', 0) for p in patient_summaries]
    ax2.scatter(stable_pcts, learning_scores, alpha=0.6, color='darkred', s=50)
    ax2.set_xlabel('Stability Percentage (%)')
    ax2.set_ylabel('Learning Score')
    ax2.set_title('Learning vs Stability Correlation')
    ax2.grid(alpha=0.3)

    # Add trend line
    if len(stable_pcts) > 1:
        z = np.polyfit(stable_pcts, learning_scores, 1)
        p = np.poly1d(z)
        ax2.plot(stable_pcts, p(stable_pcts), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Cohort learning visualization saved: {save_path}")
    return str(save_path)

if __name__ == "__main__":
    # Example usage for single patient
    # In practice, this would be called from main_engine.py

    # Mock A matrices (replace with real data)
    initial_A = np.array([[0.9, 0.3],
                          [0.1, 0.7]])

    final_A = np.array([[0.85, 0.35],
                        [0.15, 0.65]])  # Some learning occurred

    plot_path = plot_agent_learning(initial_A, final_A, patient_id="Demo_Patient")
    print(f"Demo plot saved to: {plot_path}")

    # For batch processing, uncomment:
    # batch_json = "/workspaces/pymdp/GIFT_Engine/batch_results/batch_results.json"
    # if Path(batch_json).exists():
    #     cohort_plot = plot_batch_learning(batch_json)
    #     print(f"Cohort plot saved to: {cohort_plot}")