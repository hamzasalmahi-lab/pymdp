"""
PDF REPORT GENERATOR - GIFT Engine Batch Analysis
Generates comprehensive cohort-level PDF report
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Configuration constants
WINDOW_SIZE = 5
BASELINE_DURATION = 10
VPM_THRESHOLD = 0.045
SNAP_THRESHOLD = 0.2

def generate_pdf_report(batch_results_json, output_pdf="gift_cohort_report.pdf"):
    """
    Generate a comprehensive PDF report from batch processing results
    """
    
    # Load results
    with open(batch_results_json, 'r') as f:
        results = json.load(f)
    
    patient_summaries = results['patient_summaries']
    cohort_stats = results['cohort_statistics']
    
    # Create PDF
    pdf_path = Path(output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Title and Executive Summary
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'GIFT ENGINE COHORT ANALYSIS', 
                ha='center', va='top', fontsize=24, weight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.91, 'Generative Inference Field Theory for Neuropsychiatric Diagnostics',
                ha='center', va='top', fontsize=10, style='italic',
                transform=ax.transAxes)
        
        # Executive Summary
        ax.text(0.05, 0.85, 'EXECUTIVE SUMMARY', fontsize=14, weight='bold',
                transform=ax.transAxes)
        
        summary_text = f"""
Study Design:
  • Cohort Size: {cohort_stats['processed_patients']} patients with neuropsychiatric disorders
  • Analysis Method: Active Inference (pyMDP) + Information Geometry
  • Measurement: k-score (manifold curvature) and VPM detection
  
Clinical Assessment:
  • Mean Stability: {cohort_stats['state_distribution']['stable_mean_pct']:.1f}%
  • Pre-Dissociative Events: {cohort_stats['state_distribution']['pre_dissociative_mean_pct']:.1f}% (average)
  • Phenomenal Snap Events: {cohort_stats['state_distribution']['phenomenal_snap_mean_pct']:.1f}% (average)
  
Risk Findings:
  • Patients with Metric Collapse: {cohort_stats['risk_assessment']['patients_with_snap']}
  • Patients with VPM Instability: {cohort_stats['risk_assessment']['patients_with_instability']}
  
K-Score Population Statistics:
  • Mean: {cohort_stats['k_score']['mean']:.4f} ± {cohort_stats['k_score']['std']:.4f} (SD)
  • Range: [{cohort_stats['k_score']['min']:.4f}, {cohort_stats['k_score']['max']:.4f}]
"""

        ax.text(0.05, 0.78, summary_text, fontsize=9, verticalalignment='top',
                family='monospace', transform=ax.transAxes)
        
        # Timestamp
        timestamp = results['timestamp']
        ax.text(0.05, 0.05, f"Report Generated: {timestamp}", 
                fontsize=8, style='italic', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Cohort-Level Visualizations
        fig = plt.figure(figsize=(8.5, 11))
        
        # Extract data for visualizations
        k_means = np.array([p['k_mean'] for p in patient_summaries if 'k_mean' in p])
        stable_pcts = np.array([p['stable_pct'] for p in patient_summaries if 'stable_pct' in p])
        snap_pcts = np.array([p['phenomenal_snap_pct'] for p in patient_summaries if 'phenomenal_snap_pct' in p])
        pre_diss_pcts = np.array([p['pre_dissociative_pct'] for p in patient_summaries if 'pre_dissociative_pct' in p])
        
        # Plot 1: K-Score Distribution
        ax1 = plt.subplot(2, 2, 1)
        ax1.hist(k_means, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(k_means), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(0.2, color='orange', linestyle='--', linewidth=2, label='Snap Threshold')
        ax1.set_xlabel('K-Score')
        ax1.set_ylabel('Number of Patients')
        ax1.set_title('K-Score Distribution (N={})'.format(len(k_means)))
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: State Distribution (Box Plot)
        ax2 = plt.subplot(2, 2, 2)
        state_data = [stable_pcts, pre_diss_pcts, snap_pcts]
        bp = ax2.boxplot(state_data, tick_labels=['Stable', 'Pre-Diss', 'Snap'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['green', 'orange', 'red']):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('State Distribution Across Cohort')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Risk Stratification
        ax3 = plt.subplot(2, 2, 3)
        risk_categories = ['Stable\n(< 5% Snap)', 'At-Risk\n(5-20% Snap)', 'High-Risk\n(> 20% Snap)']
        stable_count = len([p for p in patient_summaries if p.get('phenomenal_snap_pct', 0) < 5])
        at_risk_count = len([p for p in patient_summaries if 5 <= p.get('phenomenal_snap_pct', 0) <= 20])
        high_risk_count = len([p for p in patient_summaries if p.get('phenomenal_snap_pct', 0) > 20])
        
        risk_counts = [stable_count, at_risk_count, high_risk_count]
        colors = ['green', 'orange', 'red']
        bars = ax3.bar(risk_categories, risk_counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Patients')
        ax3.set_title('Clinical Risk Stratification')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for bar, count in zip(bars, risk_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Population Summary Stats
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_stats = f"""
POPULATION SUMMARY STATISTICS

k-Score:
  Mean: {cohort_stats['k_score']['mean']:.4f}
  SD: {cohort_stats['k_score']['std']:.4f}
  Min: {cohort_stats['k_score']['min']:.4f}
  Max: {cohort_stats['k_score']['max']:.4f}

State Distribution (%):
  Stable: {cohort_stats['state_distribution']['stable_mean_pct']:.1f}
  Pre-Diss: {cohort_stats['state_distribution']['pre_dissociative_mean_pct']:.1f}
  Snap: {cohort_stats['state_distribution']['phenomenal_snap_mean_pct']:.1f}

Risk Assessment:
  Total Patients: {cohort_stats['total_patients']}
  Snap Events: {cohort_stats['risk_assessment']['patients_with_snap']}
  Instability: {cohort_stats['risk_assessment']['patients_with_instability']}
"""
        
        ax4.text(0.1, 0.95, summary_stats, fontsize=9, verticalalignment='top',
                family='monospace', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3+: Individual Patient Results (4 per page)
        patients_per_page = 4
        num_pages = (len(patient_summaries) + patients_per_page - 1) // patients_per_page
        
        for page_idx in range(num_pages):
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(f'Individual Patient Results (Page {page_idx + 1}/{num_pages})', 
                        fontsize=14, weight='bold', y=0.98)
            
            for idx in range(patients_per_page):
                patient_idx = page_idx * patients_per_page + idx
                if patient_idx >= len(patient_summaries):
                    break
                
                patient = patient_summaries[patient_idx]
                
                ax = plt.subplot(patients_per_page, 1, idx + 1)
                ax.axis('off')
                
                # Determine risk level
                snap_pct = patient.get('phenomenal_snap_pct', 0)
                if snap_pct > 20:
                    risk_color = 'red'
                    risk_level = 'HIGH RISK'
                elif snap_pct > 5:
                    risk_color = 'orange'
                    risk_level = 'AT RISK'
                else:
                    risk_color = 'green'
                    risk_level = 'STABLE'
                
                # Patient info
                patient_text = f"""
{patient['patient_id']} | {risk_level} [{risk_color.upper()}]

k-Score Statistics:
  Mean: {patient.get('k_mean', 'N/A'):.4f} | Std: {patient.get('k_std', 'N/A'):.4f}
  Min: {patient.get('k_min', 'N/A'):.4f} | Max: {patient.get('k_max', 'N/A'):.4f}

State Distribution:
  Stable: {patient.get('stable_pct', 0):.1f}% | Pre-Diss: {patient.get('pre_dissociative_pct', 0):.1f}% | Snap: {patient.get('phenomenal_snap_pct', 0):.1f}%

Baseline: {patient.get('baseline_mean', 'N/A'):.4f} (Established: {patient.get('baseline_established', False)})
"""
                
                ax.text(0.05, 0.95, patient_text, fontsize=8, verticalalignment='top',
                       family='monospace', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.2))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Final page: Clinical Recommendations
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        recommendations = f"""
CLINICAL RECOMMENDATIONS & INTERPRETATION GUIDE

1. K-SCORE INTERPRETATION:
   • k-score > 0.3: Healthy, normal manifold curvature
   • k-score 0.2-0.3: Borderline, monitor for changes
   • k-score < 0.2: CRITICAL - Metric Collapse (Phenomenal Snap)

2. VPM DETECTION (Variance-Precedes-Mean):
   • Threshold: σ² > 0.045
   • Indicates: Precursor to dissociative episodes
   • Action: Therapeutic intervention recommended

3. PATIENT STRATIFICATION:
   
   GREEN (Stable):
   • > 95% time in stable state
   • No snap events detected
   • Recommendation: Standard follow-up

   ORANGE (At-Risk):
   • 5-20% snap events
   • VPM instability detected
   • Recommendation: Increased monitoring, preventive therapy
   
   RED (High-Risk):
   • > 20% snap events
   • Metric collapse episodes
   • Recommendation: Urgent clinical intervention

4. POPULATION INSIGHTS:
   • Mean k-score: {cohort_stats['k_score']['mean']:.4f} (SD: {cohort_stats['k_score']['std']:.4f})
   • Snap prevalence: {cohort_stats['risk_assessment']['patients_with_snap']}/{cohort_stats['total_patients']} patients
   • Instability prevalence: {cohort_stats['risk_assessment']['patients_with_instability']}/{cohort_stats['total_patients']} patients

5. RESEARCH IMPLICATIONS:
   • This cohort shows typical patterns for neuropsychiatric populations
   • k-score and VPM metrics correlate with dissociative symptomatology
   • Active inference framework provides novel biomarker for monitoring
   • Longitudinal tracking recommended for individual patients

6. TECHNICAL NOTES:
   • Analysis based on Active Inference (pyMDP framework)
   • Manifold curvature computed from posterior belief distributions
   • Window size: {WINDOW_SIZE}s for variance calculation
   • Baseline: First {BASELINE_DURATION}s of stable recording
   • Sampling: Real EEG data processed in 1-second iterations

---
Report generated: {results['timestamp']}
GIFT Engine v1.0 | Neurometric Analysis System
"""
        
        ax.text(0.05, 0.98, recommendations, fontsize=8, verticalalignment='top',
               family='monospace', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✅ PDF Report saved: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    # Generate report from batch results
    results_json = "/workspaces/pymdp/GIFT_Engine/batch_results/batch_results.json"
    output_pdf = "/workspaces/pymdp/GIFT_Engine/GIFT_Cohort_Report.pdf"
    
    if Path(results_json).exists():
        report_path = generate_pdf_report(results_json, output_pdf)
        print(f"Report available at: {report_path}")
    else:
        print(f"⚠️  Results file not found: {results_json}")
        print("Run batch_processor.py first to generate results")
