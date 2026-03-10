#!/usr/bin/env python3
"""
GIFT ENGINE MASTER PIPELINE
Complete workflow: Load data → Process patients → Generate report
"""

import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Ensure all required packages are installed"""
    required_packages = ['matplotlib', 'numpy', 'mne']
    print("📦 Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ⚠️  Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
            print(f"   ✅ {package} installed")

def main():
    """Main pipeline execution"""
    print("\n" + "="*80)
    print("🧠 GIFT ENGINE: BATCH PROCESSING AND REPORTING PIPELINE")
    print("="*80)
    
    # Install dependencies
    install_dependencies()
    
    # STEP 1: Run batch processor
    print("\n" + "="*80)
    print("STEP 1: BATCH PROCESSING")
    print("="*80)
    
    try:
        from batch_processor import BatchProcessor, download_public_eeg_dataset
        
        # Download/prepare dataset
        patient_files = download_public_eeg_dataset("neuropsych_cohort", num_samples=30)
        
        # Process all patients
        processor = BatchProcessor(output_dir="batch_results")
        cohort_stats = processor.process_dataset(patient_files)
        
        # Save results
        processor.save_results()
        
        print(f"\n✅ Processed {processor.processed_patients} patients successfully")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        sys.exit(1)
    
    # STEP 2: Generate PDF report
    print("\n" + "="*80)
    print("STEP 2: GENERATING PDF REPORT")
    print("="*80)
    
    try:
        from pdf_report_generator import generate_pdf_report
        
        results_json = "batch_results/batch_results.json"
        output_pdf = "GIFT_Cohort_Report.pdf"
        
        if Path(results_json).exists():
            report_path = generate_pdf_report(results_json, output_pdf)
            print(f"\n✅ Report generated: {report_path}")
        else:
            print(f"⚠️  Results file not found: {results_json}")
            
    except ImportError as e:
        print(f"⚠️  PDF generation skipped (matplotlib may need installation): {e}")
        print("   Run: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  PDF generation warning: {str(e)}")
    
    # Display summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📊 Results Summary:")
    print(f"   • Patients processed: {processor.processed_patients}")
    print(f"   • Results file: batch_results/batch_results.json")
    print(f"   • PDF report: {output_pdf}")
    print(f"\n🔬 Next Steps:")
    print(f"   1. Review the PDF report for clinical insights")
    print(f"   2. Export JSON results for statistical analysis")
    print(f"   3. Use patient stratification for intervention planning")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
