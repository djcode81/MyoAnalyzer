#!/usr/bin/env python3
"""
Batch MRI Muscle Health Quantifier
Process all subjects in the dataset automatically
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import traceback
from tqdm import tqdm

# Import the MuscleAnalyzer class
from muscle_analyzer import MuscleAnalyzer

class BatchMuscleAnalyzer:
    def __init__(self, data_root="../data/raw"):
        self.data_root = Path(data_root)
        self.analyzer = MuscleAnalyzer()
        self.results_summary = []
        
    def get_all_subjects(self):
        """Get list of all complete subjects"""
        complete_subjects = []
        
        for subject_dir in self.data_root.glob("*"):
            if not subject_dir.is_dir() or not subject_dir.name.startswith(('HV', 'P')):
                continue
                
            # Check if subject has required data
            water1 = subject_dir / "ImageData" / f"{subject_dir.name}_WATER" / f"{subject_dir.name}_WATER_stack1.nii"
            fat1 = subject_dir / "ImageData" / f"{subject_dir.name}_FAT" / f"{subject_dir.name}_FAT_stack1.nii"
            
            if water1.exists() and fat1.exists():
                complete_subjects.append(subject_dir.name)
        
        return sorted(complete_subjects)
    
    def process_single_subject(self, subject_id):
        """Process one subject with both stacks"""
        subject_results = {
            'subject_id': subject_id,
            'processing_date': datetime.now().isoformat(),
            'status': 'success',
            'error_message': None,
            'stacks': {}
        }
        
        subject_path = self.data_root / subject_id
        print(f"\n{'='*60}")
        print(f"Processing {subject_id}")
        print(f"{'='*60}")
        
        # Process both stacks
        for stack in ["stack1", "stack2"]:
            try:
                print(f"\n--- {stack.upper()} ---")
                
                # Check if required files exist
                water_file = subject_path / "ImageData" / f"{subject_id}_WATER" / f"{subject_id}_WATER_{stack}.nii"
                fat_file = subject_path / "ImageData" / f"{subject_id}_FAT" / f"{subject_id}_FAT_{stack}.nii"
                
                if not (water_file.exists() and fat_file.exists()):
                    print(f"Missing Dixon files for {stack}, skipping...")
                    subject_results['stacks'][stack] = {'status': 'missing_files'}
                    continue
                
                # Run analysis
                output_dir = Path("outputs") / f"{subject_id}_{stack}"
                results = self.analyzer.analyze_subject(subject_path, stack, output_dir)
                
                if results:
                    subject_results['stacks'][stack] = {
                        'status': 'success',
                        'muscle_composition': results,
                        'summary_stats': self.calculate_summary_stats(results)
                    }
                    
                    # Add to summary for batch analysis
                    for muscle, metrics in results.items():
                        self.results_summary.append({
                            'subject_id': subject_id,
                            'stack': stack,
                            'muscle': muscle,
                            **metrics
                        })
                        
                    print(f"{stack} completed successfully - {len(results)} muscles analyzed")
                else:
                    subject_results['stacks'][stack] = {'status': 'analysis_failed'}
                    print(f"{stack} analysis failed")
                    
            except Exception as e:
                error_msg = f"Error processing {stack}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                subject_results['stacks'][stack] = {
                    'status': 'error',
                    'error_message': error_msg
                }
        
        return subject_results
    
    def calculate_summary_stats(self, muscle_results):
        """Calculate summary statistics for a subject"""
        fat_fractions = [m['mean_fat_fraction_percent'] for m in muscle_results.values()]
        total_volume = sum(m['total_volume_cm3'] for m in muscle_results.values())
        total_lean_volume = sum(m['lean_volume_cm3'] for m in muscle_results.values())
        
        return {
            'total_muscle_volume_cm3': round(total_volume, 2),
            'total_lean_volume_cm3': round(total_lean_volume, 2),
            'mean_fat_fraction_percent': round(np.mean(fat_fractions), 2),
            'std_fat_fraction_percent': round(np.std(fat_fractions), 2),
            'min_fat_fraction_percent': round(min(fat_fractions), 2),
            'max_fat_fraction_percent': round(max(fat_fractions), 2),
            'n_muscles_analyzed': len(muscle_results)
        }
    
    def process_all_subjects(self, subject_list=None):
        """Process all subjects in batch"""
        if subject_list is None:
            subject_list = self.get_all_subjects()
        
        print(f"Found {len(subject_list)} complete subjects to process:")
        print(subject_list)
        
        # Create outputs directory
        Path("outputs").mkdir(exist_ok=True)
        
        all_results = {}
        failed_subjects = []
        
        # Process each subject
        for subject_id in tqdm(subject_list, desc="Processing subjects"):
            try:
                subject_results = self.process_single_subject(subject_id)
                all_results[subject_id] = subject_results
                
                # Check if any stack succeeded
                success_count = sum(1 for stack_result in subject_results['stacks'].values() 
                                  if stack_result.get('status') == 'success')
                if success_count == 0:
                    failed_subjects.append(subject_id)
                    
            except Exception as e:
                print(f"CRITICAL ERROR processing {subject_id}: {str(e)}")
                failed_subjects.append(subject_id)
                all_results[subject_id] = {
                    'subject_id': subject_id,
                    'status': 'critical_error',
                    'error_message': str(e)
                }
        
        # Save comprehensive results
        batch_results = {
            'processing_date': datetime.now().isoformat(),
            'total_subjects': len(subject_list),
            'successful_subjects': len(subject_list) - len(failed_subjects),
            'failed_subjects': failed_subjects,
            'subject_results': all_results
        }
        
        # Save batch results
        with open("outputs/batch_results.json", 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Save summary CSV
        if self.results_summary:
            df = pd.DataFrame(self.results_summary)
            df.to_csv("outputs/all_subjects_summary.csv", index=False)
            
            # Create subject-level summary
            subject_summary = []
            for subject_id, subject_data in all_results.items():
                if subject_data.get('status') == 'success':
                    for stack, stack_data in subject_data.get('stacks', {}).items():
                        if stack_data.get('status') == 'success':
                            summary_stats = stack_data.get('summary_stats', {})
                            subject_summary.append({
                                'subject_id': subject_id,
                                'stack': stack,
                                **summary_stats
                            })
            
            if subject_summary:
                df_subjects = pd.DataFrame(subject_summary)
                df_subjects.to_csv("outputs/subject_level_summary.csv", index=False)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total subjects processed: {len(subject_list)}")
        print(f"Successful subjects: {len(subject_list) - len(failed_subjects)}")
        print(f"Failed subjects: {len(failed_subjects)}")
        if failed_subjects:
            print(f"Failed: {failed_subjects}")
        print(f"Total muscle measurements: {len(self.results_summary)}")
        print(f"Results saved to: outputs/")
        
        return batch_results

def main():
    # Initialize batch processor
    batch_analyzer = BatchMuscleAnalyzer()
    
    # Process all subjects
    results = batch_analyzer.process_all_subjects()
    
    # Print quick statistics
    if batch_analyzer.results_summary:
        df = pd.DataFrame(batch_analyzer.results_summary)
        print(f"\nQuick Statistics:")
        print(f"Fat fraction range: {df['mean_fat_fraction_percent'].min():.1f}% - {df['mean_fat_fraction_percent'].max():.1f}%")
        print(f"Average fat fraction: {df['mean_fat_fraction_percent'].mean():.1f}%")
        print(f"Average muscle volume: {df['total_volume_cm3'].mean():.1f} cm³")

if __name__ == "__main__":
    main()
