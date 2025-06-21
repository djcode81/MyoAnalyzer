#!/usr/bin/env python3
"""
MRI Muscle Health Quantifier - Complete Pipeline
Combines TotalSegmentator muscle segmentation with Dixon fat-fraction analysis
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import json
import argparse
from datetime import datetime

class MuscleAnalyzer:
    def __init__(self):
        self.muscle_mapping = {
            'quadriceps_femoris_left': 'Left Quadriceps',
            'quadriceps_femoris_right': 'Right Quadriceps',
            'thigh_posterior_compartment_left': 'Left Hamstrings',
            'thigh_posterior_compartment_right': 'Right Hamstrings',
            'sartorius_left': 'Left Sartorius',
            'sartorius_right': 'Right Sartorius',
            'thigh_medial_compartment_left': 'Left Medial (Gracilis)',
            'thigh_medial_compartment_right': 'Right Medial (Gracilis)'
        }
    
    def segment_muscles(self, water_image_path, output_dir):
        """Run TotalSegmentator muscle segmentation"""
        print(f"Running TotalSegmentator on {water_image_path}")
        
        # Run TotalSegmentator
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(
            str(water_image_path), 
            str(output_dir), 
            task='thigh_shoulder_muscles_mr'
        )
        
        return output_dir
    
    def load_dixon_data(self, subject_path, stack):
        """Load Dixon fat and water images for proper PDFF calculation"""
        dixon_data = {}
        
        # Load fat image
        fat_path = subject_path / "ImageData" / f"{subject_path.name}_FAT" / f"{subject_path.name}_FAT_{stack}.nii"
        if fat_path.exists():
            dixon_data['fat'] = sitk.ReadImage(str(fat_path))
            print(f"Loaded fat image: {fat_path}")
        
        # Load water image
        water_path = subject_path / "ImageData" / f"{subject_path.name}_WATER" / f"{subject_path.name}_WATER_{stack}.nii"
        if water_path.exists():
            dixon_data['water'] = sitk.ReadImage(str(water_path))
            print(f"Loaded water image: {water_path}")
        
        return dixon_data
    
    def calculate_muscle_composition(self, muscle_masks_dir, dixon_data):
        """Calculate fat fraction for each muscle using proper PDFF formula"""
        results = {}
        
        if 'fat' not in dixon_data or 'water' not in dixon_data:
            print("ERROR: Missing fat or water images for PDFF calculation")
            return results
        
        # Load fat and water images
        fat_img = dixon_data['fat']
        water_img = dixon_data['water']
        fat_array = sitk.GetArrayFromImage(fat_img)
        water_array = sitk.GetArrayFromImage(water_img)
        
        # Calculate PDFF using proper formula: Fat / (Fat + Water)
        total_signal = fat_array + water_array
        pdff_array = np.divide(fat_array, total_signal, 
                              out=np.zeros_like(fat_array), 
                              where=total_signal > 0)
        
        # Get voxel volume for volume calculations
        spacing = fat_img.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
        
        # Process each muscle mask
        for muscle_file in Path(muscle_masks_dir).glob("*.nii.gz"):
            muscle_name = muscle_file.stem.replace(".nii", "")
            
            if muscle_name in self.muscle_mapping:
                print(f"Processing {self.muscle_mapping[muscle_name]}")
                
                # Load muscle mask
                mask_img = sitk.ReadImage(str(muscle_file))
                mask_array = sitk.GetArrayFromImage(mask_img)
                
                # Ensure same dimensions
                if mask_array.shape != pdff_array.shape:
                    print(f"WARNING: Shape mismatch for {muscle_name}")
                    continue
                
                # Calculate muscle composition
                muscle_voxels = mask_array > 0
                n_voxels = np.sum(muscle_voxels)
                
                if n_voxels > 0:
                    # Volume calculations
                    muscle_volume_cm3 = n_voxels * voxel_volume_cm3
                    
                    # PDFF statistics (only in muscle region)
                    muscle_pdff_values = pdff_array[muscle_voxels]
                    mean_pdff = np.mean(muscle_pdff_values)
                    std_pdff = np.std(muscle_pdff_values)
                    median_pdff = np.median(muscle_pdff_values)
                    
                    # Convert to percentages
                    mean_fat_percent = mean_pdff * 100
                    
                    # Calculate lean and fat volumes
                    lean_volume_cm3 = muscle_volume_cm3 * (1 - mean_pdff)
                    fat_volume_cm3 = muscle_volume_cm3 * mean_pdff
                    
                    results[self.muscle_mapping[muscle_name]] = {
                        'total_volume_cm3': round(muscle_volume_cm3, 2),
                        'lean_volume_cm3': round(lean_volume_cm3, 2),
                        'fat_volume_cm3': round(fat_volume_cm3, 2),
                        'mean_fat_fraction_percent': round(mean_fat_percent, 2),
                        'std_fat_fraction_percent': round(std_pdff * 100, 2),
                        'median_fat_fraction_percent': round(median_pdff * 100, 2),
                        'n_voxels': int(n_voxels)
                    }
        
        return results
    
    def analyze_subject(self, subject_path, stack="stack1", output_dir=None):
        """Complete analysis pipeline for one subject"""
        subject_path = Path(subject_path)
        
        if output_dir is None:
            output_dir = Path("outputs") / f"{subject_path.name}_{stack}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== Analyzing {subject_path.name} {stack} ===")
        
        # Step 1: Load Dixon data
        dixon_data = self.load_dixon_data(subject_path, stack)
        if not dixon_data:
            print("ERROR: Could not load Dixon data")
            return None
        
        # Step 2: Run muscle segmentation
        water_image = subject_path / "ImageData" / f"{subject_path.name}_WATER" / f"{subject_path.name}_WATER_{stack}.nii"
        segmentation_dir = output_dir / "muscle_masks"
        
        self.segment_muscles(water_image, segmentation_dir)
        
        # Step 3: Calculate muscle composition
        results = self.calculate_muscle_composition(segmentation_dir, dixon_data)
        
        # Step 4: Save results
        results_file = output_dir / "muscle_composition.json"
        with open(results_file, 'w') as f:
            json.dump({
                'subject_id': subject_path.name,
                'stack': stack,
                'analysis_date': datetime.now().isoformat(),
                'muscle_composition': results
            }, f, indent=2)
        
        # Create summary CSV
        df_data = []
        for muscle, metrics in results.items():
            row = {'muscle': muscle}
            row.update(metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = output_dir / "muscle_composition.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {output_dir}")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="MRI Muscle Health Quantifier - Automated Dixon fat-fraction analysis"
    )
    parser.add_argument(
        "--subject", 
        type=str, 
        help="Path to subject directory (e.g., /path/to/HV003_1)"
    )
    parser.add_argument(
        "--stack", 
        type=str, 
        default="stack1", 
        choices=["stack1", "stack2", "stack3"],
        help="Which stack to analyze (default: stack1)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory (default: outputs/SUBJECT_STACK)"
    )
    parser.add_argument(
        "--both-stacks", 
        action="store_true",
        help="Analyze both stack1 and stack2"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MuscleAnalyzer()
    
    # Determine subject path
    if args.subject:
        subject_path = Path(args.subject)
    else:
        # Look for subjects in standard locations
        possible_paths = [
            Path("../data/raw"),
            Path("../../data/raw"),
            Path("data/raw"),
            Path(".")
        ]
        
        subject_path = None
        for base_path in possible_paths:
            if base_path.exists():
                subjects = list(base_path.glob("HV*")) + list(base_path.glob("P*"))
                if subjects:
                    print(f"Available subjects in {base_path}:")
                    for s in sorted(subjects):
                        print(f"  {s.name}")
                    
                    # Use first available subject as default
                    subject_path = subjects[0]
                    print(f"\nUsing default subject: {subject_path.name}")
                    break
        
        if not subject_path:
            print("ERROR: No subject specified and no subjects found in standard locations")
            print("Usage: python muscle_analyzer.py --subject /path/to/subject")
            return
    
    if not subject_path.exists():
        print(f"ERROR: Subject path not found: {subject_path}")
        return
    
    # Analyze specified stacks
    stacks_to_analyze = ["stack1", "stack2"] if args.both_stacks else [args.stack]
    
    for stack in stacks_to_analyze:
        try:
            output_dir = Path(args.output) if args.output else None
            results = analyzer.analyze_subject(subject_path, stack, output_dir)
            
            if results:
                print(f"\n=== {stack.upper()} RESULTS ===")
                total_vol = sum(m['total_volume_cm3'] for m in results.values())
                fat_fractions = [m['mean_fat_fraction_percent'] for m in results.values()]
                
                print(f"Total muscle volume: {total_vol:.1f} cm³")
                print(f"Mean fat fraction: {sum(fat_fractions)/len(fat_fractions):.1f}%")
                print(f"Fat fraction range: {min(fat_fractions):.1f}% - {max(fat_fractions):.1f}%")
                print()
                
                for muscle, metrics in results.items():
                    print(f"{muscle}: {metrics['mean_fat_fraction_percent']:.1f}% fat, {metrics['total_volume_cm3']:.1f} cm³")
        except Exception as e:
            print(f"ERROR processing {stack}: {e}")

if __name__ == "__main__":
    main()
