#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
from pathlib import Path
import json
import argparse
from datetime import datetime

class MuscleAnalyzer:
    def __init__(self):
        self.muscle_mapping = {
            'quadriceps_femoris_left': 'Left Quadriceps Femoris',
            'quadriceps_femoris_right': 'Right Quadriceps Femoris',
            'thigh_posterior_compartment_left': 'Left Posterior Compartment (Hamstrings)',
            'thigh_posterior_compartment_right': 'Right Posterior Compartment (Hamstrings)',
            'sartorius_left': 'Left Sartorius',
            'sartorius_right': 'Right Sartorius',
            'thigh_medial_compartment_left': 'Left Medial Compartment (Gracilis, Adductors)',
            'thigh_medial_compartment_right': 'Right Medial Compartment (Gracilis, Adductors)'
        }
    
    def postprocess_mask(self, mask_array):
        mask_array = mask_array.astype(np.uint8)
        mask_array = ndi.binary_opening(mask_array, iterations=2).astype(np.uint8)
        mask_array = ndi.binary_closing(mask_array, iterations=2).astype(np.uint8)
        
        labeled, num_features = ndi.label(mask_array)
        if num_features > 1:
            sizes = ndi.sum(mask_array, labeled, index=range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask_array = (labeled == largest_label).astype(np.uint8)
        
        return mask_array
    
    def segment_muscles(self, water_image_path, output_dir):
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(str(water_image_path), str(output_dir), task='thigh_shoulder_muscles_mr')
        return output_dir
    
    def load_dixon_data(self, subject_path, stack):
        dixon_data = {}
        fat_path = subject_path / "ImageData" / f"{subject_path.name}_FAT" / f"{subject_path.name}_FAT_{stack}.nii"
        water_path = subject_path / "ImageData" / f"{subject_path.name}_WATER" / f"{subject_path.name}_WATER_{stack}.nii"
        
        if fat_path.exists():
            dixon_data['fat'] = sitk.ReadImage(str(fat_path))
        if water_path.exists():
            dixon_data['water'] = sitk.ReadImage(str(water_path))
        
        return dixon_data
    
    def calculate_muscle_composition(self, muscle_masks_dir, dixon_data):
        results = {}
        
        if 'fat' not in dixon_data or 'water' not in dixon_data:
            return results
        
        fat_img = dixon_data['fat']
        water_img = dixon_data['water']
        fat_array = sitk.GetArrayFromImage(fat_img)
        water_array = sitk.GetArrayFromImage(water_img)
        
        total_signal = fat_array + water_array
        uff_array = np.divide(fat_array, total_signal, out=np.zeros_like(fat_array), where=total_signal > 0)
        
        spacing = fat_img.GetSpacing()
        voxel_volume_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
        
        for muscle_file in Path(muscle_masks_dir).glob("*.nii.gz"):
            muscle_name = muscle_file.stem.replace(".nii", "")
            
            if muscle_name in self.muscle_mapping:
                mask_img = sitk.ReadImage(str(muscle_file))
                mask_array = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
                
                if mask_array.shape != uff_array.shape:
                    continue
                
                mask_array = self.postprocess_mask(mask_array)
                muscle_voxels = mask_array > 0
                n_voxels = np.sum(muscle_voxels)
                
                if n_voxels > 0:
                    muscle_volume_cm3 = n_voxels * voxel_volume_cm3
                    muscle_uff_values = uff_array[muscle_voxels]
                    mean_uff = np.mean(muscle_uff_values)
                    std_uff = np.std(muscle_uff_values)
                    median_uff = np.median(muscle_uff_values)
                    
                    mean_uff_percent = mean_uff * 100
                    lean_volume_cm3 = muscle_volume_cm3 * (1 - mean_uff)
                    fat_volume_cm3 = muscle_volume_cm3 * mean_uff
                    
                    results[self.muscle_mapping[muscle_name]] = {
                        'total_volume_cm3': round(muscle_volume_cm3, 2),
                        'lean_volume_cm3': round(lean_volume_cm3, 2),
                        'fat_volume_cm3': round(fat_volume_cm3, 2),
                        'mean_uncorrected_fat_fraction_percent': round(mean_uff_percent, 2),
                        'std_uncorrected_fat_fraction_percent': round(std_uff * 100, 2),
                        'median_uncorrected_fat_fraction_percent': round(median_uff * 100, 2),
                        'n_voxels': int(n_voxels)
                    }
        
        return results
    
    def analyze_subject(self, subject_path, stack="stack1", output_dir=None):
        subject_path = Path(subject_path)
        
        if output_dir is None:
            output_dir = Path("outputs") / f"{subject_path.name}_{stack}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dixon_data = self.load_dixon_data(subject_path, stack)
        if not dixon_data:
            return None
        
        water_image = subject_path / "ImageData" / f"{subject_path.name}_WATER" / f"{subject_path.name}_WATER_{stack}.nii"
        segmentation_dir = output_dir / "muscle_masks"
        
        self.segment_muscles(water_image, segmentation_dir)
        results = self.calculate_muscle_composition(segmentation_dir, dixon_data)
        
        results_file = output_dir / "muscle_composition.json"
        with open(results_file, 'w') as f:
            json.dump({
                'subject_id': subject_path.name,
                'stack': stack,
                'analysis_date': datetime.now().isoformat(),
                'muscle_composition': results
            }, f, indent=2)
        
        df_data = [{'muscle': muscle, **metrics} for muscle, metrics in results.items()]
        df = pd.DataFrame(df_data)
        csv_file = output_dir / "muscle_composition.csv"
        df.to_csv(csv_file, index=False)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MRI Muscle Health Quantifier")
    parser.add_argument("--subject", type=str, help="Path to subject directory")
    parser.add_argument("--stack", type=str, default="stack1", choices=["stack1", "stack2"])
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--both-stacks", action="store_true", help="Analyze both stack1 and stack2")
    
    args = parser.parse_args()
    analyzer = MuscleAnalyzer()
    
    if args.subject:
        subject_path = Path(args.subject)
    else:
        possible_paths = [Path("../data/raw"), Path("../../data/raw"), Path("data/raw"), Path(".")]
        subject_path = None
        for base_path in possible_paths:
            if base_path.exists():
                subjects = list(base_path.glob("HV*")) + list(base_path.glob("P*"))
                if subjects:
                    subject_path = subjects[0]
                    break
        
        if not subject_path:
            return
    
    if not subject_path.exists():
        return
    
    stacks_to_analyze = ["stack1", "stack2"] if args.both_stacks else [args.stack]
    
    for stack in stacks_to_analyze:
        output_dir = Path(args.output) if args.output else None
        results = analyzer.analyze_subject(subject_path, stack, output_dir)
        
        if results:
            total_vol = sum(m['total_volume_cm3'] for m in results.values())
            fat_fractions = [m['mean_uncorrected_fat_fraction_percent'] for m in results.values()]
            
            print(f"\n=== {stack.upper()} RESULTS ===")
            print(f"Total muscle volume: {total_vol:.1f} cm³")
            print(f"Mean UFF: {sum(fat_fractions)/len(fat_fractions):.1f}%")
            print(f"UFF range: {min(fat_fractions):.1f}% - {max(fat_fractions):.1f}%\n")
            
            for muscle, metrics in results.items():
                print(f"{muscle}: {metrics['mean_uncorrected_fat_fraction_percent']:.1f}% UFF, {metrics['total_volume_cm3']:.1f} cm³")

if __name__ == "__main__":
    main()
