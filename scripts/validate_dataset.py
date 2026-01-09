"""
Dataset validation and statistics script.

This script validates the generated dataset and provides detailed statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict


class DatasetValidator:
    """Validate and analyze dataset."""
    
    def __init__(self, dataset_dir: str):
        """Initialize validator.
        
        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.splits = ['train', 'val', 'test']
        
    def validate_sample(self, sample_dir: Path) -> Dict[str, any]:
        """Validate a single sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check metadata
        metadata_path = sample_dir / 'metadata.json'
        if not metadata_path.exists():
            issues.append('Missing metadata.json')
            return {'valid': False, 'issues': issues}
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            issues.append('Invalid metadata.json format')
            return {'valid': False, 'issues': issues}
            
        # Check transforms
        transforms_path = sample_dir / 'transforms.json'
        if not transforms_path.exists():
            issues.append('Missing transforms.json')
        else:
            try:
                with open(transforms_path, 'r') as f:
                    transforms = json.load(f)
                    
                # Validate transform structure
                if 'parts' not in transforms:
                    issues.append('transforms.json missing "parts" field')
                else:
                    for part in transforms['parts']:
                        if 'translation' not in part or len(part['translation']) != 3:
                            issues.append(f'Invalid translation in part {part.get("part_name", "unknown")}')
                        if 'rotation' not in part or len(part['rotation']) != 4:
                            issues.append(f'Invalid rotation in part {part.get("part_name", "unknown")}')
                        if 'scale' not in part or len(part['scale']) != 3:
                            issues.append(f'Invalid scale in part {part.get("part_name", "unknown")}')
                            
            except json.JSONDecodeError:
                issues.append('Invalid transforms.json format')
                
        # Check for image files
        image_files = list(sample_dir.glob('*.png')) + list(sample_dir.glob('*.jpg'))
        if not image_files:
            issues.append('No image files found')
            
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_images': len(image_files)
        }
        
    def validate_split(self, split: str) -> Dict:
        """Validate a dataset split.
        
        Args:
            split: Split name (train/val/test)
            
        Returns:
            Dictionary with validation results
        """
        split_dir = self.dataset_dir / split
        if not split_dir.exists():
            return {
                'exists': False,
                'num_samples': 0,
                'valid_samples': 0,
                'issues': []
            }
            
        samples = sorted(split_dir.glob('sample_*'))
        valid_count = 0
        all_issues = []
        
        for sample_dir in samples:
            result = self.validate_sample(sample_dir)
            if result['valid']:
                valid_count += 1
            else:
                all_issues.append({
                    'sample': sample_dir.name,
                    'issues': result['issues']
                })
                
        return {
            'exists': True,
            'num_samples': len(samples),
            'valid_samples': valid_count,
            'invalid_samples': len(samples) - valid_count,
            'issues': all_issues
        }
        
    def validate_dataset(self) -> Dict:
        """Validate entire dataset.
        
        Returns:
            Dictionary with validation results for all splits
        """
        results = {}
        
        for split in self.splits:
            print(f"\nValidating {split} split...")
            results[split] = self.validate_split(split)
            
            if results[split]['exists']:
                print(f"  Total samples: {results[split]['num_samples']}")
                print(f"  Valid samples: {results[split]['valid_samples']}")
                print(f"  Invalid samples: {results[split]['invalid_samples']}")
                
                if results[split]['issues']:
                    print(f"  Issues found in {len(results[split]['issues'])} samples")
                    # Show first 5 issues
                    for issue in results[split]['issues'][:5]:
                        print(f"    - {issue['sample']}: {', '.join(issue['issues'])}")
            else:
                print(f"  Split directory not found")
                
        return results
        
    def compute_statistics(self) -> Dict:
        """Compute dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'splits': {},
            'transforms': {
                'translation': {'min': [], 'max': [], 'mean': [], 'std': []},
                'rotation': {'min': [], 'max': [], 'mean': [], 'std': []},
                'scale': {'min': [], 'max': [], 'mean': [], 'std': []}
            }
        }
        
        all_translations = []
        all_rotations = []
        all_scales = []
        
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                continue
                
            samples = list(split_dir.glob('sample_*'))
            stats['splits'][split] = {'num_samples': len(samples)}
            
            # Collect transform statistics
            for sample_dir in samples:
                transforms_path = sample_dir / 'transforms.json'
                if not transforms_path.exists():
                    continue
                    
                try:
                    with open(transforms_path, 'r') as f:
                        transforms = json.load(f)
                        
                    for part in transforms.get('parts', []):
                        if 'translation' in part:
                            all_translations.append(part['translation'])
                        if 'rotation' in part:
                            all_rotations.append(part['rotation'])
                        if 'scale' in part:
                            all_scales.append(part['scale'])
                except:
                    continue
                    
        # Compute statistics
        if all_translations:
            translations = np.array(all_translations)
            stats['transforms']['translation'] = {
                'min': translations.min(axis=0).tolist(),
                'max': translations.max(axis=0).tolist(),
                'mean': translations.mean(axis=0).tolist(),
                'std': translations.std(axis=0).tolist()
            }
            
        if all_rotations:
            rotations = np.array(all_rotations)
            stats['transforms']['rotation'] = {
                'min': rotations.min(axis=0).tolist(),
                'max': rotations.max(axis=0).tolist(),
                'mean': rotations.mean(axis=0).tolist(),
                'std': rotations.std(axis=0).tolist()
            }
            
        if all_scales:
            scales = np.array(all_scales)
            stats['transforms']['scale'] = {
                'min': scales.min(axis=0).tolist(),
                'max': scales.max(axis=0).tolist(),
                'mean': scales.mean(axis=0).tolist(),
                'std': scales.std(axis=0).tolist()
            }
            
        return stats
        
    def print_statistics(self, stats: Dict):
        """Print statistics in a readable format.
        
        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print("\nSplit sizes:")
        for split, split_stats in stats['splits'].items():
            print(f"  {split}: {split_stats['num_samples']} samples")
            
        print("\nTransformation statistics:")
        for transform_type, transform_stats in stats['transforms'].items():
            if transform_stats.get('mean'):
                print(f"\n  {transform_type.capitalize()}:")
                print(f"    Mean: {[f'{x:.4f}' for x in transform_stats['mean']]}")
                print(f"    Std:  {[f'{x:.4f}' for x in transform_stats['std']]}")
                print(f"    Min:  {[f'{x:.4f}' for x in transform_stats['min']]}")
                print(f"    Max:  {[f'{x:.4f}' for x in transform_stats['max']]}")
                
    def save_report(self, validation_results: Dict, stats: Dict, output_path: str):
        """Save validation report to file.
        
        Args:
            validation_results: Validation results
            stats: Statistics
            output_path: Path to save report
        """
        report = {
            'validation': validation_results,
            'statistics': stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nReport saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Validate and analyze dataset')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='dataset_report.json',
                       help='Path to save validation report')
    
    args = parser.parse_args()
    
    # Create validator
    validator = DatasetValidator(args.dataset_dir)
    
    # Validate dataset
    print("Validating dataset...")
    validation_results = validator.validate_dataset()
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = validator.compute_statistics()
    validator.print_statistics(stats)
    
    # Save report
    validator.save_report(validation_results, stats, args.output)
    
    # Summary
    total_samples = sum(r['num_samples'] for r in validation_results.values() if r['exists'])
    total_valid = sum(r['valid_samples'] for r in validation_results.values() if r['exists'])
    
    print("\n" + "="*60)
    print(f"SUMMARY: {total_valid}/{total_samples} samples are valid")
    print("="*60)


if __name__ == '__main__':
    main()
