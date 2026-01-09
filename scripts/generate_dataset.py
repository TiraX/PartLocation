"""
Dataset generation script.

This script orchestrates the complete dataset generation process:
1. Load 3D models
2. Render multi-view images
3. Extract transformation parameters
4. Organize into train/val/test splits
5. Apply data augmentation

Usage:
    python generate_dataset.py --config configs/dataset_config.yaml
"""

import argparse
import subprocess
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm


class DatasetGenerator:
    """Generate training dataset from 3D models."""
    
    def __init__(self, config: dict):
        """Initialize dataset generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.blender_path = config.get('blender_path', 'blender')
        self.output_dir = Path(config['output_dir'])
        self.models_dir = Path(config['models_dir'])
        
        # Dataset splits
        self.train_ratio = config.get('train_ratio', 0.7)
        self.val_ratio = config.get('val_ratio', 0.15)
        self.test_ratio = config.get('test_ratio', 0.15)
        
        # Rendering settings
        self.views = config.get('views', ['front', 'back', 'left', 'right'])
        self.combine_mode = config.get('combine_mode', 'grid')
        self.resolution = config.get('resolution', 512)
        
        # Data augmentation
        self.augmentation = config.get('augmentation', {})
        self.elevation_range = self.augmentation.get('elevation_range', [-10, 10])
        self.azimuth_range = self.augmentation.get('azimuth_range', [-10, 10])
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directory structure."""
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
    def get_model_files(self) -> List[Path]:
        """Get list of 3D model files.
        
        Returns:
            List of model file paths
        """
        model_extensions = ['.blend', '.obj', '.fbx', '.gltf', '.glb']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(self.models_dir.glob(f'**/*{ext}'))
            
        return sorted(model_files)
        
    def split_dataset(self, num_samples: int) -> Dict[str, List[int]]:
        """Split dataset into train/val/test.
        
        Args:
            num_samples: Total number of samples
            
        Returns:
            Dictionary mapping split names to sample indices
        """
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        
        splits = {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }
        
        return splits
        
    def generate_augmentation_params(self) -> Dict[str, float]:
        """Generate random augmentation parameters.
        
        Returns:
            Dictionary with augmentation parameters
        """
        params = {}
        
        if self.augmentation.get('enable_elevation', True):
            params['elevation'] = random.uniform(*self.elevation_range)
        else:
            params['elevation'] = 0.0
            
        if self.augmentation.get('enable_azimuth', True):
            params['azimuth_offset'] = random.uniform(*self.azimuth_range)
        else:
            params['azimuth_offset'] = 0.0
            
        return params
        
    def render_sample(self, model_path: Path, sample_id: str, output_dir: Path,
                     augmentation_params: Dict[str, float]) -> Dict[str, Path]:
        """Render a single sample.
        
        Args:
            model_path: Path to 3D model file
            sample_id: Sample identifier
            output_dir: Output directory for this sample
            augmentation_params: Augmentation parameters
            
        Returns:
            Dictionary with paths to rendered images
        """
        # Create sample directory
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare Blender command
        render_script = Path(__file__).parent / 'blender_render.py'
        
        cmd = [
            self.blender_path,
            str(model_path),
            '--background',
            '--python', str(render_script),
            '--',
            str(sample_dir),
            str(self.resolution),
            ','.join(self.views),
            self.combine_mode,
            str(augmentation_params['elevation']),
            str(augmentation_params['azimuth_offset'])
        ]
        
        # Run Blender rendering
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Error rendering {sample_id}: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Timeout rendering {sample_id}")
            return None
            
        return {'sample_dir': sample_dir}
        
    def extract_transforms(self, model_path: Path, sample_id: str, 
                          output_dir: Path) -> Dict:
        """Extract transformation parameters.
        
        Args:
            model_path: Path to 3D model file
            sample_id: Sample identifier
            output_dir: Output directory
            
        Returns:
            Dictionary with transformation data
        """
        sample_dir = output_dir / sample_id
        transform_path = sample_dir / 'transforms.json'
        
        # Prepare Blender command
        extract_script = Path(__file__).parent / 'extract_transforms.py'
        
        # Note: You need to specify whole_name and part_names based on your model
        # This is a placeholder - adjust based on your model structure
        cmd = [
            self.blender_path,
            str(model_path),
            '--background',
            '--python', str(extract_script),
            '--',
            'whole_object',  # Replace with actual whole object name
            str(transform_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"Error extracting transforms for {sample_id}: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"Timeout extracting transforms for {sample_id}")
            return None
            
        # Load and return transforms
        if transform_path.exists():
            with open(transform_path, 'r') as f:
                return json.load(f)
        return None
        
    def generate_sample(self, model_path: Path, sample_idx: int, 
                       split: str) -> bool:
        """Generate a single dataset sample.
        
        Args:
            model_path: Path to 3D model file
            sample_idx: Sample index
            split: Dataset split (train/val/test)
            
        Returns:
            True if successful, False otherwise
        """
        sample_id = f"sample_{sample_idx:06d}"
        output_dir = self.output_dir / split
        
        # Generate augmentation parameters
        aug_params = self.generate_augmentation_params()
        
        # Render images
        render_result = self.render_sample(model_path, sample_id, output_dir, aug_params)
        if render_result is None:
            return False
            
        # Extract transforms
        transforms = self.extract_transforms(model_path, sample_id, output_dir)
        if transforms is None:
            return False
            
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'model_path': str(model_path),
            'split': split,
            'views': self.views,
            'augmentation': aug_params,
            'transforms': transforms
        }
        
        metadata_path = output_dir / sample_id / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return True
        
    def generate_dataset(self, num_samples: int):
        """Generate complete dataset.
        
        Args:
            num_samples: Total number of samples to generate
        """
        print(f"Generating dataset with {num_samples} samples...")
        
        # Get model files
        model_files = self.get_model_files()
        if not model_files:
            raise ValueError(f"No model files found in {self.models_dir}")
            
        print(f"Found {len(model_files)} model files")
        
        # Split dataset
        splits = self.split_dataset(num_samples)
        print(f"Dataset splits: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        # Generate samples
        sample_idx = 0
        for split_name, indices in splits.items():
            print(f"\nGenerating {split_name} split...")
            
            success_count = 0
            for idx in tqdm(indices, desc=split_name):
                # Cycle through model files
                model_path = model_files[idx % len(model_files)]
                
                if self.generate_sample(model_path, sample_idx, split_name):
                    success_count += 1
                    
                sample_idx += 1
                
            print(f"{split_name}: {success_count}/{len(indices)} samples generated successfully")
            
        # Generate dataset statistics
        self.generate_statistics()
        
    def generate_statistics(self):
        """Generate and save dataset statistics."""
        stats = {
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            samples = list(split_dir.glob('sample_*'))
            
            stats['splits'][split] = {
                'num_samples': len(samples),
                'samples': [s.name for s in samples]
            }
            
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"\nDataset statistics saved to {stats_path}")
        
    def validate_dataset(self):
        """Validate generated dataset."""
        print("\nValidating dataset...")
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            samples = list(split_dir.glob('sample_*'))
            
            for sample_dir in samples:
                # Check for required files
                required_files = ['metadata.json', 'transforms.json']
                
                for filename in required_files:
                    filepath = sample_dir / filename
                    if not filepath.exists():
                        issues.append(f"Missing {filename} in {sample_dir.name}")
                        
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
        else:
            print("Dataset validation passed!")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate dataset from 3D models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to generate (overrides config)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing dataset')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override num_samples if specified
    if args.num_samples is not None:
        config['num_samples'] = args.num_samples
        
    # Create generator
    generator = DatasetGenerator(config)
    
    if args.validate_only:
        generator.validate_dataset()
    else:
        num_samples = config.get('num_samples', 1000)
        generator.generate_dataset(num_samples)
        generator.validate_dataset()


if __name__ == '__main__':
    main()
