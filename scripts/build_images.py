"""
Build images from processed FBX models.

This script renders multi-view images for whole models and parts,
and optionally validates the assembly by comparing rendered assembled parts
with the whole model.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Import blender_render if running in Blender
from .blender_render import BlenderRenderer, compare_images
BLENDER_AVAILABLE = True

class ImageBuilder:
    """Build images from processed FBX models."""
    
    def __init__(self, data_dir: str, images_dir: str, 
                 resolution: int = 512, samples: int = 64,
                 enable_validation: bool = False):
        """
        Initialize image builder.
        
        Args:
            data_dir: Directory containing processed FBX files
            images_dir: Output directory for images
            resolution: Image resolution (default: 512)
            samples: Number of render samples (default: 64)
            enable_validation: Whether to validate assembly (default: False)
        """
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.resolution = resolution
        self.samples = samples
        self.enable_validation = enable_validation
        
        # Create images directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize renderer
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender is required for rendering. Please run this script with Blender.")
        
        self.renderer = BlenderRenderer(resolution=resolution, samples=samples)
    
    def find_samples(self) -> List[Path]:
        """
        Find all sample directories in data_dir.
        
        Returns:
            List of sample directory paths
        """
        samples = []
        if not self.data_dir.exists():
            return samples
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if it contains a whole model FBX
                whole_fbx = list(item.glob("*-whole.fbx"))
                if whole_fbx:
                    samples.append(item)
        
        return sorted(samples)
        
    def render_sample(self, sample_dir: Path) -> bool:
        """
        Render images for a single sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            True if rendering successful
        """
        sample_name = sample_dir.name
        print(f"\nProcessing sample: {sample_name}")
        print("="*60)
        
        # Create output directory for this sample
        sample_images_dir = self.images_dir / sample_name
        sample_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Find whole model FBX
        whole_fbx_list = list(sample_dir.glob("*-whole.fbx"))
        if not whole_fbx_list:
            print(f"  ERROR: No whole model FBX found")
            return False
        
        whole_fbx = whole_fbx_list[0]
        
        # Render whole model (4 views)
        print(f"\n[1/3] Rendering WHOLE model")
        print(f"  Input:  {whole_fbx.name}")
        print(f"  Output: {sample_name}/{whole_fbx.stem}_{{front,back,left,right}}.png")
        
        if not self.renderer.render_model(str(whole_fbx), str(sample_images_dir)):
            print(f"  ERROR: Failed to render whole model")
            return False
        
        # Find and render parts
        part_fbx_files = [f for f in sample_dir.glob("*.fbx") 
                         if not f.name.endswith("-whole.fbx")]
        
        print(f"\n[2/3] Rendering PARTS ({len(part_fbx_files)} parts found)")
        
        part_data_map = {}
        for idx, part_fbx in enumerate(part_fbx_files, 1):
            part_name = part_fbx.stem.replace(f"{sample_name}-", "")
            
            print(f"\n  Part [{idx}/{len(part_fbx_files)}]: {part_name}")
            print(f"    Input:  {part_fbx.name}")
            print(f"    Output: {sample_name}/{part_fbx.stem}_{{front,back,left,right}}.png")
            
            if self.renderer.render_model(str(part_fbx), str(sample_images_dir)):
                # Load part data for validation
                part_json = sample_dir / f"{sample_name}-{part_name}.json"
                if part_json.exists():
                    with open(part_json, 'r', encoding='utf-8') as f:
                        part_data = json.load(f)
                        part_data_map[part_name] = {
                            'fbx_path': str(part_fbx),
                            'translation': part_data['translation'],
                            'rotation': part_data['rotation'],
                            'scale': part_data['scale']
                        }
            else:
                print(f"    ERROR: Failed to render part {part_name}")
        
        # Validation: render assembled parts
        if self.enable_validation and len(part_data_map) > 0:
            print(f"\n[3/3] VALIDATION: Assembling parts")
            
            # Prepare part FBX paths and transform parameters
            part_fbx_list = []
            transform_params_list = []
            
            for part_name, part_info in part_data_map.items():
                part_fbx_list.append(part_info['fbx_path'])
                transform_params_list.append({
                    'translation': part_info['translation'],
                    'rotation': part_info['rotation'],
                    'scale': part_info['scale']
                })
            
            # Render assembled parts
            validation_name = f"{sample_name}-validation"
            print(f"  Output: {sample_name}/{validation_name}_{{front,back,left,right}}.png")
            
            if not self.renderer.render_assembled_parts(
                part_fbx_list,
                transform_params_list,
                str(sample_images_dir),
                validation_name
            ):
                print(f"  ERROR: Failed to render assembled parts for validation")
        else:
            print(f"\n[3/3] VALIDATION: Skipped (validation disabled or no parts)")
        
        print("="*60)
        return True
        
    def build_all(self) -> Dict[str, int]:
        """
        Build images for all samples.
        
        Returns:
            Dictionary with statistics: 'total', 'success', 'failed'
        """
        samples = self.find_samples()
        
        if not samples:
            print(f"No samples found in {self.data_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"Found {len(samples)} samples to process")
        
        success_count = 0
        failed_count = 0
        
        for i, sample_dir in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] Processing {sample_dir.name}")
            
            try:
                if self.render_sample(sample_dir):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"  Error: {e}")
                failed_count += 1
        
        return {
            'total': len(samples),
            'success': success_count,
            'failed': failed_count
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Build images from processed FBX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render images for all samples
  blender --background --python build_images.py -- --data output/raw --images output/images
  
  # Render with validation
  blender --background --python build_images.py -- --data output/raw --images output/images --validate
  
  # Custom resolution and samples
  blender --background --python build_images.py -- --data output/raw --images output/images --resolution 1024 --samples 128
"""
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Directory containing processed FBX files'
    )
    parser.add_argument(
        '--images',
        required=True,
        help='Output directory for rendered images'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Image resolution (default: 512)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=64,
        help='Number of render samples (default: 64)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Enable validation by rendering assembled parts'
    )
    parser.add_argument(
        '--sample',
        help='Process only a specific sample (by name)'
    )
    
    # Parse arguments (handle Blender's -- separator)
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args()
    
    # Check Blender availability
    if not BLENDER_AVAILABLE:
        print("Error: This script must be run with Blender")
        print("Usage: blender --background --python build_images.py -- [arguments]")
        return 1
    
    # Create image builder
    builder = ImageBuilder(
        data_dir=args.data,
        images_dir=args.images,
        resolution=args.resolution,
        samples=args.samples,
        enable_validation=args.validate
    )
    
    # Process specific sample or all samples
    if args.sample:
        sample_dir = Path(args.data) / args.sample
        if not sample_dir.exists():
            print(f"Error: Sample not found: {sample_dir}")
            return 1
        
        print(f"Processing single sample: {args.sample}")
        if builder.render_sample(sample_dir):
            print("\nSuccess!")
            return 0
        else:
            print("\nFailed!")
            return 1
    else:
        # Process all samples
        stats = builder.build_all()
        
        print("\n" + "="*60)
        print("Image Building Summary")
        print("="*60)
        print(f"Total samples:    {stats['total']}")
        print(f"Success:          {stats['success']}")
        print(f"Failed:           {stats['failed']}")
        print("="*60)
        
        return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
