"""
Generate dataset from FBX files.

This script processes FBX files to create training data:
1. Load FBX model
2. Normalize whole model to [-0.5, 0.5] with padding
3. Split model into parts
4. Normalize each part individually
5. Calculate relative transformation parameters
6. Save normalized models and transformation parameters
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add mesh_dump to path
sys.path.insert(0, str(Path(__file__).parent))

from mesh_dump.fbx_utils import FbxUtil
from mesh_dump.model import Model
from mesh_dump.mesh import Mesh


class DatasetGenerator:
    """Generate training dataset from FBX files."""
    
    def __init__(self, output_dir: str, padding: float = 0.01):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for processed data
            padding: Padding ratio for normalization (default: 0.01)
        """
        self.output_dir = Path(output_dir)
        self.padding = padding
        self.min_val = -0.5
        self.max_val = 0.5
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_fbx_file(self, fbx_path: str, sample_name: Optional[str] = None) -> bool:
        """
        Process a single FBX file.
        
        Args:
            fbx_path: Path to FBX file
            sample_name: Optional custom sample name (default: use filename)
            
        Returns:
            True if processing successful, False otherwise
        """
        fbx_path = Path(fbx_path)
        if not fbx_path.exists():
            print(f"Error: FBX file not found: {fbx_path}")
            return False
        
        # Use filename as sample name if not provided
        if sample_name is None:
            sample_name = fbx_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {fbx_path.name}")
        print(f"Sample name: {sample_name}")
        print(f"{'='*60}")
        
        # Create sample directory
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {sample_dir}")
        
        try:
            # Step 1: Load FBX model
            print("\n[1/5] Loading FBX model...")
            model = FbxUtil.load_model(str(fbx_path), ignore_skeleton=True)
            if model is None:
                print(f"Error: Failed to load FBX file: {fbx_path}")
                return False
            if len(model.meshes) == 1:
                print(f"Warning: Model has only 1 mesh, skipping processing")
                return False
            print(f"  Loaded model with {model.get_mesh_count()} meshes")
            
            # Step 2: Get original bounds before normalization
            print("\n[2/5] Recording original bounds...")
            original_bounds = model.get_bounds()
            if original_bounds is None:
                print("Error: Model has no valid bounds")
                return False
            
            original_bounds_min, original_bounds_max = original_bounds
            original_center = (original_bounds_min + original_bounds_max) * 0.5
            original_size = original_bounds_max - original_bounds_min
            print(f"  Original bounds: min={original_bounds_min}, max={original_bounds_max}")
            print(f"  Original center: {original_center}")
            print(f"  Original size: {original_size}")
            
            # Step 3: Normalize whole model
            print("\n[3/5] Normalizing whole model...")
            model.normalize(self.min_val, self.max_val, self.padding)
            
            # Get normalized bounds
            normalized_bounds = model.get_bounds()
            if normalized_bounds is None:
                print("Error: Failed to get normalized bounds")
                return False
            
            normalized_bounds_min, normalized_bounds_max = normalized_bounds
            normalized_center = (normalized_bounds_min + normalized_bounds_max) * 0.5
            normalized_size = normalized_bounds_max - normalized_bounds_min
            print(f"  Normalized bounds: min={normalized_bounds_min}, max={normalized_bounds_max}")
            print(f"  Normalized center: {normalized_center}")
            print(f"  Normalized size: {normalized_size}")
            
            # Calculate normalization parameters
            max_dimension = np.max(original_size)
            target_range = self.max_val - self.min_val
            effective_range = target_range * (1.0 - 2.0 * self.padding)
            scale_factor = effective_range / max_dimension if max_dimension > 0 else 1.0
            
            whole_normalization = {
                "original_center": original_center.tolist(),
                "original_size": original_size.tolist(),
                "scale_factor": float(scale_factor),
                "target_range": [self.min_val, self.max_val],
                "padding": self.padding
            }
            
            # Save normalized whole model
            whole_output_path = sample_dir / f"{sample_name}-whole.fbx"
            print(f"  Saving normalized whole model to: {whole_output_path}")
            if not FbxUtil.save_model(model, output_path=str(whole_output_path), ignore_skeleton=True):
                print(f"Error: Failed to save whole model")
                return False
            
            # Step 4: Split model into parts and process each part
            print("\n[4/5] Splitting and processing parts...")
            part_models = model.split()
            print(f"  Split into {len(part_models)} parts")
            
            parts_data = {}
            
            for i, part_model in enumerate(part_models):
                part_name = part_model.name
                print(f"\n  Processing part {i+1}/{len(part_models)}: {part_name}")
                
                # Get part bounds in whole model's normalized space
                part_bounds_in_whole = part_model.get_bounds()
                if part_bounds_in_whole is None:
                    print(f"    Warning: Part {part_name} has no valid bounds, skipping")
                    continue
                
                part_bounds_min_in_whole, part_bounds_max_in_whole = part_bounds_in_whole
                part_center_in_whole = (part_bounds_min_in_whole + part_bounds_max_in_whole) * 0.5
                part_size_in_whole = part_bounds_max_in_whole - part_bounds_min_in_whole
                
                print(f"    Part bounds in whole space: min={part_bounds_min_in_whole}, max={part_bounds_max_in_whole}")
                print(f"    Part center in whole space: {part_center_in_whole}")
                print(f"    Part size in whole space: {part_size_in_whole}")
                
                # Normalize part individually
                print(f"    Normalizing part individually...")
                part_model.normalize(self.min_val, self.max_val, self.padding)
                
                # Get part's own normalization parameters
                part_normalized_bounds = part_model.get_bounds()
                if part_normalized_bounds is None:
                    print(f"    Warning: Failed to get normalized bounds for part {part_name}, skipping")
                    continue
                
                part_normalized_min, part_normalized_max = part_normalized_bounds
                part_normalized_center = (part_normalized_min + part_normalized_max) * 0.5
                
                # Calculate part's normalization scale factor
                part_max_dimension = np.max(part_size_in_whole)
                part_scale_factor = effective_range / part_max_dimension if part_max_dimension > 0 else 1.0
                
                print(f"    Part normalized bounds: min={part_normalized_min}, max={part_normalized_max}")
                print(f"    Part scale factor: {part_scale_factor}")
                
                # Calculate relative transformation
                # Translation: part center in whole's normalized space
                translation = part_center_in_whole
                
                # Rotation: identity (no rotation in this simple case)
                rotation = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
                
                # Scale: ratio of part's scale to whole's scale
                scale_ratio = part_scale_factor / scale_factor if scale_factor > 0 else 1.0
                scale = np.array([scale_ratio, scale_ratio, scale_ratio])
                
                print(f"    Relative transformation:")
                print(f"      Translation: {translation}")
                print(f"      Rotation (quat): {rotation}")
                print(f"      Scale: {scale}")
                
                # Save normalized part model
                part_output_path = sample_dir / f"{sample_name}-{part_name}.fbx"
                print(f"    Saving normalized part to: {part_output_path}")
                if not FbxUtil.save_model(part_model, output_path=str(part_output_path), ignore_skeleton=True):
                    print(f"    Warning: Failed to save part {part_name}")
                    continue
                
                # Prepare part data
                part_data = {
                    "model_name": sample_name,
                    "part_name": part_name,
                    "translation": translation.tolist(),
                    "rotation": rotation.tolist(),
                    "scale": scale.tolist(),
                    "part_normalization": {
                        "original_center_in_whole": part_center_in_whole.tolist(),
                        "original_size_in_whole": part_size_in_whole.tolist(),
                        "scale_factor": float(part_scale_factor)
                    },
                    "bounds_in_whole": {
                        "min": part_bounds_min_in_whole.tolist(),
                        "max": part_bounds_max_in_whole.tolist()
                    },
                    "whole_normalization": whole_normalization
                }
                
                # Save part JSON
                part_json_path = sample_dir / f"{sample_name}-{part_name}.json"
                with open(part_json_path, 'w', encoding='utf-8') as f:
                    json.dump(part_data, f, indent=2, ensure_ascii=False)
                print(f"    Part JSON saved to: {part_json_path}")
                
                # Store part data for summary
                parts_data[part_name] = part_data
            
            # Step 5: Summary
            print(f"\n✓ Successfully processed {sample_name}")
            print(f"  Output directory: {sample_dir}")
            print(f"  - Whole model: {whole_output_path.name}")
            print(f"  - Parts: {len(parts_data)} parts saved")
            for part_name in parts_data.keys():
                print(f"    - {sample_name}-{part_name}.fbx")
                print(f"    - {sample_name}-{part_name}.json")
            
            return True
            
        except Exception as e:
            print(f"\nError processing {fbx_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir: str, recursive: bool = False) -> Tuple[int, int]:
        """
        Process all FBX files in a directory.
        
        Args:
            input_dir: Input directory containing FBX files
            recursive: Whether to search recursively
            
        Returns:
            Tuple of (success_count, total_count)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 0, 0
        
        # Find all FBX files
        if recursive:
            fbx_files = list(input_path.rglob("*.fbx"))
        else:
            fbx_files = list(input_path.glob("*.fbx"))
        
        if not fbx_files:
            print(f"No FBX files found in: {input_dir}")
            return 0, 0
        
        print(f"\nFound {len(fbx_files)} FBX files to process")
        
        success_count = 0
        for i, fbx_file in enumerate(fbx_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(fbx_files)}")
            print(f"{'='*60}")
            
            if self.process_fbx_file(str(fbx_file)):
                success_count += 1
        
        return success_count, len(fbx_files)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate training dataset from FBX files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single FBX file
  python generate_dataset.py input.fbx -o output_dir
  
  # Process all FBX files in a directory
  python generate_dataset.py input_dir/ -o output_dir
  
  # Process recursively with custom padding
  python generate_dataset.py input_dir/ -o output_dir -r --padding 0.02
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for FBX files recursively in subdirectories"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.01,
        help="Padding ratio for normalization (default: 0.01)"
    )
    parser.add_argument(
        "--name",
        help="Custom sample name (only for single file processing)"
    )
    
    args = parser.parse_args()
    
    # Validate padding
    if not 0.0 <= args.padding < 0.5:
        print("Error: Padding must be between 0.0 and 0.5")
        return 1
    
    # Create generator
    generator = DatasetGenerator(args.output, padding=args.padding)
    
    # Check if input is file or directory
    input_paths = (
        ('d:/Data/ShadowUnit', False), 
        ('d:/Data/DiabloChar/Processed', True), 
        ('d:/Data/models_gta5.peds_only/models/peds', True), 
        ('d:/Data/models_rdr2.peds_only/models/peds', True), 
        )
    
    for input_path, recursive in input_paths:
        # Process directory
        if args.name:
            print("Warning: --name argument is ignored when processing a directory")
        
        success_count, total_count = generator.process_directory(
            str(input_path),
            recursive=recursive
        )
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {success_count}/{total_count} files")
        print(f"Output directory: {generator.output_dir}")
        
        return 0 if success_count == total_count else 1
        
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
