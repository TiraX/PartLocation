"""
Extract transformation parameters (Ground Truth) from Blender scene.

This script extracts the relative transformation of parts with respect to the whole object.
Run within Blender using:
    blender --background --python extract_transforms.py -- [arguments]
"""

import bpy
import json
import sys
from pathlib import Path
from mathutils import Vector, Matrix, Quaternion


class TransformExtractor:
    """Extract transformation parameters from Blender objects."""
    
    def __init__(self):
        """Initialize the extractor."""
        pass
        
    def get_object_transform(self, obj_name: str) -> dict:
        """Get world transformation of an object.
        
        Args:
            obj_name: Name of the object
            
        Returns:
            Dictionary containing transformation parameters
        """
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")
            
        # Get world matrix
        matrix_world = obj.matrix_world.copy()
        
        # Decompose matrix into location, rotation, scale
        location, rotation, scale = matrix_world.decompose()
        
        # Convert rotation to quaternion (already in quaternion format)
        # Blender uses (w, x, y, z) format
        quat = rotation
        
        return {
            'location': [location.x, location.y, location.z],
            'rotation_quaternion': [quat.w, quat.x, quat.y, quat.z],
            'scale': [scale.x, scale.y, scale.z],
            'matrix_world': [list(row) for row in matrix_world]
        }
        
    def get_relative_transform(self, part_name: str, whole_name: str) -> dict:
        """Get relative transformation of part with respect to whole.
        
        Args:
            part_name: Name of the part object
            whole_name: Name of the whole object
            
        Returns:
            Dictionary containing relative transformation parameters
        """
        part_obj = bpy.data.objects.get(part_name)
        whole_obj = bpy.data.objects.get(whole_name)
        
        if part_obj is None:
            raise ValueError(f"Part object '{part_name}' not found")
        if whole_obj is None:
            raise ValueError(f"Whole object '{whole_name}' not found")
            
        # Get world matrices
        part_matrix = part_obj.matrix_world.copy()
        whole_matrix = whole_obj.matrix_world.copy()
        
        # Calculate relative matrix: M_relative = M_whole^-1 * M_part
        whole_matrix_inv = whole_matrix.inverted()
        relative_matrix = whole_matrix_inv @ part_matrix
        
        # Decompose relative matrix
        location, rotation, scale = relative_matrix.decompose()
        
        # Normalize quaternion
        rotation.normalize()
        
        return {
            'part_name': part_name,
            'whole_name': whole_name,
            'translation': [location.x, location.y, location.z],
            'rotation': [rotation.w, rotation.x, rotation.y, rotation.z],
            'scale': [scale.x, scale.y, scale.z]
        }
        
    def extract_scene_transforms(self, whole_name: str, part_names: list) -> dict:
        """Extract transformations for all parts in the scene.
        
        Args:
            whole_name: Name of the whole object
            part_names: List of part object names
            
        Returns:
            Dictionary containing all transformation data
        """
        result = {
            'whole_object': whole_name,
            'parts': []
        }
        
        for part_name in part_names:
            try:
                transform = self.get_relative_transform(part_name, whole_name)
                result['parts'].append(transform)
            except ValueError as e:
                print(f"Warning: {e}")
                
        return result
        
    def save_transforms(self, transforms: dict, output_path: str):
        """Save transformation data to JSON file.
        
        Args:
            transforms: Dictionary containing transformation data
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transforms, f, indent=2, ensure_ascii=False)
            
        print(f"Transforms saved to: {output_path}")
        
    def validate_transforms(self, transforms: dict) -> bool:
        """Validate transformation data.
        
        Args:
            transforms: Dictionary containing transformation data
            
        Returns:
            True if valid, False otherwise
        """
        if 'whole_object' not in transforms:
            print("Error: Missing 'whole_object' field")
            return False
            
        if 'parts' not in transforms or not transforms['parts']:
            print("Error: No parts found")
            return False
            
        for part in transforms['parts']:
            # Check required fields
            required_fields = ['part_name', 'translation', 'rotation', 'scale']
            for field in required_fields:
                if field not in part:
                    print(f"Error: Missing field '{field}' in part {part.get('part_name', 'unknown')}")
                    return False
                    
            # Check dimensions
            if len(part['translation']) != 3:
                print(f"Error: Invalid translation dimension for {part['part_name']}")
                return False
            if len(part['rotation']) != 4:
                print(f"Error: Invalid rotation dimension for {part['part_name']}")
                return False
            if len(part['scale']) != 3:
                print(f"Error: Invalid scale dimension for {part['part_name']}")
                return False
                
        return True


def get_all_mesh_objects() -> list:
    """Get all mesh objects in the scene.
    
    Returns:
        List of mesh object names
    """
    return [obj.name for obj in bpy.data.objects if obj.type == 'MESH']


def main():
    """Main function for command-line usage."""
    # Parse arguments (after --)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    if len(argv) < 2:
        print("Usage: blender --background --python extract_transforms.py -- <whole_object_name> <output_path> [part1] [part2] ...")
        print("\nAvailable mesh objects in scene:")
        for obj_name in get_all_mesh_objects():
            print(f"  - {obj_name}")
        return
    
    whole_name = argv[0]
    output_path = argv[1]
    part_names = argv[2:] if len(argv) > 2 else []
    
    # If no parts specified, use all mesh objects except the whole
    if not part_names:
        all_meshes = get_all_mesh_objects()
        part_names = [name for name in all_meshes if name != whole_name]
        print(f"No parts specified. Using all mesh objects except '{whole_name}':")
        for name in part_names:
            print(f"  - {name}")
    
    # Extract transforms
    extractor = TransformExtractor()
    transforms = extractor.extract_scene_transforms(whole_name, part_names)
    
    # Validate
    if not extractor.validate_transforms(transforms):
        print("Error: Transform validation failed")
        return
    
    # Save
    extractor.save_transforms(transforms, output_path)
    print(f"Successfully extracted transforms for {len(transforms['parts'])} parts")


if __name__ == "__main__":
    main()
