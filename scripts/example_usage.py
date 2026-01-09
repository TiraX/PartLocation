"""
Example script demonstrating how to use the rendering and extraction tools.

This script can be run within Blender's scripting interface or via command line.
It shows a complete workflow for a single sample.
"""

import bpy
import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from blender_render import MultiViewRenderer
from extract_transforms import TransformExtractor


def setup_example_scene():
    """Setup a simple example scene with a cube and smaller cube as part."""
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    # Create whole object (larger cube)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    whole = bpy.context.active_object
    whole.name = "Whole"
    
    # Create part object (smaller cube on top)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 1.25))
    part = bpy.context.active_object
    part.name = "Part"
    
    # Add some color to distinguish them
    # Whole - Blue
    mat_whole = bpy.data.materials.new(name="WholeMaterial")
    mat_whole.use_nodes = True
    mat_whole.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.1, 0.1, 0.8, 1)
    whole.data.materials.append(mat_whole)
    
    # Part - Red
    mat_part = bpy.data.materials.new(name="PartMaterial")
    mat_part.use_nodes = True
    mat_part.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.1, 0.1, 1)
    part.data.materials.append(mat_part)
    
    print("Example scene created: 'Whole' and 'Part' objects")
    return whole, part


def example_render_workflow():
    """Example: Render multi-view images."""
    print("\n" + "="*60)
    print("EXAMPLE: Multi-View Rendering")
    print("="*60)
    
    # Setup scene
    whole, part = setup_example_scene()
    
    # Create renderer
    output_dir = Path(__file__).parent.parent / "data" / "examples"
    renderer = MultiViewRenderer(str(output_dir), resolution=512)
    
    # Setup lighting
    renderer.setup_lighting()
    
    # Define views to render
    views = ['front', 'back', 'left', 'right']
    
    # Render whole object
    print("\nRendering whole object...")
    whole_path = renderer.render_object(
        obj_name="Whole",
        views=views,
        output_name="example_whole",
        combine_mode='grid'
    )
    print(f"Whole object rendered: {whole_path}")
    
    # Render part object
    print("\nRendering part object...")
    part_path = renderer.render_object(
        obj_name="Part",
        views=views,
        output_name="example_part",
        combine_mode='grid'
    )
    print(f"Part object rendered: {part_path}")
    
    print("\n✓ Rendering complete!")
    return output_dir


def example_transform_workflow():
    """Example: Extract transformation parameters."""
    print("\n" + "="*60)
    print("EXAMPLE: Transform Extraction")
    print("="*60)
    
    # Create extractor
    extractor = TransformExtractor()
    
    # Extract relative transform
    print("\nExtracting transformation of 'Part' relative to 'Whole'...")
    transform = extractor.get_relative_transform("Part", "Whole")
    
    print("\nExtracted transformation:")
    print(f"  Translation: {transform['translation']}")
    print(f"  Rotation (quaternion): {transform['rotation']}")
    print(f"  Scale: {transform['scale']}")
    
    # Extract all transforms
    transforms = extractor.extract_scene_transforms("Whole", ["Part"])
    
    # Validate
    if extractor.validate_transforms(transforms):
        print("\n✓ Transform validation passed!")
    else:
        print("\n✗ Transform validation failed!")
        
    # Save to file
    output_dir = Path(__file__).parent.parent / "data" / "examples"
    output_path = output_dir / "example_transforms.json"
    extractor.save_transforms(transforms, str(output_path))
    
    return transforms


def example_complete_workflow():
    """Example: Complete workflow for one sample."""
    print("\n" + "="*60)
    print("EXAMPLE: Complete Workflow")
    print("="*60)
    
    # Setup scene
    print("\n1. Setting up example scene...")
    whole, part = setup_example_scene()
    
    # Render images
    print("\n2. Rendering multi-view images...")
    output_dir = example_render_workflow()
    
    # Extract transforms
    print("\n3. Extracting transformation parameters...")
    transforms = example_transform_workflow()
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - example_whole_combined.png (4-view grid)")
    print("  - example_part_combined.png (4-view grid)")
    print("  - example_transforms.json (ground truth)")
    print("\nYou can now use these files for training!")


def example_with_augmentation():
    """Example: Rendering with data augmentation."""
    print("\n" + "="*60)
    print("EXAMPLE: Data Augmentation")
    print("="*60)
    
    # Setup scene
    whole, part = setup_example_scene()
    
    # Create renderer
    output_dir = Path(__file__).parent.parent / "data" / "examples" / "augmented"
    renderer = MultiViewRenderer(str(output_dir), resolution=512)
    renderer.setup_lighting()
    
    views = ['front', 'back', 'left', 'right']
    
    # Render with different augmentations
    augmentations = [
        {'elevation': 0, 'azimuth_offset': 0, 'name': 'no_aug'},
        {'elevation': 10, 'azimuth_offset': 0, 'name': 'elev_10'},
        {'elevation': -10, 'azimuth_offset': 0, 'name': 'elev_-10'},
        {'elevation': 0, 'azimuth_offset': 15, 'name': 'azim_15'},
        {'elevation': 5, 'azimuth_offset': -5, 'name': 'mixed'},
    ]
    
    print("\nRendering with different augmentations...")
    for aug in augmentations:
        print(f"\n  Rendering: {aug['name']}")
        renderer.render_object(
            obj_name="Whole",
            views=views,
            output_name=f"aug_{aug['name']}",
            combine_mode='grid',
            elevation=aug['elevation'],
            azimuth_offset=aug['azimuth_offset']
        )
    
    print(f"\n✓ Augmented samples saved to: {output_dir}")


def main():
    """Main function - run examples."""
    print("\n" + "="*60)
    print("PART LOCATION MODEL - EXAMPLE SCRIPTS")
    print("="*60)
    
    # Run complete workflow
    example_complete_workflow()
    
    # Optionally run augmentation example
    print("\n\nWould you like to see augmentation examples?")
    print("(This will generate additional images)")
    # Uncomment to run:
    # example_with_augmentation()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Check if running in Blender
    try:
        import bpy
        main()
    except ImportError:
        print("Error: This script must be run within Blender")
        print("Usage: blender --python example_usage.py")
