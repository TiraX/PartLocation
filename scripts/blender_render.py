"""
Blender rendering module for generating multi-view images.

This module provides functionality to:
1. Load FBX models into Blender
2. Render from standard camera positions (front, back, left, right)
3. Stitch multiple views into a single image
4. Validate part assembly by rendering assembled parts
"""

import bpy
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from mathutils import Vector, Quaternion, Matrix


class BlenderRenderer:
    """Blender renderer for multi-view image generation."""
    
    # Standard camera positions (distance from origin)
    CAMERA_DISTANCE = 2.0
    
    # Camera view directions
    VIEW_FRONT = 0
    VIEW_BACK = 1
    VIEW_LEFT = 2
    VIEW_RIGHT = 3
    
    VIEW_NAMES = ['front', 'back', 'left', 'right']
    
    def __init__(self, resolution: int = 512, samples: int = 64):
        """
        Initialize Blender renderer.
        
        Args:
            resolution: Image resolution (width and height)
            samples: Number of render samples for quality
        """
        self.resolution = resolution
        self.samples = samples
        
        # Setup Blender scene
        self._setup_scene()
        
    def _setup_scene(self):
        """Setup Blender scene with proper render settings."""
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Set render engine to Cycles for better quality
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = self.samples
        bpy.context.scene.cycles.use_denoising = True
        
        # Set resolution
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        
        # Set transparent background
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # Setup lighting
        self._setup_lighting()
        
    def _setup_lighting(self):
        """Setup three-point lighting for good model visibility."""
        # Key light (main light)
        key_light = bpy.data.lights.new(name="KeyLight", type='SUN')
        key_light.energy = 2.0
        key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
        bpy.context.collection.objects.link(key_light_obj)
        key_light_obj.location = (5, -5, 5)
        key_light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
        
        # Fill light (softer, from opposite side)
        fill_light = bpy.data.lights.new(name="FillLight", type='SUN')
        fill_light.energy = 1.0
        fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light)
        bpy.context.collection.objects.link(fill_light_obj)
        fill_light_obj.location = (-5, 5, 3)
        fill_light_obj.rotation_euler = (math.radians(60), 0, math.radians(-135))
        
        # Back light (rim light)
        back_light = bpy.data.lights.new(name="BackLight", type='SUN')
        back_light.energy = 0.5
        back_light_obj = bpy.data.objects.new(name="BackLight", object_data=back_light)
        bpy.context.collection.objects.link(back_light_obj)
        back_light_obj.location = (0, 5, 5)
        back_light_obj.rotation_euler = (math.radians(45), 0, math.radians(180))
        
    def load_fbx(self, fbx_path: str, clear_scene: bool = True) -> List[bpy.types.Object]:
        """
        Load FBX file into Blender.
        
        Args:
            fbx_path: Path to FBX file
            clear_scene: Whether to clear existing objects before loading
            
        Returns:
            List of imported objects
        """
        if clear_scene:
            # Clear existing mesh objects (keep lights and camera)
            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    obj.select_set(True)
            bpy.ops.object.delete()
        
        # Import FBX
        bpy.ops.import_scene.fbx(filepath=fbx_path)
        
        # Get imported objects
        imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
        return imported_objects
        
    def _create_camera_at_view(self, view_index: int) -> bpy.types.Object:
        """
        Create camera at specified view position.
        
        Args:
            view_index: View index (0=front, 1=back, 2=left, 3=right)
            
        Returns:
            Camera object
        """
        # Create camera
        camera_data = bpy.data.cameras.new(name=f"Camera_{self.VIEW_NAMES[view_index]}")
        camera_data.type = 'ORTHO'
        camera_data.ortho_scale = 1.2  # Slightly larger than normalized model
        
        camera_obj = bpy.data.objects.new(f"Camera_{self.VIEW_NAMES[view_index]}", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        
        # Set camera position and rotation based on view
        if view_index == self.VIEW_FRONT:
            # Front view: looking from +Y towards -Y (along -Y axis)
            camera_obj.location = (0, self.CAMERA_DISTANCE, 0)
            camera_obj.rotation_euler = (math.radians(90), 0, math.radians(180))
        elif view_index == self.VIEW_BACK:
            # Back view: looking from -Y towards +Y (along +Y axis)
            camera_obj.location = (0, -self.CAMERA_DISTANCE, 0)
            camera_obj.rotation_euler = (math.radians(90), 0, 0)
        elif view_index == self.VIEW_LEFT:
            # Left view: looking from -X towards +X (along +X axis)
            camera_obj.location = (-self.CAMERA_DISTANCE, 0, 0)
            camera_obj.rotation_euler = (math.radians(90), 0, math.radians(-90))
        elif view_index == self.VIEW_RIGHT:
            # Right view: looking from +X towards -X (along -X axis)
            camera_obj.location = (self.CAMERA_DISTANCE, 0, 0)
            camera_obj.rotation_euler = (math.radians(90), 0, math.radians(90))
        
        return camera_obj
        
    def render_view(self, view_index: int, output_path: str) -> bool:
        """
        Render a single view.
        
        Args:
            view_index: View index (0=front, 1=back, 2=left, 3=right)
            output_path: Output image path
            
        Returns:
            True if rendering successful
        """
        # Create camera for this view
        camera = self._create_camera_at_view(view_index)
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        # Render
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        
        # Clean up camera
        bpy.data.objects.remove(camera, do_unlink=True)
        
        return os.path.exists(output_path)
        
    def render_multi_view(self, output_dir: str, base_name: str, 
                         views: List[int] = None) -> List[str]:
        """
        Render multiple views.
        
        Args:
            output_dir: Output directory for images
            base_name: Base name for output files
            views: List of view indices to render (default: all 4 views)
            
        Returns:
            List of output image paths
        """
        if views is None:
            views = [self.VIEW_FRONT, self.VIEW_BACK, self.VIEW_LEFT, self.VIEW_RIGHT]
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        for view_idx in views:
            view_name = self.VIEW_NAMES[view_idx]
            output_path = os.path.join(output_dir, f"{base_name}_{view_name}.png")
            
            if self.render_view(view_idx, output_path):
                output_paths.append(output_path)
            else:
                print(f"Warning: Failed to render view {view_name}")
        
        return output_paths
        
    def stitch_views(self, image_paths: List[str], output_path: str, 
                    layout: str = 'grid') -> bool:
        """
        Stitch multiple view images into a single image.
        
        Args:
            image_paths: List of image paths to stitch (order: front, back, left, right)
            output_path: Output path for stitched image
            layout: Layout mode ('horizontal' or 'grid', default: 'grid')
            
        Returns:
            True if stitching successful
        """
        try:
            from PIL import Image
        except ImportError:
            print("Error: PIL/Pillow is required for image stitching")
            return False
        
        if not image_paths:
            return False
        
        # Load images
        images = [Image.open(path) for path in image_paths]
        
        # Get image dimensions
        width, height = images[0].size
        
        # Create stitched image
        if layout == 'horizontal':
            # Horizontal layout: [front, back, left, right]
            stitched_width = width * len(images)
            stitched_height = height
            stitched = Image.new('RGBA', (stitched_width, stitched_height))
            
            for i, img in enumerate(images):
                stitched.paste(img, (i * width, 0))
                
        elif layout == 'grid':
            # Grid layout: 2x2 for 4 views
            # Layout: [front, back]
            #         [left,  right]
            cols = 2
            rows = 2
            stitched_width = width * cols
            stitched_height = height * rows
            stitched = Image.new('RGBA', (stitched_width, stitched_height))
            
            # Paste images in order: front(0,0), back(1,0), left(0,1), right(1,1)
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                stitched.paste(img, (col * width, row * height))
        else:
            print(f"Error: Unknown layout mode: {layout}")
            return False
        
        # Save stitched image
        stitched.save(output_path)
        
        # Clean up temporary images
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return True
        
    def render_model(self, fbx_path: str, output_path: str, 
                    num_views: int = 4, layout: str = 'grid') -> bool:
        """
        Render a model with multiple views and stitch into single image.
        
        Args:
            fbx_path: Path to FBX model file
            output_path: Output path for stitched image
            num_views: Number of views (1, 3, or 4)
            layout: Layout mode ('horizontal' or 'grid')
            
        Returns:
            True if rendering successful
        """
        print(f"    [Render] Loading FBX: {Path(fbx_path).name}")
        
        # Load model
        objects = self.load_fbx(fbx_path)
        if not objects:
            print(f"    [Render] ERROR: Failed to load FBX file: {fbx_path}")
            return False
        
        print(f"    [Render] Loaded {len(objects)} objects")
        
        # Determine which views to render
        if num_views == 1:
            views = [self.VIEW_FRONT]
        elif num_views == 3:
            views = [self.VIEW_FRONT, self.VIEW_LEFT, self.VIEW_RIGHT]
        elif num_views == 4:
            views = [self.VIEW_FRONT, self.VIEW_BACK, self.VIEW_LEFT, self.VIEW_RIGHT]
        else:
            print(f"    [Render] ERROR: Unsupported number of views: {num_views}")
            return False
        
        # Create temporary directory for individual views (unique per model)
        base_name = Path(fbx_path).stem
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_views', base_name)
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"    [Render] Rendering {num_views} views to temp dir: {temp_dir}")
        
        # Render views
        view_paths = self.render_multi_view(temp_dir, base_name, views)
        
        if len(view_paths) != len(views):
            print(f"    [Render] ERROR: Failed to render all views (got {len(view_paths)}/{len(views)})")
            return False
        
        print(f"    [Render] Successfully rendered {len(view_paths)} views")
        print(f"    [Render] Stitching views into: {Path(output_path).name}")
        
        # Stitch views
        success = self.stitch_views(view_paths, output_path, layout)
        
        if success:
            print(f"    [Render] SUCCESS: Saved to {Path(output_path).name}")
        else:
            print(f"    [Render] ERROR: Failed to stitch views")
        
        # Clean up temp directory for this model
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        return success
        
    def render_assembled_parts(self, part_fbx_paths: List[str], 
                              transform_params: List[Dict],
                              output_path: str,
                              num_views: int = 4,
                              layout: str = 'grid') -> bool:
        """
        Render assembled parts for validation.
        
        This loads multiple part models, applies transformation parameters,
        and renders the assembled result.
        
        Args:
            part_fbx_paths: List of paths to part FBX files
            transform_params: List of transformation parameters for each part
                Each dict should contain: 'translation', 'rotation', 'scale'
            output_path: Output path for rendered image
            num_views: Number of views (1, 3, or 4)
            layout: Layout mode ('horizontal' or 'grid')
            
        Returns:
            True if rendering successful
        """
        if len(part_fbx_paths) != len(transform_params):
            print("    [Render] ERROR: Number of parts and transform parameters must match")
            return False
        
        print(f"    [Render] Assembling {len(part_fbx_paths)} parts for validation")
        
        # Clear scene
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # Load and transform each part
        for fbx_path, params in zip(part_fbx_paths, transform_params):
            # Load part
            objects = self.load_fbx(fbx_path, clear_scene=False)
            
            if not objects:
                print(f"    [Render] WARNING: Failed to load part: {fbx_path}")
                continue
            
            # Apply transformation to all objects of this part
            translation = Vector(params['translation'])
            rotation_quat = Quaternion(params['rotation'])  # [w, x, y, z]
            scale = Vector(params['scale'])
            
            for obj in objects:
                # Create transformation matrix
                mat_trans = Matrix.Translation(translation)
                mat_rot = rotation_quat.to_matrix().to_4x4()
                mat_scale = Matrix.Scale(scale.x, 4, (1, 0, 0)) @ \
                           Matrix.Scale(scale.y, 4, (0, 1, 0)) @ \
                           Matrix.Scale(scale.z, 4, (0, 0, 1))
                
                # Apply transformation
                obj.matrix_world = mat_trans @ mat_rot @ mat_scale
        
        # Render assembled model
        # Determine which views to render
        if num_views == 1:
            views = [self.VIEW_FRONT]
        elif num_views == 3:
            views = [self.VIEW_FRONT, self.VIEW_LEFT, self.VIEW_RIGHT]
        elif num_views == 4:
            views = [self.VIEW_FRONT, self.VIEW_BACK, self.VIEW_LEFT, self.VIEW_RIGHT]
        else:
            print(f"    [Render] ERROR: Unsupported number of views: {num_views}")
            return False
        
        # Create temporary directory for individual views (unique for validation)
        base_name = Path(output_path).stem
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_views', base_name)
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"    [Render] Rendering {num_views} validation views")
        
        # Render views
        view_paths = self.render_multi_view(temp_dir, base_name, views)
        
        if len(view_paths) != len(views):
            print(f"    [Render] ERROR: Failed to render all views (got {len(view_paths)}/{len(views)})")
            return False
        
        print(f"    [Render] Successfully rendered {len(view_paths)} validation views")
        
        # Stitch views
        success = self.stitch_views(view_paths, output_path, layout)
        
        if success:
            print(f"    [Render] SUCCESS: Saved validation to {Path(output_path).name}")
        else:
            print(f"    [Render] ERROR: Failed to stitch validation views")
        
        # Clean up temp directory for validation
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        return success


def compare_images(image1_path: str, image2_path: str) -> Dict[str, float]:
    """
    Compare two images and return similarity metrics.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        
    Returns:
        Dictionary with metrics: 'ssim', 'mse', 'psnr'
    """
    try:
        from PIL import Image
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("Error: PIL, numpy, and scikit-image are required for image comparison")
        return {}
    
    # Load images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    
    # Convert to numpy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Calculate MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Calculate SSIM
    ssim_value = ssim(arr1, arr2, multichannel=True, channel_axis=2, data_range=255.0)
    
    return {
        'ssim': float(ssim_value),
        'mse': float(mse),
        'psnr': float(psnr)
    }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Render FBX models with Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input',
        help='Input FBX file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output image path'
    )
    parser.add_argument(
        '--views',
        type=int,
        default=4,
        choices=[1, 3, 4],
        help='Number of views to render (default: 4)'
    )
    parser.add_argument(
        '--layout',
        default='horizontal',
        choices=['horizontal', 'grid'],
        help='Layout mode for stitching (default: horizontal)'
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
        help='Render samples for quality (default: 64)'
    )
    
    args = parser.parse_args()
    
    # Create renderer
    renderer = BlenderRenderer(resolution=args.resolution, samples=args.samples)
    
    # Render model
    success = renderer.render_model(
        args.input,
        args.output,
        num_views=args.views,
        layout=args.layout
    )
    
    if success:
        print(f"Successfully rendered to: {args.output}")
        return 0
    else:
        print("Error: Rendering failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
