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
    
    def __init__(self, resolution: int = 512, samples: int = 64, use_gpu: bool = True):
        """
        Initialize Blender renderer.
        
        Args:
            resolution: Image resolution (width and height)
            samples: Number of render samples for quality
            use_gpu: Whether to use GPU for rendering (default: True)
        """
        self.resolution = resolution
        self.samples = samples
        self.use_gpu = use_gpu
        
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
        
        # Configure GPU rendering if enabled
        if self.use_gpu:
            self._setup_gpu_rendering()
        
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
        
    def _setup_gpu_rendering(self):
        """Setup GPU rendering for Cycles."""
        try:
            # Get preferences
            preferences = bpy.context.preferences
            cycles_preferences = preferences.addons['cycles'].preferences
            
            # Enable GPU compute
            cycles_preferences.compute_device_type = 'CUDA'  # Try CUDA first
            
            # Get available devices
            cycles_preferences.get_devices()
            
            # Enable all available GPU devices
            gpu_found = False
            for device in cycles_preferences.devices:
                if device.type in {'CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'}:
                    device.use = True
                    gpu_found = True
                    print(f"    [GPU] Enabled: {device.name} ({device.type})")
            
            if gpu_found:
                # Set scene to use GPU
                bpy.context.scene.cycles.device = 'GPU'
                print(f"    [GPU] Rendering mode: GPU")
            else:
                print(f"    [GPU] No GPU devices found, falling back to CPU")
                bpy.context.scene.cycles.device = 'CPU'
                
        except Exception as e:
            print(f"    [GPU] Failed to setup GPU rendering: {e}")
            print(f"    [GPU] Falling back to CPU rendering")
            bpy.context.scene.cycles.device = 'CPU'
        
    def _setup_lighting(self):
        light_boost = 1.6#1.8
        amb_light_boost = 0.2#0.6
        """Setup three-point lighting for good model visibility."""
        
        # Setup HDR environment lighting
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Add Background node
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.inputs['Strength'].default_value = 1.0 * amb_light_boost
        # Set a neutral gray color for basic ambient lighting
        bg_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        
        # Add Output node
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        
        # Link nodes
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        
        # Key light (main light)
        key_light = bpy.data.lights.new(name="KeyLight", type='SUN')
        key_light.energy = 2.0 * light_boost
        key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
        bpy.context.collection.objects.link(key_light_obj)
        key_light_obj.location = (5, -5, 5)
        key_light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
        
        # Fill light (softer, from opposite side)
        fill_light = bpy.data.lights.new(name="FillLight", type='SUN')
        fill_light.energy = 1.0 * light_boost
        fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light)
        bpy.context.collection.objects.link(fill_light_obj)
        fill_light_obj.location = (-5, 5, 3)
        fill_light_obj.rotation_euler = (math.radians(60), 0, math.radians(-135))
        
        # Back light (rim light)
        back_light = bpy.data.lights.new(name="BackLight", type='SUN')
        back_light.energy = 0.5 * light_boost
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
        
    def render_model(self, fbx_path: str, output_dir: str) -> bool:
        """
        Render a model with 4 views (front, back, left, right).
        
        Args:
            fbx_path: Path to FBX model file
            output_dir: Output directory for view images
            
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
        
        # Render 4 views
        views = [self.VIEW_FRONT, self.VIEW_BACK, self.VIEW_LEFT, self.VIEW_RIGHT]
        base_name = Path(fbx_path).stem
        
        print(f"    [Render] Rendering 4 views to: {output_dir}")
        
        # Render views directly to output directory
        view_paths = self.render_multi_view(output_dir, base_name, views)
        
        if len(view_paths) != len(views):
            print(f"    [Render] ERROR: Failed to render all views (got {len(view_paths)}/{len(views)})")
            return False
        
        print(f"    [Render] SUCCESS: Rendered {len(view_paths)} views")
        
        return True
        
    def render_assembled_parts(self, part_fbx_paths: List[str], 
                              transform_params: List[Dict],
                              output_dir: str,
                              model_name: str) -> bool:
        """
        Render assembled parts for validation with 4 views.
        
        This loads multiple part models, applies transformation parameters,
        and renders the assembled result.
        
        Args:
            part_fbx_paths: List of paths to part FBX files
            transform_params: List of transformation parameters for each part
                Each dict should contain: 'translation', 'rotation', 'scale'
            output_dir: Output directory for view images
            model_name: Base name for output files
            
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
        
        # Render assembled model with 4 views
        views = [self.VIEW_FRONT, self.VIEW_BACK, self.VIEW_LEFT, self.VIEW_RIGHT]
        
        print(f"    [Render] Rendering 4 validation views")
        
        # Render views directly to output directory
        view_paths = self.render_multi_view(output_dir, model_name, views)
        
        if len(view_paths) != len(views):
            print(f"    [Render] ERROR: Failed to render all views (got {len(view_paths)}/{len(views)})")
            return False
        
        print(f"    [Render] SUCCESS: Rendered {len(view_paths)} validation views")
        
        return True


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
