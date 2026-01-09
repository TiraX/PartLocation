"""
Blender rendering script for generating multi-view images.

This script should be run within Blender using:
    blender --background --python blender_render.py -- [arguments]

Or from Blender's scripting interface.
"""

import bpy
import math
import sys
import os
from pathlib import Path
from mathutils import Vector, Quaternion, Euler


class MultiViewRenderer:
    """Multi-view renderer for 3D models."""
    
    def __init__(self, output_dir: str, resolution: int = 512):
        """Initialize the renderer.
        
        Args:
            output_dir: Directory to save rendered images
            resolution: Image resolution (width and height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        
        # Setup render settings
        self.setup_render_settings()
        
    def setup_render_settings(self):
        """Configure Blender render settings."""
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'  # Use Cycles for better quality
        scene.render.resolution_x = self.resolution
        scene.render.resolution_y = self.resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        
        # Cycles settings
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        
        # Transparent background
        scene.render.film_transparent = True
        
    def setup_camera(self, distance: float = 3.0, focal_length: float = 50.0):
        """Setup or get camera.
        
        Args:
            distance: Distance from origin
            focal_length: Camera focal length in mm
            
        Returns:
            Camera object
        """
        # Remove existing cameras
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='CAMERA')
        bpy.ops.object.delete()
        
        # Create new camera
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.data.lens = focal_length
        
        # Set as active camera
        bpy.context.scene.camera = camera
        
        return camera
        
    def setup_lighting(self, energy: float = 1000.0):
        """Setup scene lighting.
        
        Args:
            energy: Light energy/strength
        """
        # Remove existing lights
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()
        
        # Add key light (front-top)
        bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
        key_light = bpy.context.active_object
        key_light.data.energy = energy
        key_light.data.size = 5
        
        # Add fill light (back)
        bpy.ops.object.light_add(type='AREA', location=(-2, 2, 2))
        fill_light = bpy.context.active_object
        fill_light.data.energy = energy * 0.5
        fill_light.data.size = 5
        
        # Add rim light (side)
        bpy.ops.object.light_add(type='AREA', location=(3, 0, 1))
        rim_light = bpy.context.active_object
        rim_light.data.energy = energy * 0.3
        rim_light.data.size = 3
        
    def set_camera_position(self, camera, view_type: str, distance: float = 3.0, 
                           elevation: float = 0.0, azimuth_offset: float = 0.0):
        """Set camera position for a specific view.
        
        Args:
            camera: Camera object
            view_type: One of 'front', 'back', 'left', 'right'
            distance: Distance from origin
            elevation: Elevation angle in degrees (for data augmentation)
            azimuth_offset: Azimuth offset in degrees (for data augmentation)
        """
        # Base azimuth angles for each view
        view_angles = {
            'front': 0,
            'right': 90,
            'back': 180,
            'left': 270
        }
        
        if view_type not in view_angles:
            raise ValueError(f"Invalid view_type: {view_type}")
        
        # Calculate camera position
        azimuth = math.radians(view_angles[view_type] + azimuth_offset)
        elevation_rad = math.radians(elevation)
        
        x = distance * math.cos(elevation_rad) * math.sin(azimuth)
        y = -distance * math.cos(elevation_rad) * math.cos(azimuth)
        z = distance * math.sin(elevation_rad)
        
        camera.location = (x, y, z)
        
        # Point camera at origin
        direction = Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
    def render_view(self, camera, view_type: str, output_path: str, 
                   elevation: float = 0.0, azimuth_offset: float = 0.0):
        """Render a single view.
        
        Args:
            camera: Camera object
            view_type: View type identifier
            output_path: Path to save the rendered image
            elevation: Elevation angle offset for augmentation
            azimuth_offset: Azimuth angle offset for augmentation
        """
        self.set_camera_position(camera, view_type, elevation=elevation, 
                                azimuth_offset=azimuth_offset)
        
        # Render
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        
    def render_multiview(self, views: list, output_prefix: str, 
                        elevation: float = 0.0, azimuth_offset: float = 0.0):
        """Render multiple views.
        
        Args:
            views: List of view types to render (e.g., ['front', 'back', 'left', 'right'])
            output_prefix: Prefix for output filenames
            elevation: Elevation angle offset
            azimuth_offset: Azimuth angle offset
            
        Returns:
            List of rendered image paths
        """
        camera = self.setup_camera()
        rendered_paths = []
        
        for view_type in views:
            output_path = self.output_dir / f"{output_prefix}_{view_type}.png"
            self.render_view(camera, view_type, str(output_path), 
                           elevation, azimuth_offset)
            rendered_paths.append(output_path)
            
        return rendered_paths
        
    def combine_views_horizontal(self, image_paths: list, output_path: str):
        """Combine multiple views into a single horizontal image.
        
        Args:
            image_paths: List of image paths to combine
            output_path: Path to save combined image
        """
        import numpy as np
        from PIL import Image
        
        images = [Image.open(p) for p in image_paths]
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        combined = Image.new('RGBA', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
            
        combined.save(output_path)
        
    def combine_views_grid(self, image_paths: list, output_path: str, grid_size: tuple = (2, 2)):
        """Combine multiple views into a grid layout.
        
        Args:
            image_paths: List of image paths to combine
            output_path: Path to save combined image
            grid_size: Grid dimensions (rows, cols)
        """
        import numpy as np
        from PIL import Image
        
        images = [Image.open(p) for p in image_paths]
        img_width, img_height = images[0].size
        
        rows, cols = grid_size
        combined = Image.new('RGBA', (img_width * cols, img_height * rows))
        
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            combined.paste(img, (col * img_width, row * img_height))
            
        combined.save(output_path)
        
    def render_object(self, obj_name: str, views: list, output_name: str,
                     combine_mode: str = 'grid', elevation: float = 0.0, 
                     azimuth_offset: float = 0.0):
        """Render a specific object with multiple views.
        
        Args:
            obj_name: Name of the object to render
            views: List of view types
            output_name: Output filename prefix
            combine_mode: 'horizontal', 'grid', or 'separate'
            elevation: Elevation angle offset
            azimuth_offset: Azimuth angle offset
            
        Returns:
            Path to the rendered image(s)
        """
        # Hide all objects except the target
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.hide_render = (obj.name != obj_name)
                
        # Render views
        rendered_paths = self.render_multiview(views, output_name, elevation, azimuth_offset)
        
        # Combine if requested
        if combine_mode == 'horizontal':
            combined_path = self.output_dir / f"{output_name}_combined.png"
            self.combine_views_horizontal(rendered_paths, str(combined_path))
            return combined_path
        elif combine_mode == 'grid':
            combined_path = self.output_dir / f"{output_name}_combined.png"
            grid_size = (2, 2) if len(views) == 4 else (1, len(views))
            self.combine_views_grid(rendered_paths, str(combined_path), grid_size)
            return combined_path
        else:
            return rendered_paths
            
    def render_scene(self, whole_name: str, part_names: list, sample_id: str,
                    views: list = ['front', 'back', 'left', 'right'],
                    combine_mode: str = 'grid', elevation: float = 0.0,
                    azimuth_offset: float = 0.0):
        """Render complete scene with whole object and parts.
        
        Args:
            whole_name: Name of the whole object
            part_names: List of part object names
            sample_id: Sample identifier
            views: List of views to render
            combine_mode: How to combine views
            elevation: Elevation angle offset
            azimuth_offset: Azimuth angle offset
            
        Returns:
            Dictionary with paths to rendered images
        """
        results = {}
        
        # Render whole object
        whole_path = self.render_object(
            whole_name, views, f"{sample_id}_whole", 
            combine_mode, elevation, azimuth_offset
        )
        results['whole'] = whole_path
        
        # Render each part
        results['parts'] = {}
        for part_name in part_names:
            part_path = self.render_object(
                part_name, views, f"{sample_id}_part_{part_name}",
                combine_mode, elevation, azimuth_offset
            )
            results['parts'][part_name] = part_path
            
        # Restore visibility
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.hide_render = False
                
        return results


def main():
    """Main function for command-line usage."""
    # Parse arguments (after --)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    # Example usage
    output_dir = argv[0] if len(argv) > 0 else "./output"
    
    renderer = MultiViewRenderer(output_dir)
    renderer.setup_lighting()
    
    # Example: render current scene
    # You would need to specify object names based on your scene
    print(f"Renderer initialized. Output directory: {output_dir}")
    print("Use renderer.render_scene() to render objects.")


if __name__ == "__main__":
    main()
