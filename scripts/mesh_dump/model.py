from typing import List, Optional, Tuple
import numpy as np
from .mesh import Mesh
from .skeleton import Skeleton


class Model:
    """Model class for managing multiple meshes and optional skeleton"""
    
    def __init__(self, name: str = ""):
        """
        Initialize model
        
        Args:
            name: Model name
        """
        self.name = name
        self.meshes: List[Mesh] = []
        self.skeleton: Optional[Skeleton] = None

    @staticmethod
    def merge(models: List['Model']) -> 'Model':
        """
        Merge multiple models into one
        
        Args:
            models: List of Model instances
            
        Returns:
            Merged Model instance
        """
        if not models:
            return Model()
        
        # Create merged model with name from first model
        merged = Model(name="merged")
        
        # Add all meshes from all models
        for model in models:
            for mesh in model.meshes:
                merged.add_mesh(mesh)
        
        # Use skeleton from first model that has one
        for model in models:
            if model.has_skeleton():
                merged.skeleton = model.skeleton
                break
        
        return merged
    
    # Mesh management
    def add_mesh(self, mesh: Mesh) -> None:
        """
        Add a mesh to the model
        
        Args:
            mesh: Mesh instance to add
        """
        self.meshes.append(mesh)
    
    def get_mesh_by_name(self, name: str) -> Optional[Mesh]:
        """
        Get mesh by name
        
        Args:
            name: Mesh name
            
        Returns:
            First Mesh instance with matching name or None if not found
        """
        for mesh in self.meshes:
            if mesh.name == name:
                return mesh
        return None
    
    def get_mesh_count(self) -> int:
        """
        Get number of meshes
        
        Returns:
            Mesh count
        """
        return len(self.meshes)
    
    def remove_mesh(self, index: int) -> bool:
        """
        Remove mesh by index
        
        Args:
            index: Mesh index
            
        Returns:
            True if mesh was removed, False if index out of range
        """
        if 0 <= index < len(self.meshes):
            self.meshes.pop(index)
            return True
        return False
    
    def clear_meshes(self) -> None:
        """Clear all meshes"""
        self.meshes.clear()
        
    def has_skeleton(self) -> bool:
        """
        Check if model has a skeleton
        
        Returns:
            True if skeleton is set
        """
        return self.skeleton is not None
    
    def is_skinned(self) -> bool:
        """
        Check if model is skinned (has skeleton and at least one mesh with skinning data)
        
        Returns:
            True if model has skeleton and skinned meshes
        """
        if not self.has_skeleton():
            return False
        return any(mesh.has_skinning_data() for mesh in self.meshes)
    
    def get_total_vertex_count(self) -> int:
        """
        Get total vertex count across all meshes
        
        Returns:
            Total vertex count
        """
        return sum(mesh.get_vertex_count() for mesh in self.meshes)
    
    def get_total_face_count(self) -> int:
        """
        Get total face count across all meshes
        
        Returns:
            Total face count
        """
        return sum(mesh.get_face_count() for mesh in self.meshes)
    
    def scale(self, scale_factor: float) -> None:
        """
        Scale all mesh positions and skeleton by a factor
        
        Args:
            scale_factor: Scale factor to apply to all positions
        """
        for mesh in self.meshes:
            mesh.scale(scale_factor)
        
        if self.has_skeleton():
            self.skeleton.scale(scale_factor)
    
    def rotate_by_x(self, angle: float) -> None:
        """
        Rotate all meshes and skeleton around X axis
        
        Args:
            angle: Rotation angle in degrees
        """
        for mesh in self.meshes:
            mesh.rotate_by_x(angle)
        
        if self.has_skeleton():
            self.skeleton.rotate_by_x(angle)
    
    def get_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate merged bounds of all meshes
        
        Returns:
            Tuple of (bounds_min, bounds_max) as 3D arrays, or None if no valid bounds
        """
        valid_bounds = []
        
        for mesh in self.meshes:
            if mesh.bounds_min is not None and mesh.bounds_max is not None:
                valid_bounds.append((mesh.bounds_min, mesh.bounds_max))
        
        if not valid_bounds:
            return None
        
        # Calculate merged bounds
        all_mins = np.array([bounds[0] for bounds in valid_bounds])
        all_maxs = np.array([bounds[1] for bounds in valid_bounds])
        
        merged_min = np.min(all_mins, axis=0)
        merged_max = np.max(all_maxs, axis=0)
        
        return (merged_min, merged_max)
    
    def normalize(self, min_val: float, max_val: float, padding: float) -> None:
        """
        Normalize all mesh vertices to specified range with padding
        
        Args:
            min_val: Minimum value of target range
            max_val: Maximum value of target range
            padding: Padding ratio (0.0 to 1.0)
        """
        bounds = self.get_bounds()
        if bounds is None:
            return
        
        bounds_min, bounds_max = bounds
        bounds_size = bounds_max - bounds_min
        bounds_center = (bounds_min + bounds_max) * 0.5
        
        # Calculate target range considering padding
        target_range = max_val - min_val
        effective_range = target_range * (1.0 - 2.0 * padding)
        
        # Calculate scale factor based on the largest dimension of model bounds
        max_dimension = np.max(bounds_size)
        if max_dimension > 0:
            scale_factor = effective_range / max_dimension
        else:
            scale_factor = 1.0
        
        # Calculate target center
        target_center = (min_val + max_val) * 0.5
        
        # Transform each mesh's vertices based on global bounds
        # We cannot use mesh.normalize() because it normalizes based on each mesh's own bounds
        # Instead, we manually transform vertices using the global transformation
        for mesh in self.meshes:
            positions = mesh.get_vertex_attribute(Mesh.POSITION)
            if positions is None:
                continue
            
            # Apply global transformation:
            # 1. Center to origin (using global bounds_center)
            # 2. Scale (using global scale_factor)
            # 3. Move to target center
            positions = (positions - bounds_center) * scale_factor + target_center
            
            # Update positions (this will also update mesh's bounds)
            mesh.set_vertex_attribute(Mesh.POSITION, positions)
        
        # Scale skeleton if exists
        if self.has_skeleton():
            # Center skeleton
            self.skeleton.translate(-bounds_center)
            # Scale skeleton
            self.skeleton.scale(scale_factor)
            # Move to target center
            self.skeleton.translate(target_center)
    
    def split(self) -> List['Model']:
        """
        Split model into multiple models, one for each mesh
        Each mesh and skeleton (if exists) will be deep copied
        
        Returns:
            List of Model instances, each containing one mesh
        """
        models = []
        
        for i, mesh in enumerate(self.meshes):
            # Create new model with mesh name or index
            mesh_name = mesh.name if mesh.name else f"{self.name}_mesh_{i}"
            model = Model(name=mesh_name)
            
            # Deep copy mesh to avoid sharing data
            model.add_mesh(mesh.copy())
            
            # Deep copy skeleton if exists
            if self.has_skeleton():
                model.skeleton = self.skeleton.copy()
            
            models.append(model)
        
        return models
    
    def __repr__(self) -> str:
        skeleton_info = f", skeleton='{self.skeleton.name}'" if self.skeleton else ""
        return f"Model(name='{self.name}', meshes={len(self.meshes)}{skeleton_info})"
