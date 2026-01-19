from typing import List, Optional, Dict
import numpy as np
from .material import Material

    
class Mesh:
    POSITION = 'position'
    NORMAL = 'normal'
    UV0 = 'uv0'
    UV1 = 'uv1'
    UV2 = 'uv2'
    UV3 = 'uv3'
    TANGENT = 'tangent'
    BINORMAL = 'binormal'
    BLEND_INDICES = 'blend_indices'
    BLEND_WEIGHTS = 'blend_weights'
    
    MAX_UV_LAYERS = 4
    """Mesh class for storing geometry data and materials"""
    
    def __init__(self, name: str = ""):
        """
        Initialize mesh
        
        Args:
            name: Mesh name
        """
        self.name = name
        self._vertices: Dict[str, np.ndarray] = {}  # Dictionary for vertex attributes, eg: {'position': np.ndarray, 'normal': np.ndarray, ...}
        self.materials: List[Material] = []
        self._sections: List[Dict] = []  # List of material sections
        self.faces: Optional[np.ndarray] = None  # Mx3 or Mx4 array (triangles or quads)
        self.bounds_min: Optional[np.ndarray] = None  # Minimum bounds (x, y, z)
        self.bounds_max: Optional[np.ndarray] = None  # Maximum bounds (x, y, z)
    
    # Vertex attribute management
    def set_vertex_attribute(self, name: str, data: np.ndarray) -> None:
        """Set vertex attribute data
        
        Args:
            name: Attribute name (e.g., 'position', 'normal', 'uv', 'blend_weights', 'blend_indices')
            data: Numpy array of attribute data
        """
        self._vertices[name] = data
        
        # Auto-calculate bounds when setting position data
        if name == Mesh.POSITION and data is not None and len(data) > 0:
            self.bounds_min = np.min(data, axis=0)
            self.bounds_max = np.max(data, axis=0)
    
    def get_vertex_attribute(self, name: str) -> Optional[np.ndarray]:
        """Get vertex attribute data
        
        Args:
            name: Attribute name
            
        Returns:
            Numpy array or None if attribute doesn't exist
        """
        return self._vertices.get(name)
    
    def has_vertex_attribute(self, name: str) -> bool:
        """Check if vertex attribute exists
        
        Args:
            name: Attribute name
            
        Returns:
            True if attribute exists
        """
        return name in self._vertices
    
    def get_vertex_attribute_names(self) -> List[str]:
        """Get all vertex attribute names
        
        Returns:
            List of attribute names
        """
        return list(self._vertices.keys())
        
    # Material management
    def add_material(self, material: Material) -> None:
        """
        Add a material to the mesh
        
        Args:
            material: Material instance to add
        """
        self.materials.append(material)
        
    def clear_materials(self) -> None:
        """Clear all materials"""
        self.materials.clear()
    
    # Section management
    def add_section(self, start_face: int, face_count: int, material_index: int = 0) -> None:
        """
        Add a material section to the mesh
        
        Args:
            start_face: Starting face index
            face_count: Number of faces in this section
            material_index: Index of material to use for this section
        """
        section = {
            'start_face': start_face,
            'face_count': face_count,
            'material_index': material_index
        }
        self._sections.append(section)
    
    def get_sections(self) -> List[Dict]:
        """
        Get all material sections
        
        Returns:
            List of section dictionaries
        """
        return self._sections
    
    def clear_sections(self) -> None:
        """Clear all sections"""
        self._sections.clear()
    
    
    def has_skinning_data(self) -> bool:
        """Check if mesh has skinning data
        
        Returns:
            True if bone weights and indices are set
        """
        return Mesh.BLEND_WEIGHTS in self._vertices and Mesh.BLEND_INDICES in self._vertices
    
    def get_vertex_count(self) -> int:
        """Get number of vertices
        
        Returns:
            Vertex count or 0 if vertices not set
        """
        position = self._vertices.get(Mesh.POSITION)
        return len(position) if position is not None else 0
    
    def get_face_count(self) -> int:
        """
        Get number of faces
        
        Returns:
            Face count or 0 if faces not set
        """
        return len(self.faces) if self.faces is not None else 0
    
    def bounds_center(self) -> Optional[np.ndarray]:
        """
        Calculate the center point of the bounding box
        
        Returns:
            3D center point (x, y, z) or None if bounds not set
        """
        if self.bounds_min is not None and self.bounds_max is not None:
            return (self.bounds_min + self.bounds_max) * 0.5
        return None
    
    def bounds_size(self) -> Optional[np.ndarray]:
        """
        Calculate the size of the bounding box
        
        Returns:
            3D size (width, height, depth) or None if bounds not set
        """
        if self.bounds_min is not None and self.bounds_max is not None:
            return self.bounds_max - self.bounds_min
        return None
    
    def scale(self, scale_factor: float) -> None:
        """
        Scale all vertex positions by a factor
        
        Args:
            scale_factor: Scale factor to apply to all positions
        """
        positions = self.get_vertex_attribute(Mesh.POSITION)
        if positions is not None:
            self.set_vertex_attribute(Mesh.POSITION, positions * scale_factor)
    
    def normalize(self, min_val: float, max_val: float, padding: float) -> None:
        """
        Normalize mesh vertices to specified range with padding
        
        Args:
            min_val: Minimum value of target range
            max_val: Maximum value of target range
            padding: Padding ratio (0.0 to 1.0)
        """
        if self.bounds_min is None or self.bounds_max is None:
            return
        
        bounds_size = self.bounds_max - self.bounds_min
        bounds_center = (self.bounds_min + self.bounds_max) * 0.5
        
        # Calculate target range considering padding
        target_range = max_val - min_val
        effective_range = target_range * (1.0 - 2.0 * padding)
        
        # Calculate scale factor based on the largest dimension
        max_dimension = np.max(bounds_size)
        if max_dimension > 0:
            scale_factor = effective_range / max_dimension
        else:
            scale_factor = 1.0
        
        # Calculate target center
        target_center = (min_val + max_val) * 0.5
        
        # Transform positions
        positions = self.get_vertex_attribute(Mesh.POSITION)
        if positions is not None:
            # Center to origin
            positions = positions - bounds_center
            # Scale
            positions = positions * scale_factor
            # Move to target center
            positions = positions + target_center
            # Update positions (this will also update bounds)
            self.set_vertex_attribute(Mesh.POSITION, positions)
    
    def rotate_by_x(self, angle: float) -> None:
        """
        Rotate mesh around X axis
        
        Args:
            angle: Rotation angle in degrees
        """
        # Convert angle from degrees to radians
        angle_rad = np.radians(angle)
        
        # Create rotation matrix for X axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        
        # Rotate positions
        positions = self.get_vertex_attribute(Mesh.POSITION)
        if positions is not None:
            new_positions = positions @ rotation_matrix.T
            self.set_vertex_attribute(Mesh.POSITION, new_positions)
        
        # Rotate normals
        normals = self.get_vertex_attribute(Mesh.NORMAL)
        if normals is not None:
            new_normals = normals @ rotation_matrix.T
            self.set_vertex_attribute(Mesh.NORMAL, new_normals)
        
        # Rotate tangents
        tangents = self.get_vertex_attribute(Mesh.TANGENT)
        if tangents is not None:
            # Tangents might be 4D (with W component for handedness)
            if tangents.shape[1] == 4:
                # Rotate only XYZ, keep W unchanged
                tangents_xyz = tangents[:, :3] @ rotation_matrix.T
                self.set_vertex_attribute(Mesh.TANGENT, np.column_stack([tangents_xyz, tangents[:, 3]]))
            else:
                # 3D tangents
                self.set_vertex_attribute(Mesh.TANGENT, tangents @ rotation_matrix.T)
    
    def copy(self) -> 'Mesh':
        """
        Create a deep copy of the mesh
        
        Returns:
            New Mesh instance with copied data
        """
        mesh = Mesh(self.name)
        
        # Copy all vertex attributes
        for name, data in self._vertices.items():
            mesh.set_vertex_attribute(name, data.copy())
        
        # Copy faces
        if self.faces is not None:
            mesh.faces = self.faces.copy()
        
        # Copy materials
        for material in self.materials:
            mesh.add_material(material.copy())
        
        # Copy sections
        for section in self._sections:
            mesh.add_section(
                section['start_face'],
                section['face_count'],
                section['material_index']
            )
        
        # Bounds will be automatically set when copying position attribute
        
        return mesh
    
    def __repr__(self) -> str:
        return f"Mesh(name='{self._name}', vertices={self.get_vertex_count()}, faces={self.get_face_count()}, materials={len(self.materials)})"
