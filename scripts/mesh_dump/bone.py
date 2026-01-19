from typing import Optional, List
import numpy as np


class Bone:
    """Bone class for storing bone transformation data"""
    
    def __init__(self, name: str, index: int):
        """
        Initialize bone
        
        Args:
            name: Bone name
            index: Bone index in skeleton
        """
        self.name = name
        self.index = index
        
        # Transform data
        self.position: np.ndarray = np.array([0.0, 0.0, 0.0])  # Translation vector
        self.rotation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)
        self.scale: np.ndarray = np.array([1.0, 1.0, 1.0])  # Scale vector
        
        # Hierarchy
        self.parent_index: Optional[int] = None
        self._parent: Optional['Bone'] = None
    
    @property
    def parent(self) -> Optional['Bone']:
        """Get parent bone reference"""
        return self._parent
    
    @parent.setter
    def parent(self, value: Optional['Bone']):
        """Set parent bone reference and update parent_index"""
        self._parent = value
        if value is not None:
            self.parent_index = value.index
    
    def has_parent(self) -> bool:
        """
        Check if bone has a parent
        
        Returns:
            True if parent is set
        """
        return self._parent is not None or self.parent_index is not None
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix from position, rotation, and scale
        
        Returns:
            4x4 transformation matrix
        """
        # Create rotation matrix from quaternion
        x, y, z, w = self.rotation
        rot_matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 0],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
            [0, 0, 0, 1]
        ])
        
        # Create scale matrix
        scale_matrix = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])
        
        # Combine: Scale * Rotation
        transform = rot_matrix @ scale_matrix
        
        # Add translation
        transform[0:3, 3] = self.position
        
        return transform
    
    def copy(self) -> 'Bone':
        """
        Create a deep copy of the bone
        
        Returns:
            New Bone instance with copied data
        """
        bone = Bone(self.name, self.index)
        bone.position = self.position.copy()
        bone.rotation = self.rotation.copy()
        bone.scale = self.scale.copy()
        bone.parent_index = self.parent_index
        # Note: parent reference is not copied, should be set after copying skeleton
        return bone
    
    def to_json(self) -> dict:
        """
        Convert bone to JSON-serializable dictionary
        
        Returns:
            Dictionary representation of bone
        """
        return {
            "name": self.name,
            "index": self.index,
            "parent_index": self.parent_index,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "scale": self.scale.tolist()
        }
    
    def __repr__(self) -> str:
        parent_info = f", parent={self.parent_index}" if self.parent_index is not None else ""
        return f"Bone(name='{self.name}', index={self.index}{parent_info})"
