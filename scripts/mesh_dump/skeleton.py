from typing import List, Optional, Dict
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .bone import Bone


class Skeleton:
    """Skeleton class for managing multiple bones"""
    
    def __init__(self, name: str = ""):
        """
        Initialize skeleton
        
        Args:
            name: Skeleton name
        """
        self.name = name
        self.bones: List[Bone] = []
        self._bone_map: Dict[str, Bone] = {}  # Name to bone mapping
    
    def add_bone(self, bone: Bone) -> None:
        """
        Add a bone to the skeleton
        
        Args:
            bone: Bone instance to add
        """
        self.bones.append(bone)
        self._bone_map[bone.name] = bone
    
    def get_bone_by_name(self, name: str) -> Optional[Bone]:
        """
        Get bone by name
        
        Args:
            name: Bone name
            
        Returns:
            Bone instance or None if not found
        """
        return self._bone_map.get(name)
    
    def get_bone_by_index(self, index: int) -> Optional[Bone]:
        """
        Get bone by index
        
        Args:
            index: Bone index
            
        Returns:
            Bone instance or None if index out of range
        """
        for bone in self.bones:
            if bone.index == index:
                return bone
        return None
        
    def get_bone_count(self) -> int:
        """
        Get number of bones
        
        Returns:
            Bone count
        """
        return len(self.bones)
    
    def remove_bone(self, name: str) -> bool:
        """
        Remove bone by name
        
        Args:
            name: Bone name
            
        Returns:
            True if bone was removed, False if not found
        """
        bone = self._bone_map.get(name)
        if bone:
            self.bones.remove(bone)
            del self._bone_map[name]
            return True
        return False
    
    def clear_bones(self) -> None:
        """Clear all bones"""
        self.bones.clear()
        self._bone_map.clear()
    
    def build_hierarchy(self) -> None:
        """
        Build bone hierarchy by setting parent references based on parent indices
        """
        for bone in self.bones:
            if bone.parent_index is not None:
                parent = self.get_bone_by_index(bone.parent_index)
                if parent:
                    bone.parent = parent
    
    def get_root_bones(self) -> List[Bone]:
        """
        Get all root bones (bones without parents)
        
        Returns:
            List of root Bone instances
        """
        return [bone for bone in self.bones if not bone.has_parent()]
    
    def get_children(self, bone: Bone) -> List[Bone]:
        """
        Get all children of a bone
        
        Args:
            bone: Parent bone
            
        Returns:
            List of child Bone instances
        """
        return [b for b in self.bones if b.parent_index == bone.index]
    
    def scale(self, scale_factor: float) -> None:
        """
        Scale all bone positions by a factor
        
        Args:
            scale_factor: Scale factor to apply to all bone positions
        """
        for bone in self.bones:
            bone.position *= scale_factor
    
    def rotate_by_x(self, angle: float) -> None:
        """
        Rotate root bones around X axis (position and rotation)
        Child bones will follow automatically due to hierarchy
        
        Args:
            angle: Rotation angle in degrees
        """
        # Create rotation object for X axis
        rotation = R.from_euler('x', angle, degrees=True)
        
        # Only rotate root bones (bones without parent)
        for bone in self.bones:
            if not bone.has_parent():
                # Rotate position
                bone.position = rotation.apply(bone.position)
                
                # Rotate quaternion: combine rotations
                bone_rotation = R.from_quat(bone.rotation)
                new_rotation = rotation * bone_rotation
                bone.rotation = new_rotation.as_quat()
    
    def copy(self) -> 'Skeleton':
        """
        Create a deep copy of the skeleton
        
        Returns:
            New Skeleton instance with copied bones
        """
        skeleton = Skeleton(self.name)
        
        # Copy all bones
        for bone in self.bones:
            skeleton.add_bone(bone.copy())
        
        # Rebuild hierarchy (set parent references)
        skeleton.build_hierarchy()
        
        return skeleton
    
    def to_json(self) -> Dict:
        """
        Convert skeleton to JSON format
        
        Returns:
            Dict containing skeleton data
        """
        return {
            "name": self.name,
            "bone_count": len(self.bones),
            "bones": [bone.to_json() for bone in self.bones]
        }
            
    def to_json_file(self, json_path: str) -> None:
        """
        Convert skeleton to JSON format and save to file
        
        Args:
            json_path: Path to save JSON file
        """
        skeleton_data = self.to_json()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        return f"Skeleton(name='{self.name}', bones={len(self.bones)})"
