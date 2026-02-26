import os
from typing import List, Optional, Dict
from .material_param import MaterialParam, TextureParam, NumericParam, TextureType


class Material:
    """Material class for managing material parameters"""
    
    def __init__(self, name: str):
        """
        Initialize material
        
        Args:
            name: Material name
        """
        self.name = name
        self._parameters: Dict[str, MaterialParam] = {}
    
    def add_parameter(self, param: MaterialParam) -> None:
        """
        Add a parameter to the material
        
        Args:
            param: MaterialParam instance to add
        """
        self._parameters[param.name] = param
    
    def get_parameter(self, name: str) -> Optional[MaterialParam]:
        """
        Get parameter by name
        
        Args:
            name: Parameter name
            
        Returns:
            MaterialParam instance or None if not found
        """
        return self._parameters.get(name)
    
    def get_all_parameters(self) -> List[MaterialParam]:
        """
        Get all parameters
        
        Returns:
            List of all MaterialParam instances
        """
        return list(self._parameters.values())
    
    def remove_parameter(self, name: str) -> bool:
        """
        Remove parameter by name
        
        Args:
            name: Parameter name
            
        Returns:
            True if parameter was removed, False if not found
        """
        if name in self._parameters:
            del self._parameters[name]
            return True
        return False
    
    def has_parameter(self, name: str) -> bool:
        """
        Check if parameter exists
        
        Args:
            name: Parameter name
            
        Returns:
            True if parameter exists, False otherwise
        """
        return name in self._parameters
    
    def clear_parameters(self) -> None:
        """Clear all parameters"""
        self._parameters.clear()
    
    def copy(self) -> 'Material':
        """
        Create a deep copy of the material
        
        Returns:
            New Material instance with copied parameters
        """
        material = Material(self.name)
        
        # Copy all parameters
        for param in self._parameters.values():
            material.add_parameter(param.copy())
        
        return material
    
    def dump_json(self, texture_relative_root: str) -> Dict:
        """
        Dump material to JSON-compatible dictionary
        
        Args:
            texture_relative_root: Root path for relative texture paths
            
        Returns:
            Dictionary containing material data with Textures, Parameters, Classified, and UVs sections
        """
        result = {
            "Textures": {},
            "Parameters": {},
            "UVs": {},
            "Classified": {}
        }
        
        # Process all parameters
        for param in self._parameters.values():
            if isinstance(param, TextureParam):
                texture_relative_path = os.path.relpath(param.texture_path, texture_relative_root)
                texture_relative_path = texture_relative_path.replace('\\', '/')
                # Add to Textures section
                result["Textures"][param.name] = texture_relative_path
                
                # Add to UVs section
                result["UVs"][param.name] = f"UVChannel{param.uv_layer}"
                
                # Add to Classified section if it's one of the special types
                if param.texture_type == TextureType.DIFFUSE:
                    result["Classified"]["Diffuse"] = texture_relative_path
                elif param.texture_type == TextureType.NORMAL:
                    result["Classified"]["Normal"] = texture_relative_path
                elif param.texture_type == TextureType.METALLIC:
                    result["Classified"]["Metallic"] = texture_relative_path
                elif param.texture_type == TextureType.ROUGHNESS:
                    result["Classified"]["Roughness"] = texture_relative_path
            elif isinstance(param, NumericParam):
                # Add to Parameters section
                result["Parameters"][param.name] = param.value
        
        return result
    
    def __repr__(self) -> str:
        return f"Material(name='{self.name}', params={len(self._parameters)})"
