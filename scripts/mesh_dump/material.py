from typing import List, Optional, Dict
from .material_param import MaterialParam


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
    
    def __repr__(self) -> str:
        return f"Material(name='{self.name}', params={len(self._parameters)})"
