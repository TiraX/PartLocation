from abc import ABC, abstractmethod
from typing import Union, List
from enum import Enum


class TextureType(Enum):
    """Enum for texture usage types"""
    DIFFUSE = "diffuse"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    SPECULAR = "specular"
    EMISSIVE = "emissive"
    AMBIENT_OCCLUSION = "ambient_occlusion"
    OPACITY = "opacity"
    UNKNOWN = "unknown"


class MaterialParam(ABC):
    """Base class for material parameters"""
    
    def __init__(self, name: str, param_type: str):
        """
        Initialize material parameter
        
        Args:
            name: Parameter name
            param_type: Parameter type identifier
        """
        self.name = name
        self.param_type = param_type
    
class NumericParam(MaterialParam):
    """Numeric parameter class for storing float or vector values"""
    
    def __init__(self, name: str, value: Union[float, List[float]] = 0.0):
        """
        Initialize numeric parameter
        
        Args:
            name: Parameter name
            value: Numeric value (float or list of floats for vectors)
        """
        super().__init__(name, "numeric")
        self.value = value
        
    def is_vector(self) -> bool:
        """Check if value is a vector (list)"""
        return isinstance(self.value, list)
    
    def copy(self) -> 'NumericParam':
        """Create a deep copy of the numeric parameter"""
        if isinstance(self.value, list):
            return NumericParam(self.name, self.value.copy())
        else:
            return NumericParam(self.name, self.value)
    
    def __repr__(self) -> str:
        return f"NumericParam(name='{self.name}', value={self.value})"


class TextureParam(MaterialParam):
    """Texture parameter class for storing texture path and usage type"""
    
    def __init__(self, name: str, texture_path: str = "", texture_type: TextureType = TextureType.UNKNOWN, uv_layer: int = 0):
        """
        Initialize texture parameter
        
        Args:
            name: Parameter name
            texture_path: Path to texture file
            texture_type: Texture usage type (diffuse, normal, etc.)
        """
        super().__init__(name, "texture")
        self.texture_path = texture_path
        self.texture_type = texture_type
        self.uv_layer = uv_layer
        
    @property
    def texture_path(self) -> str:
        """Get texture file path"""
        return self._texture_path_internal
    
    @texture_path.setter
    def texture_path(self, path: str):
        """Set texture file path"""
        self._texture_path_internal = path
    
    @property
    def texture_type(self) -> TextureType:
        """Get texture usage type"""
        return self._texture_type_internal
    
    @texture_type.setter
    def texture_type(self, tex_type: TextureType):
        """Set texture usage type"""
        self._texture_type_internal = tex_type
    
    def copy(self) -> 'TextureParam':
        """Create a deep copy of the texture parameter"""
        return TextureParam(self.name, self.texture_path, self.texture_type, self.uv_layer)
    
    def __repr__(self) -> str:
        return f"TextureParam(name='{self.name}', type={self.texture_type.value}, path='{self.texture_path}')"
