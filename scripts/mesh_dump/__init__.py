"""
Mesh Dump Module

This module provides classes for representing 3D model data including:
- Material parameters (numeric and texture)
- Materials
- Meshes with geometry and skinning data
- Bones and skeletons
- Models
- Animation tracks and animations
"""

# Material parameter classes
from .material_param import (
    MaterialParam,
    NumericParam,
    TextureParam,
    TextureType
)

# Material class
from .material import Material

# Mesh class
from .mesh import Mesh

# Bone and skeleton classes
from .bone import Bone
from .skeleton import Skeleton

# Model class
from .model import Model

# Animation classes
from .track import (
    Track,
    Keyframe,
    InterpolationType
)
from .anim import Anim

# Define public API
__all__ = [
    # Material parameters
    'MaterialParam',
    'NumericParam',
    'TextureParam',
    'TextureType',
    
    # Material
    'Material',
    
    # Mesh
    'Mesh',
    
    # Skeleton
    'Bone',
    'Skeleton',
    
    # Model
    'Model',
    
    # Animation
    'Track',
    'Keyframe',
    'InterpolationType',
    'Anim',
]

__version__ = '1.0.0'
