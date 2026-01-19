import os
from PIL import Image
from typing import Dict, Tuple, List
from enum import Enum
from .material_param import TextureType


class ChannelType(Enum):
    """Enum for texture channel types"""
    RED = 'r'
    GREEN = 'g'
    BLUE = 'b'
    ALPHA = 'a'


class ImageUtil:
    """Utility class for image processing operations"""
    
    @staticmethod
    def split_texture_channels(
        texture_path: str,
        channel_mapping: Dict[ChannelType, TextureType]
    ) -> List[Tuple[str, TextureType]]:
        """
        Split a multi-channel texture into separate texture files
        
        Args:
            texture_path: Path to the source texture file
            channel_mapping: Dictionary mapping channel types to TextureType
                           e.g., {ChannelType.RED: TextureType.AMBIENT_OCCLUSION, ChannelType.GREEN: TextureType.ROUGHNESS}
        
        Returns:
            List of tuples containing (texture_path, TextureType)
            e.g., [('path/to/texture_roughness.png', TextureType.ROUGHNESS), ('path/to/texture_metallic.png', TextureType.METALLIC)]
        
        Raises:
            FileNotFoundError: If texture_path does not exist
            ValueError: If image cannot be loaded or has invalid format
        """
        if not os.path.exists(texture_path):
            raise FileNotFoundError(f"Texture file not found: {texture_path}")
        
        # Load the image
        img = Image.open(texture_path)
        
        # Convert to RGB/RGBA if needed
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGBA' if 'A' in img.mode else 'RGB')
        
        # Get base path and filename
        base_path = os.path.dirname(texture_path)
        filename = os.path.basename(texture_path)
        name_without_ext, ext = os.path.splitext(filename)
        
        # Remove common suffixes like _ORM if present
        if name_without_ext.endswith('_ORM'):
            name_without_ext = name_without_ext[:-4]
        
        result_list = []
        
        # Split channels
        channels = img.split()
        channel_map = {
            ChannelType.RED: 0,
            ChannelType.GREEN: 1,
            ChannelType.BLUE: 2,
            ChannelType.ALPHA: 3 if len(channels) > 3 else None
        }
        
        for channel_type, texture_type in channel_mapping.items():
            channel_index = channel_map.get(channel_type)
            
            if channel_index is None or channel_index >= len(channels):
                continue
            
            # Generate output filename based on texture type
            texture_type_name = texture_type.value
            output_filename = f"{name_without_ext}_{texture_type_name}.png"
            output_path = os.path.join(base_path, output_filename)
            
            # Check if file already exists
            if not os.path.exists(output_path):
                # Get the specific channel
                channel_data = channels[channel_index]
                
                # Create a grayscale image from this channel
                output_img = Image.merge('L', (channel_data,))
                
                # Save the channel as a separate image
                output_img.save(output_path)
            
            result_list.append((output_path, texture_type))
        
        return result_list

