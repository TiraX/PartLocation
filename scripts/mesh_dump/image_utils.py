import os
import numpy as np
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
    
    @staticmethod
    def normalize_normalmap(texture_path: str, output_path: str = None) -> str:
        """
        Normalize a normal map by calculating the B (blue) channel from R and G channels
        
        Normal maps store directional vectors where each pixel represents a surface normal.
        The formula used is: B = sqrt(1 - R² - G²)
        where R, G, B are normalized to [0, 1] range (with 0.5 representing 0 in tangent space)
        
        Args:
            texture_path: Path to the source normal map texture file
            output_path: Optional path for the output file. If None, will append '_normalized' to the original filename
        
        Returns:
            Path to the normalized normal map file
        
        Raises:
            FileNotFoundError: If texture_path does not exist
            ValueError: If image cannot be loaded or has invalid format
        """
        if not os.path.exists(texture_path):
            raise FileNotFoundError(f"Normal map file not found: {texture_path}")
        
        # Load the image
        img = Image.open(texture_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array for calculation
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Extract R and G channels
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        
        # Convert from [0, 1] to [-1, 1] range (normal map space)
        r_normalized = r_channel * 2.0 - 1.0
        g_normalized = g_channel * 2.0 - 1.0
        
        # Calculate B channel: B = sqrt(1 - R² - G²)
        # Clamp the value to avoid negative numbers due to floating point errors
        b_squared = np.maximum(0.0, 1.0 - r_normalized**2 - g_normalized**2)
        b_normalized = np.sqrt(b_squared)
        
        # Convert back to [0, 1] range
        b_channel = (b_normalized + 1.0) / 2.0
        
        # Update the B channel in the image array
        img_array[:, :, 2] = b_channel
        
        # Convert back to uint8
        img_array = (img_array * 255.0).astype(np.uint8)
        
        # Create output image
        output_img = Image.fromarray(img_array, 'RGB')
        
        # Determine output path
        if output_path is None:
            base_path = os.path.dirname(texture_path)
            filename = os.path.basename(texture_path)
            name_without_ext, ext = os.path.splitext(filename)
            output_path = os.path.join(base_path, f"{name_without_ext}_normalized{ext}")
        
        # Save the normalized normal map
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_img.save(output_path)
        
        return output_path
