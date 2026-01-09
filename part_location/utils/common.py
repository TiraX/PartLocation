"""Common utility functions for the project."""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary containing YAML data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data to YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Path to output YAML file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion.
    
    Args:
        q: Quaternion as [qw, qx, qy, qz]
        
    Returns:
        Normalized quaternion
    """
    return q / np.linalg.norm(q)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)
        
    Returns:
        Quaternion as [qw, qx, qy, qz]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q: np.ndarray) -> tuple:
    """Convert quaternion to Euler angles.
    
    Args:
        q: Quaternion as [qw, qx, qy, qz]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    qw, qx, qy, qz = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
