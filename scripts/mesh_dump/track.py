from typing import List, Tuple, Any
from enum import Enum
import numpy as np


class InterpolationType(Enum):
    """Enum for keyframe interpolation types"""
    LINEAR = "linear"
    STEP = "step"
    BEZIER = "bezier"
    CUBIC = "cubic"


class Keyframe:
    """Keyframe class for storing time-value pairs"""
    
    def __init__(self, time: float, value: Any):
        """
        Initialize keyframe
        
        Args:
            time: Time in seconds
            value: Keyframe value (can be float, vector, quaternion, etc.)
        """
        self.time = time
        self.value = value
    
    def __repr__(self) -> str:
        return f"Keyframe(time={self.time}, value={self.value})"


class Track:
    """Track class for storing animation data for a specific target"""
    
    def __init__(self, target: str, property_path: str = ""):
        """
        Initialize track
        
        Args:
            target: Target name (e.g., bone name)
            property_path: Property path (e.g., "position", "rotation", "scale")
        """
        self.target = target
        self.property_path = property_path
        self.keyframes: List[Keyframe] = []
        self.interpolation_type: InterpolationType = InterpolationType.LINEAR
    
    def add_keyframe(self, time: float, value: Any) -> None:
        """
        Add a keyframe to the track
        
        Args:
            time: Time in seconds
            value: Keyframe value
        """
        keyframe = Keyframe(time, value)
        self.keyframes.append(keyframe)
        # Keep keyframes sorted by time
        self.keyframes.sort(key=lambda k: k.time)
    
    def get_keyframes(self) -> List[Keyframe]:
        """
        Get all keyframes
        
        Returns:
            List of Keyframe instances
        """
        return self.keyframes
    
    def get_keyframe_count(self) -> int:
        """
        Get number of keyframes
        
        Returns:
            Keyframe count
        """
        return len(self.keyframes)
    
    def clear_keyframes(self) -> None:
        """Clear all keyframes"""
        self.keyframes.clear()
    
    def get_value_at_time(self, time: float) -> Any:
        """
        Get interpolated value at specific time
        
        Args:
            time: Time in seconds
            
        Returns:
            Interpolated value or None if no keyframes
        """
        if not self.keyframes:
            return None
        
        # If time is before first keyframe, return first value
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        
        # If time is after last keyframe, return last value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        
        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            k1 = self.keyframes[i]
            k2 = self.keyframes[i + 1]
            
            if k1.time <= time <= k2.time:
                # Interpolate based on interpolation type
                if self.interpolation_type == InterpolationType.STEP:
                    return k1.value
                elif self.interpolation_type == InterpolationType.LINEAR:
                    # Linear interpolation
                    t = (time - k1.time) / (k2.time - k1.time)
                    return self._lerp(k1.value, k2.value, t)
                else:
                    # For BEZIER and CUBIC, use linear for now (can be extended)
                    t = (time - k1.time) / (k2.time - k1.time)
                    return self._lerp(k1.value, k2.value, t)
        
        return None
    
    def _lerp(self, v1: Any, v2: Any, t: float) -> Any:
        """
        Linear interpolation between two values
        
        Args:
            v1: Start value
            v2: End value
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated value
        """
        # Handle numpy arrays
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return v1 + (v2 - v1) * t
        
        # Handle lists
        if isinstance(v1, list) and isinstance(v2, list):
            return [v1[i] + (v2[i] - v1[i]) * t for i in range(len(v1))]
        
        # Handle scalars
        return v1 + (v2 - v1) * t
    
    def get_time_range(self) -> Tuple[float, float]:
        """
        Get time range of the track
        
        Returns:
            Tuple of (start_time, end_time) or (0, 0) if no keyframes
        """
        if not self.keyframes:
            return (0.0, 0.0)
        return (self.keyframes[0].time, self.keyframes[-1].time)
    
    def __repr__(self) -> str:
        return f"Track(target='{self.target}', property='{self.property_path}', keyframes={len(self.keyframes)})"
