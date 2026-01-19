from typing import List, Optional, Dict
from .track import Track


class Anim:
    """Animation class for managing multiple tracks"""
    
    def __init__(self, name: str, duration: float = 0.0):
        """
        Initialize animation
        
        Args:
            name: Animation name
            duration: Animation duration in seconds
        """
        self.name = name
        self.duration = duration
        self._tracks: List[Track] = []
        self._track_map: Dict[str, List[Track]] = {}  # Target to tracks mapping
    
    def add_track(self, track: Track) -> None:
        """
        Add a track to the animation
        
        Args:
            track: Track instance to add
        """
        self._tracks.append(track)
        
        # Update track map
        if track.target not in self._track_map:
            self._track_map[track.target] = []
        self._track_map[track.target].append(track)
    
    def get_tracks(self) -> List[Track]:
        """
        Get all tracks
        
        Returns:
            List of Track instances
        """
        return self._tracks
    
    def get_track_count(self) -> int:
        """
        Get number of tracks
        
        Returns:
            Track count
        """
        return len(self._tracks)
    
    def get_tracks_for_target(self, target: str) -> List[Track]:
        """
        Get all tracks for a specific target
        
        Args:
            target: Target name (e.g., bone name)
            
        Returns:
            List of Track instances for the target
        """
        return self._track_map.get(target, [])
    
    def get_track_by_property(self, target: str, property_path: str) -> Optional[Track]:
        """
        Get track by target and property path
        
        Args:
            target: Target name
            property_path: Property path
            
        Returns:
            Track instance or None if not found
        """
        tracks = self.get_tracks_for_target(target)
        for track in tracks:
            if track.property_path == property_path:
                return track
        return None
    
    def remove_track(self, track: Track) -> bool:
        """
        Remove a track from the animation
        
        Args:
            track: Track instance to remove
            
        Returns:
            True if track was removed, False if not found
        """
        if track in self._tracks:
            self._tracks.remove(track)
            
            # Update track map
            if track.target in self._track_map:
                self._track_map[track.target].remove(track)
                if not self._track_map[track.target]:
                    del self._track_map[track.target]
            
            return True
        return False
    
    def clear_tracks(self) -> None:
        """Clear all tracks"""
        self._tracks.clear()
        self._track_map.clear()
    
    def get_animation_data_at_time(self, time: float) -> Dict[str, Dict[str, any]]:
        """
        Get animation data for all targets at specific time
        
        Args:
            time: Time in seconds
            
        Returns:
            Dictionary mapping target names to property-value dictionaries
        """
        result = {}
        
        for target, tracks in self._track_map.items():
            if target not in result:
                result[target] = {}
            
            for track in tracks:
                value = track.get_value_at_time(time)
                if value is not None:
                    result[target][track.property_path] = value
        
        return result
    
    def calculate_duration(self) -> float:
        """
        Calculate animation duration from all tracks
        
        Returns:
            Maximum end time across all tracks
        """
        max_time = 0.0
        for track in self._tracks:
            _, end_time = track.get_time_range()
            max_time = max(max_time, end_time)
        return max_time
    
    def update_duration_from_tracks(self) -> None:
        """Update animation duration based on track data"""
        self.duration = self.calculate_duration()
    
    def get_target_names(self) -> List[str]:
        """
        Get all unique target names
        
        Returns:
            List of target names
        """
        return list(self._track_map.keys())
    
    def __repr__(self) -> str:
        return f"Anim(name='{self.name}', duration={self.duration}s, tracks={len(self._tracks)})"
