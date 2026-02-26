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
    
    @staticmethod
    def calc_match_info(get_anim_bone_info, get_model_bone_info, cache_file: str = 'anim_model_match_info.json'):
        """
        Load or calculate animation-model bone match information
        
        Args:
            get_anim_bone_info: Function that returns list of animation bone info, format {'file': str(anim_file_path), 'bones': List[str]}
            get_model_bone_info: Function that returns list of model bone info, format {'file': str(model_file_path), 'bones': List[str]}
            cache_file: Path to the match info JSON file, cache result to avoid calc everytime
            
        Returns:
            Dictionary mapping animation files to their matched models
        """
        import os
        import json
        
        # Try to load from cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                content = json.load(f)
                return content['matches']
        
        anim_bone_info = get_anim_bone_info()
        model_bone_info = get_model_bone_info()
        
        matches = {}
        total_comparisons = len(anim_bone_info) * len(model_bone_info)
        current = 0
        
        # Calculate match rate between each animation and model
        for idx, anim_info in enumerate(anim_bone_info):
            anim_file = anim_info['file']
            anim_bones = set(anim_info['bones'])
            
            if not anim_bones:
                continue
            
            matches[anim_file] = []
            for model_info in model_bone_info:
                current += 1
                if current % 10000 == 0:
                    print(f"Processing {current}/{total_comparisons} comparisons...")
                
                model_file = model_info['file']
                model_bones = set(model_info['bones'])
                
                if not model_bones:
                    continue
                
                # Calculate match rate: intersection / union
                intersection = anim_bones & model_bones
                union = anim_bones | model_bones
                
                if len(union) == 0:
                    continue
                
                match_rate = len(intersection) / len(union)
                
                # Only save matches with rate > 0.5 and matched bones > 30
                if match_rate > 0.5 and len(intersection) > 30:
                    matches[anim_file].append({
                        'model_file': model_file,
                        'match_rate': round(match_rate, 4),
                        'matched_bones': len(intersection),
                        'anim_bones': len(anim_bones),
                        'model_bones': len(model_bones)
                    })
        
            # Sort by match rate (descending), then by model file size (descending)
            def get_sort_key(match):
                model_file = match['model_file']
                file_size = 0
                if os.path.exists(model_file):
                    file_size = os.path.getsize(model_file)
                return (-match['match_rate'], -file_size)
            
            matches[anim_file].sort(key=get_sort_key)
            matches[anim_file] = matches[anim_file][:3]
        
        # Save to file
        result = {
            'total_matches': len(matches),
            'total_anims': len(anim_bone_info),
            'total_models': len(model_bone_info),
            'matches': matches
        }
        
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Match info saved to {cache_file}")
        print(f"Total matches found: {len(matches)}")
        
        return matches
