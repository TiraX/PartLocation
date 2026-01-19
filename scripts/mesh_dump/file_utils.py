import os
import json
from typing import Dict, List

def get_file_cache(target_dir: str, ext_name: str) -> Dict[str, List[str]]:
    """
    Scan directory and cache file paths by filename
    
    Args:
        target_dir: Root directory to scan
        ext_name: File extension to filter (e.g., '.fbx')
        
    Returns:
        Dictionary mapping filename to list of full paths
        Format: {"filename.fbx": ["/path/to/file1.fbx", "/path/to/file2.fbx"]}
    """
    cache_file = os.path.join(target_dir, f"file_cache_{ext_name.strip('.')}.json")
    
    # Check if cache exists and is valid
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            return cache_data['files']
    
    # Build new cache
    print(f"Building file cache for {ext_name} files in: {target_dir}")
    file_cache: Dict[str, List[str]] = {}
    
    if not ext_name.startswith('.'):
        ext_name = '.' + ext_name
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(ext_name.lower()):
                full_path = os.path.join(root, file)
                if file not in file_cache:
                    file_cache[file] = []
                file_cache[file].append(full_path)
    
    # Save cache
    content = {
        'num': len(file_cache),
        'files': file_cache
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    print(f"Cache saved to: {cache_file}")
    
    return file_cache