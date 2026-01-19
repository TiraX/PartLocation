"""
Generate dataset from FBX files.

This script processes FBX files to create training data:
1. Load FBX model
2. Normalize whole model to [-0.5, 0.5] with padding
3. Split model into parts
4. Normalize each part individually
5. Calculate relative transformation parameters
6. Save normalized models and transformation parameters
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add mesh_dump to path
sys.path.insert(0, str(Path(__file__).parent))

from mesh_dump.fbx_utils import FbxUtil
from mesh_dump.model import Model
from mesh_dump.mesh import Mesh
from mesh_dump.parallel_tasks import ParallelTasks, Task


def find_connected_components(faces: np.ndarray, num_vertices: int) -> List[List[int]]:
    """
    Find connected components in a mesh based on shared vertices.
    
    Args:
        faces: Face indices array (Nx3 for triangles)
        num_vertices: Total number of vertices
        
    Returns:
        List of connected components, each component is a list of face indices
    """
    # Build adjacency list for faces
    # Two faces are adjacent if they share at least one vertex
    vertex_to_faces = [[] for _ in range(num_vertices)]
    
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_to_faces[vertex_idx].append(face_idx)
    
    # Find connected components using BFS
    visited = set()
    components = []
    
    for start_face in range(len(faces)):
        if start_face in visited:
            continue
        
        # BFS to find all connected faces
        component = []
        queue = [start_face]
        visited.add(start_face)
        
        while queue:
            current_face = queue.pop(0)
            component.append(current_face)
            
            # Get all vertices of current face
            for vertex_idx in faces[current_face]:
                # Get all faces that share this vertex
                for neighbor_face in vertex_to_faces[vertex_idx]:
                    if neighbor_face not in visited:
                        visited.add(neighbor_face)
                        queue.append(neighbor_face)
        
        components.append(component)
    
    return components


def calculate_component_bounds(positions: np.ndarray, faces: np.ndarray, face_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate bounds for a connected component.
    
    Args:
        positions: Vertex positions array (Nx3)
        faces: Face indices array (Mx3)
        face_indices: List of face indices in this component
        
    Returns:
        Tuple of (bounds_min, bounds_max)
    """
    # Get all vertices used by these faces
    vertex_indices = set()
    for face_idx in face_indices:
        for vertex_idx in faces[face_idx]:
            vertex_indices.add(vertex_idx)
    
    # Get positions of these vertices
    component_positions = positions[list(vertex_indices)]
    
    # Calculate bounds
    bounds_min = np.min(component_positions, axis=0)
    bounds_max = np.max(component_positions, axis=0)
    
    return bounds_min, bounds_max


def bounds_intersect(bounds1: Tuple[np.ndarray, np.ndarray], bounds2: Tuple[np.ndarray, np.ndarray]) -> bool:
    """
    Check if two bounding boxes intersect.
    
    Args:
        bounds1: First bounds (min, max)
        bounds2: Second bounds (min, max)
        
    Returns:
        True if bounds intersect
    """
    min1, max1 = bounds1
    min2, max2 = bounds2
    
    # Check if bounds overlap in all three dimensions
    return (min1[0] <= max2[0] and max1[0] >= min2[0] and
            min1[1] <= max2[1] and max1[1] >= min2[1] and
            min1[2] <= max2[2] and max1[2] >= min2[2])


def group_components_by_bounds(components: List[List[int]], positions: np.ndarray, faces: np.ndarray) -> List[List[List[int]]]:
    """
    Group connected components based on bounds intersection.
    
    Args:
        components: List of connected components (each is a list of face indices)
        positions: Vertex positions array
        faces: Face indices array
        
    Returns:
        List of component groups, each group contains components with intersecting bounds
    """
    # Calculate bounds for each component
    component_bounds = []
    for component in components:
        bounds = calculate_component_bounds(positions, faces, component)
        component_bounds.append(bounds)
    
    # Group components with intersecting bounds
    groups = []
    assigned = [False] * len(components)
    
    for i in range(len(components)):
        if assigned[i]:
            continue
        
        # Start a new group
        group = [components[i]]
        assigned[i] = True
        
        # Find all components that intersect with any component in this group
        changed = True
        while changed:
            changed = False
            for j in range(len(components)):
                if assigned[j]:
                    continue
                
                # Check if component j intersects with any component in the group
                for group_component_idx in range(len(group)):
                    # Find original index of this group component
                    original_idx = components.index(group[group_component_idx])
                    if bounds_intersect(component_bounds[j], component_bounds[original_idx]):
                        group.append(components[j])
                        assigned[j] = True
                        changed = True
                        break
        
        groups.append(group)
    
    return groups


def create_mesh_from_faces(original_mesh: Mesh, face_indices: List[int], name: str) -> Mesh:
    """
    Create a new mesh from selected faces of an original mesh.
    
    Args:
        original_mesh: Original mesh
        face_indices: List of face indices to include
        name: Name for the new mesh
        
    Returns:
        New Mesh instance
    """
    # Get original data
    original_positions = original_mesh.get_vertex_attribute(Mesh.POSITION)
    original_faces = original_mesh.faces
    
    if original_positions is None or original_faces is None:
        return None
    
    # Get all vertices used by selected faces
    vertex_indices = set()
    for face_idx in face_indices:
        for vertex_idx in original_faces[face_idx]:
            vertex_indices.add(vertex_idx)
    
    vertex_indices = sorted(list(vertex_indices))
    
    # Create vertex index mapping (old index -> new index)
    vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
    
    # Create new mesh
    new_mesh = Mesh(name)
    
    # Copy vertex attributes
    for attr_name in original_mesh.get_vertex_attribute_names():
        original_attr = original_mesh.get_vertex_attribute(attr_name)
        if original_attr is not None:
            new_attr = original_attr[vertex_indices]
            new_mesh.set_vertex_attribute(attr_name, new_attr)
    
    # Create new faces with remapped indices
    new_faces = []
    for face_idx in face_indices:
        old_face = original_faces[face_idx]
        new_face = [vertex_mapping[old_idx] for old_idx in old_face]
        new_faces.append(new_face)
    
    new_mesh.faces = np.array(new_faces)
    
    # Copy materials
    for material in original_mesh.materials:
        new_mesh.add_material(material.copy())
    
    # Note: Sections are not copied as they may not be valid for the new mesh
    
    return new_mesh


def analysis_model(model: Model) -> Optional[List[Model]]:
    """
    Analyze a model and split meshes with multiple disconnected component groups.
    
    For each mesh in the model:
    1. Find connected components (faces sharing vertices)
    2. Calculate bounds for each component
    3. Group components with intersecting bounds
    4. If multiple groups exist, split the mesh into sub-meshes
    
    Args:
        model: Model to analyze
        
    Returns:
        List of Models if any mesh was split, None if no splitting needed
    """
    split_occurred = False
    result_models = []
    
    for mesh in model.meshes:
        positions = mesh.get_vertex_attribute(Mesh.POSITION)
        faces = mesh.faces
        
        if positions is None or faces is None or len(faces) == 0:
            # Keep mesh as is
            continue
        
        # Step 1: Find connected components
        components = find_connected_components(faces, len(positions))
        
        if len(components) <= 1:
            # Single component, no need to analyze further
            continue
        
        # Step 2 & 3: Group components by bounds intersection
        groups = group_components_by_bounds(components, positions, faces)
        
        if len(groups) <= 1:
            # All components are in one group, no splitting needed
            continue
        
        # Step 4: Split mesh into multiple meshes
        split_occurred = True
        for group_idx, group in enumerate(groups):
            # Flatten group (list of components) into single list of face indices
            face_indices = []
            for component in group:
                face_indices.extend(component)
            
            # Create new mesh for this group
            sub_mesh_name = f"{mesh.name}_group{group_idx}"
            sub_mesh = create_mesh_from_faces(mesh, face_indices, sub_mesh_name)
            
            if sub_mesh is not None:
                # Create a new model for this sub-mesh
                sub_model = Model(name=sub_mesh_name)
                sub_model.add_mesh(sub_mesh)
                
                # Copy skeleton if exists
                if model.has_skeleton():
                    sub_model.skeleton = model.skeleton.copy()
                
                result_models.append(sub_model)
    
    return result_models if split_occurred else None


class DatasetGenerator:
    """Generate training dataset from FBX files."""
    
    def __init__(self, output_dir: str, padding: float = 0.01):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Output directory for processed data
            padding: Padding ratio for normalization (default: 0.01)
        """
        self.output_dir = Path(output_dir)
        self.padding = padding
        self.min_val = -0.5
        self.max_val = 0.5
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_fbx_file(self, fbx_path: str) -> bool:
        """
        Process a single FBX file.
        
        Args:
            fbx_path: Path to FBX file
            
        Returns:
            True if processing successful, False otherwise
        """
        fbx_path = Path(fbx_path)
        if not fbx_path.exists():
            print(f"Error: FBX file not found: {fbx_path}")
            return False
        
        # Use filename as sample name
        sample_name = fbx_path.stem
        
        # Create sample directory
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load FBX model
            model = FbxUtil.load_model(str(fbx_path), ignore_skeleton=True)
            if model is None:
                print(f"Error: Failed to load FBX file: {fbx_path}")
                return False
            if len(model.meshes) == 1:
                print(f"Warning: Model has only 1 mesh, skipping processing: {fbx_path}")
                return False
            
            # Step 2: Get original bounds before normalization
            original_bounds = model.get_bounds()
            if original_bounds is None:
                print(f"Error: Model has no valid bounds: {fbx_path}")
                return False
            
            original_bounds_min, original_bounds_max = original_bounds
            original_center = (original_bounds_min + original_bounds_max) * 0.5
            original_size = original_bounds_max - original_bounds_min
            
            # Step 3: Normalize whole model
            model.normalize(self.min_val, self.max_val, self.padding)
            
            # Get normalized bounds
            normalized_bounds = model.get_bounds()
            if normalized_bounds is None:
                print(f"Error: Failed to get normalized bounds: {fbx_path}")
                return False
            
            normalized_bounds_min, normalized_bounds_max = normalized_bounds
            normalized_center = (normalized_bounds_min + normalized_bounds_max) * 0.5
            normalized_size = normalized_bounds_max - normalized_bounds_min
            
            # Calculate normalization parameters
            max_dimension = np.max(original_size)
            target_range = self.max_val - self.min_val
            effective_range = target_range * (1.0 - 2.0 * self.padding)
            scale_factor = effective_range / max_dimension if max_dimension > 0 else 1.0
            
            # Save normalized whole model
            whole_output_path = sample_dir / f"{sample_name}-whole.fbx"
            if not FbxUtil.save_model(model, output_path=str(whole_output_path), ignore_skeleton=True):
                print(f"Error: Failed to save whole model: {whole_output_path}")
                return False
            
            # Step 4: Split model into parts and process each part
            part_models = model.split()
            
            parts_data = {}

            # Step 4.5: Find extra splits
            extra_splits = []
            for i, part_model in enumerate(part_models):
                part_model_name = getattr(part_model, "name", f"part_{i}")
                tri_count = 0
                vert_count = 0
                try:
                    for _mesh in getattr(part_model, "meshes", []) or []:
                        _faces = getattr(_mesh, "faces", None)
                        if _faces is not None:
                            tri_count += len(_faces)
                        _pos = _mesh.get_vertex_attribute(Mesh.POSITION)
                        if _pos is not None:
                            vert_count += len(_pos)
                except Exception:
                    pass

                start_time = time.perf_counter()
                split_models = analysis_model(part_model)
                elapsed_sec = time.perf_counter() - start_time

                print(
                    f"analysis_model elapsed={elapsed_sec:.3f}s | part_model={part_model_name} | triangles={tri_count} | vertices={vert_count}"
                )

                if split_models is not None:
                    extra_splits.extend(split_models)
            part_models.extend(extra_splits)
            
            for i, part_model in enumerate(part_models):
                part_name = part_model.name
            
                # Get part bounds in whole model's normalized space
                part_bounds_in_whole = part_model.get_bounds()
                if part_bounds_in_whole is None:
                    print(f"Warning: Part {part_name} has no valid bounds, skipping: {fbx_path}")
                    continue
                
                part_bounds_min_in_whole, part_bounds_max_in_whole = part_bounds_in_whole
                part_center_in_whole = (part_bounds_min_in_whole + part_bounds_max_in_whole) * 0.5
                part_size_in_whole = part_bounds_max_in_whole - part_bounds_min_in_whole
                
                # Normalize part individually
                part_model.normalize(self.min_val, self.max_val, self.padding)
                
                # Get part's own normalization parameters
                part_normalized_bounds = part_model.get_bounds()
                if part_normalized_bounds is None:
                    print(f"Warning: Failed to get normalized bounds for part {part_name}, skipping: {fbx_path}")
                    continue
                
                part_normalized_min, part_normalized_max = part_normalized_bounds
                part_normalized_center = (part_normalized_min + part_normalized_max) * 0.5
                
                # Calculate part's normalization scale factor
                part_max_dimension_in_whole = np.max(part_size_in_whole)
                part_max_dimension_normalized = np.max(part_normalized_max - part_normalized_min)
                part_scale_factor = part_max_dimension_in_whole / part_max_dimension_normalized
                
                # Calculate relative transformation
                # Translation: part center in whole's normalized space
                translation = part_center_in_whole - part_normalized_center
                
                # Rotation: identity (no rotation in this simple case)
                rotation = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
                
                # Scale: ratio of part's scale to whole's scale
                scale = np.array([part_scale_factor, part_scale_factor, part_scale_factor])
                
                # Save normalized part model
                part_output_path = sample_dir / f"{sample_name}-{part_name}.fbx"
                if not FbxUtil.save_model(part_model, output_path=str(part_output_path), ignore_skeleton=True):
                    print(f"Warning: Failed to save part {part_name}: {part_output_path}")
                    continue
                
                # Prepare part data
                part_data = {
                    "model_name": sample_name,
                    "part_name": part_name,
                    "translation": translation.tolist(),
                    "rotation": rotation.tolist(),
                    "scale": scale.tolist()
                }
                
                # Save part JSON
                part_json_path = sample_dir / f"{sample_name}-{part_name}.json"
                with open(part_json_path, 'w', encoding='utf-8') as f:
                    json.dump(part_data, f, indent=2, ensure_ascii=False)
                
                # Store part data for summary
                parts_data[part_name] = part_data
            
            return True
            
        except Exception as e:
            print(f"Error processing {fbx_path}: {e}")
            return False

def generate_dataset_task(input_paths: List[Tuple[str, bool]]) -> List[Task]:
    """
    Generate all tasks for dataset generation.
    
    Args:
        input_paths: List of (input_path, recursive) tuples
        
    Returns:
        List of Task objects
    """
    tasks = []
    
    for input_path, recursive in input_paths:
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            print(f"Warning: Input path does not exist: {input_path}")
            continue
        
        # Find all FBX files
        if recursive:
            fbx_files = list(input_path_obj.rglob("*.fbx"))
        else:
            fbx_files = list(input_path_obj.glob("*.fbx"))
        
        # Create task for each FBX file
        for fbx_file in fbx_files:
            task = Task(
                task_id=str(fbx_file),
                data={
                    'fbx_path': str(fbx_file)
                }
            )
            tasks.append(task)
    
    return tasks


def build_data(task_data: Dict) -> bool:
    """
    Execute data processing for a single FBX file.
    This function is called by worker processes.
    
    Args:
        task_data: Dictionary containing 'fbx_path', 'output_dir', and 'padding'
        
    Returns:
        True if processing successful, False otherwise
    """
    fbx_path = task_data['fbx_path']
    output_dir = task_data['output_dir']
    padding = task_data['padding']
    
    # Create generator for this task
    generator = DatasetGenerator(output_dir, padding=padding)
    
    # Process the FBX file
    return generator.process_fbx_file(fbx_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate training dataset from FBX files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single FBX file
  python generate_dataset.py input.fbx -o output_dir
  
  # Process all FBX files in a directory
  python generate_dataset.py input_dir/ -o output_dir
  
  # Process recursively with custom padding
  python generate_dataset.py input_dir/ -o output_dir -r --padding 0.02
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for FBX files recursively in subdirectories"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.01,
        help="Padding ratio for normalization (default: 0.01)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing with multiple processes"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_dataset.json",
        help="Checkpoint file for resume capability (default: checkpoint_dataset.json)"
    )
    
    args = parser.parse_args()
    
    # Validate padding
    if not 0.0 <= args.padding < 0.5:
        print("Error: Padding must be between 0.0 and 0.5")
        return 1
    
    # Define input paths
    input_paths = [
        ('/Users/shuaizhao/Documents/aaa/snowdrop/ue/Fbx', False)
        # ('d:/Data/ShadowUnit', False), 
        # ('d:/Data/DiabloChar/Processed', True), 
        # ('d:/Data/models_gta5.peds_only/models/peds', True), 
        # ('d:/Data/models_rdr2.peds_only/models/peds', True), 
    ]
    
    if args.parallel:
        # Use parallel processing with ParallelTasks framework
        
        # Create parallel tasks framework
        parallel_tasks = ParallelTasks(
            task_func=build_data,
            checkpoint_file=args.checkpoint,
            num_processes=args.num_processes
        )
        
        # Generate tasks with output_dir and padding included
        def task_generator():
            tasks = generate_dataset_task(input_paths)
            # Add output_dir and padding to each task's data
            for task in tasks:
                task.data['output_dir'] = args.output
                task.data['padding'] = args.padding
            return tasks
        
        parallel_tasks.create_tasks(task_generator)
        
        # Run tasks
        stats = parallel_tasks.run()
        
        print(f"\nProcessing complete: {stats['completed']}/{stats['total_tasks']} succeeded, {stats['failed']} failed, {stats['skipped']} skipped")
        if stats['failed'] > 0:
            print(f"Warning: {stats['failed']} tasks failed")
        
        return 0 if stats['failed'] == 0 else 1
        
    else:
        # Use sequential processing
        # Generate all tasks
        tasks = generate_dataset_task(input_paths)
        
        if not tasks:
            print("Error: No FBX files found to process")
            return 1
        
        # Process tasks sequentially
        success_count = 0
        for i, task in enumerate(tasks, 1):
            print(f"Processing task {i}/{len(tasks)}: {task.task_id}")
            # Prepare task data
            task_data = task.data.copy()
            task_data['output_dir'] = args.output
            task_data['padding'] = args.padding
            
            # Execute task
            if build_data(task_data):
                success_count += 1
        
        print(f"\nProcessing complete: {success_count}/{len(tasks)} succeeded")
        if success_count < len(tasks):
            print(f"Warning: {len(tasks) - success_count} tasks failed")
        
        return 0 if success_count == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
