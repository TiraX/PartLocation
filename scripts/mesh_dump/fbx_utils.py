import fbx
import numpy as np
from typing import Optional, List, Dict
from scipy.spatial.transform import Rotation

from .model import Model
from .mesh import Mesh
from .material import Material
from .skeleton import Skeleton
from .bone import Bone
from .anim import Anim
from .material_param import TextureParam, NumericParam, TextureType


class FbxUtil:
    # FBX rotation order constants
    FBX_EULER_ORDER = 'XYZ'
    FBX_ROTATION_ORDER_ENUM = fbx.EFbxRotationOrder.eEulerXYZ
    """Utility class for exporting Model and Animation data to FBX format"""
    
    def __init__(self):
        """Initialize FBX utility"""
        self.manager = None
        self.scene = None
    
    @staticmethod
    def save_model(model: Model, 
                   anim: Optional[Anim] = None, 
                   output_path: str = "output.fbx",
                   ignore_skeleton: bool = False
                   ) -> bool:
        """
        Save Model and optional Animation to FBX format
        
        Args:
            model: Model instance to export
            anim: Optional Anim instance to export
            output_path: Output FBX file path
            
        Returns:
            True if export successful, False otherwise
        """
        fbx_util = FbxUtil()
        try:
            # Initialize FBX SDK
            fbx_util._initialize_fbx_scene()
            
            # Create materials
            fbx_materials = fbx_util._create_materials(model)
            
            # Create skeleton if model has one
            fbx_bone_nodes = {}
            if model.has_skeleton() and not ignore_skeleton:
                fbx_bone_nodes = fbx_util._create_skeleton(model.skeleton)
            
            # Create meshes
            for mesh in model.meshes:
                fbx_util._create_mesh(mesh, fbx_materials, fbx_bone_nodes)
            
            # Add animation if provided
            if anim is not None and model.has_skeleton() and not ignore_skeleton:
                fbx_util._create_animation(anim, fbx_bone_nodes, model.skeleton)
            
            # Export to file
            success = fbx_util._export_to_file(output_path)
            
            return success
            
        except Exception as e:
            print(f"  Error exporting model to FBX: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            fbx_util._cleanup()
    
    @staticmethod
    def load_model(fbx_path: str,                   
                   ignore_skeleton: bool = False
                   ) -> Optional[Model]:
        """
        Load Model from FBX file
        
        Args:
            fbx_path: Path to FBX file to load
            
        Returns:
            Model instance if successful, None otherwise
        """
        fbx_util = FbxUtil()
        try:
            # Initialize FBX SDK
            fbx_util._initialize_fbx_scene()
            
            # Import FBX file
            if not fbx_util._import_from_file(fbx_path):
                print(f"  Failed to import FBX file: {fbx_path}")
                return None
            
            # Create model
            model = Model()
            
            # Extract model name from file path
            import os
            model.name = os.path.splitext(os.path.basename(fbx_path))[0]
            
            # Parse materials first
            materials_dict = fbx_util._parse_materials()
            
            # Parse skeleton
            skeleton = None
            if not ignore_skeleton:
                skeleton = fbx_util._parse_skeleton()
            if skeleton:
                model.skeleton = skeleton
                print(f"  Found skeleton with {skeleton.get_bone_count()} bones")
            
            # Parse meshes
            fbx_util._parse_meshes(model, materials_dict)
            
            # Parse skinning data for each mesh if skeleton exists
            if skeleton:
                for mesh in model.meshes:
                    # Find the FBX mesh node for this mesh
                    fbx_mesh_node = fbx_util._find_mesh_node_by_name(mesh.name)
                    if fbx_mesh_node:
                        fbx_mesh = fbx_mesh_node.GetMesh()
                        if fbx_mesh:
                            fbx_util._parse_skin_data(fbx_mesh, mesh, skeleton)
            
            return model
            
        except Exception as e:
            print(f"  Error loading model from FBX: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            fbx_util._cleanup()
    
    def _initialize_fbx_scene(self):
        """Initialize FBX manager and scene"""
        self.manager = fbx.FbxManager.Create()
        self.scene = fbx.FbxScene.Create(self.manager, "Scene")
        
        # Set scene unit to meters
        global_settings = self.scene.GetGlobalSettings()
        global_settings.SetSystemUnit(fbx.FbxSystemUnit.m)
    
    def _create_materials(self, model: Model) -> Dict[str, fbx.FbxSurfacePhong]:
        """
        Create FBX materials from model meshes
        
        Args:
            model: Model instance
            
        Returns:
            Dictionary mapping material names to FBX materials
        """
        fbx_materials = {}
        
        for mesh in model.meshes:
            for material in mesh.materials:
                if material.name in fbx_materials:
                    continue
                
                fbx_material = self._create_single_material(material)
                fbx_materials[material.name] = fbx_material
        
        return fbx_materials
    
    def _create_single_material(self, material: Material) -> fbx.FbxSurfacePhong:
        """
        Create a single FBX material from Material instance
        
        Args:
            material: Material instance
            
        Returns:
            FBX material
        """
        fbx_material = fbx.FbxSurfacePhong.Create(self.scene, material.name)
        
        # Set default material properties
        fbx_material.Diffuse.Set(fbx.FbxDouble3(0.8, 0.8, 0.8))
        fbx_material.Specular.Set(fbx.FbxDouble3(0.5, 0.5, 0.5))
        fbx_material.Shininess.Set(0.8)
        fbx_material.ReflectionFactor.Set(0.0)
        
        # Process material parameters
        for param in material.get_all_parameters():
            if isinstance(param, TextureParam):
                self._apply_texture_parameter(fbx_material, param)
        
        return fbx_material
    
    def _apply_numeric_parameter(self, fbx_material: fbx.FbxSurfacePhong, param: NumericParam):
        """
        Apply numeric parameter to FBX material
        
        Args:
            fbx_material: FBX material
            param: NumericParam instance
        """
        pass
    
    def _apply_texture_parameter(self, fbx_material: fbx.FbxSurfacePhong, param: TextureParam):
        """
        Apply texture parameter to FBX material
        
        Args:
            fbx_material: FBX material
            param: TextureParam instance
        """
        if not param.texture_path or param.texture_type == TextureType.UNKNOWN:
            return
        
        # Create FBX texture
        fbx_texture = fbx.FbxFileTexture.Create(self.scene, param.name)
        fbx_texture.SetFileName(param.texture_path)
        fbx_texture.SetTextureUse(fbx.FbxTexture.ETextureUse.eStandard)
        fbx_texture.SetMappingType(fbx.FbxTexture.EMappingType.eUV)
        
        # Connect texture to material based on texture type
        texture_type = param.texture_type
        
        if texture_type == TextureType.DIFFUSE:
            fbx_material.Diffuse.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.NORMAL:
            fbx_material.NormalMap.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.ROUGHNESS:
            fbx_material.Shininess.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.METALLIC:
            fbx_material.ReflectionFactor.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.SPECULAR:
            fbx_material.Specular.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.EMISSIVE:
            fbx_material.Emissive.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.AMBIENT_OCCLUSION:
            fbx_material.AmbientFactor.ConnectSrcObject(fbx_texture)
        elif texture_type == TextureType.OPACITY:
            fbx_material.TransparencyFactor.ConnectSrcObject(fbx_texture)
    
    def _create_skeleton(self, skeleton: Skeleton) -> Dict[int, fbx.FbxNode]:
        """
        Create FBX skeleton from Skeleton instance
        
        Args:
            skeleton: Skeleton instance
            
        Returns:
            Dictionary mapping bone indices to FBX bone nodes
        """
        fbx_bone_nodes = {}
        
        # Create all bone nodes first
        for bone in skeleton.bones:
            bone_node = self._create_bone_node(bone)
            fbx_bone_nodes[bone.index] = bone_node
        
        # Set up hierarchy
        for bone in skeleton.bones:
            bone_node = fbx_bone_nodes[bone.index]
            
            if bone.has_parent() and bone.parent_index in fbx_bone_nodes:
                parent_node = fbx_bone_nodes[bone.parent_index]
                parent_node.AddChild(bone_node)
            else:
                # Root bone - add to scene root
                self.scene.GetRootNode().AddChild(bone_node)
        
        return fbx_bone_nodes
    
    def _create_bone_node(self, bone: Bone) -> fbx.FbxNode:
        """
        Create a single FBX bone node from Bone instance
        
        Args:
            bone: Bone instance
            
        Returns:
            FBX bone node
        """
        bone_node = fbx.FbxNode.Create(self.scene, bone.name)
        bone_skeleton = fbx.FbxSkeleton.Create(self.scene, f"{bone.name}_Skeleton")
        
        # Set skeleton type (root or limb)
        if not bone.has_parent():
            bone_skeleton.SetSkeletonType(fbx.FbxSkeleton.EType.eRoot)
        else:
            bone_skeleton.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)
        
        bone_node.SetRotationOrder(fbx.FbxNode.EPivotSet.eSourcePivot, FbxUtil.FBX_ROTATION_ORDER_ENUM)
        bone_node.SetNodeAttribute(bone_skeleton)
        
        # Set bone transform
        self._set_bone_transform(bone_node, bone)
        
        return bone_node
    
    def _set_bone_transform(self, bone_node: fbx.FbxNode, bone: Bone):
        """
        Set bone node transform from Bone instance
        
        Args:
            bone_node: FBX bone node
            bone: Bone instance
        """
        # Set translation
        pos = bone.position
        bone_node.LclTranslation.Set(fbx.FbxDouble3(float(pos[0]), float(pos[1]), float(pos[2])))
        
        # Set rotation (quaternion to euler)
        rot = bone.rotation
        # Convert quaternion to Euler angles using scipy (consistent with loading)
        euler_rad = Rotation.from_quat(rot).as_euler(FbxUtil.FBX_EULER_ORDER)
        euler = np.degrees(euler_rad)
        bone_node.LclRotation.Set(fbx.FbxDouble3(euler[0], euler[1], euler[2]))
        
        # Set scale
        scale = bone.scale
        bone_node.LclScaling.Set(fbx.FbxDouble3(float(scale[0]), float(scale[1]), float(scale[2])))
    
    def _create_mesh(self, mesh: Mesh, fbx_materials: Dict[str, fbx.FbxSurfacePhong], 
                     fbx_bone_nodes: Dict[int, fbx.FbxNode]):
        """
        Create FBX mesh from Mesh instance
        
        Args:
            mesh: Mesh instance
            fbx_materials: Dictionary of FBX materials
            fbx_bone_nodes: Dictionary of FBX bone nodes
        """
        # Create mesh node
        mesh_node = fbx.FbxNode.Create(self.scene, mesh.name)
        fbx_mesh = fbx.FbxMesh.Create(self.scene, f"{mesh.name}_Mesh")
        mesh_node.SetNodeAttribute(fbx_mesh)
        self.scene.GetRootNode().AddChild(mesh_node)
        
        # Add materials to mesh node
        for material in mesh.materials:
            if material.name in fbx_materials:
                mesh_node.AddMaterial(fbx_materials[material.name])
        
        # Set geometry data
        self._set_mesh_geometry(fbx_mesh, mesh)
        
        # Add skinning if available
        if mesh.has_skinning_data() and fbx_bone_nodes:
            self._add_skinning(fbx_mesh, mesh_node, mesh, fbx_bone_nodes)
    
    def _set_mesh_geometry(self, fbx_mesh: fbx.FbxMesh, mesh: Mesh):
        """
        Set mesh geometry data (positions, faces, normals, UVs)
        
        Args:
            fbx_mesh: FBX mesh
            mesh: Mesh instance
        """
        positions = mesh.get_vertex_attribute(Mesh.POSITION)
        faces = mesh.faces
        normals = mesh.get_vertex_attribute(Mesh.NORMAL)
        uv0 = mesh.get_vertex_attribute(Mesh.UV0)
        uv1 = mesh.get_vertex_attribute(Mesh.UV1)
        uv2 = mesh.get_vertex_attribute(Mesh.UV2)
        uv3 = mesh.get_vertex_attribute(Mesh.UV3)
        
        if positions is None or faces is None:
            print(f"  Warning: Mesh {mesh.name} has no vertex or face data")
            return
        
        # Initialize control points (positions)
        num_verts = len(positions)
        fbx_mesh.InitControlPoints(num_verts)
        
        # Create LayerElementMaterial for per-polygon material assignment
        layer_element_material = fbx_mesh.CreateElementMaterial()
        layer_element_material.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygon)
        layer_element_material.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        
        # Set vertex positions
        for i, pos in enumerate(positions):
            point = fbx.FbxVector4(float(pos[0]), float(pos[1]), float(pos[2]))
            fbx_mesh.SetControlPointAt(point, i)
        
        # Create layer 0
        if fbx_mesh.GetLayerCount() < 1:
            fbx_mesh.CreateLayer()
        layer = fbx_mesh.GetLayer(0)
        
        # Set normals if available
        if normals is not None:
            self._set_mesh_normals(fbx_mesh, layer, normals)
        
        # Set UVs (up to 4 layers)
        uv_layers = [uv0, uv1, uv2, uv3]
        for uv_layer_index, uvs in enumerate(uv_layers):
            if uvs is None:
                continue

            while fbx_mesh.GetLayerCount() <= uv_layer_index:
                fbx_mesh.CreateLayer()
            uv_layer = fbx_mesh.GetLayer(uv_layer_index)

            uv_set_name = f"UV{uv_layer_index}"
            self._set_mesh_uvs(fbx_mesh, uv_layer, uvs, uv_set_name=uv_set_name)
        
        # Build face to material index mapping from sections
        face_to_material = {}
        sections = mesh.get_sections()
        
        if sections:
            for idx, section in enumerate(sections):
                start_face = section['start_face']
                face_count = section['face_count']
                material_index = section['material_index']
                for face_idx in range(start_face, start_face + face_count):
                    face_to_material[face_idx] = material_index
                
        # Add faces with material indices
        for face_idx, face in enumerate(faces):
            # Get material index for this face, default to -1 if not specified
            material_index = face_to_material.get(face_idx, -1)
            fbx_mesh.BeginPolygon(material_index)
            for vertex_index in face:
                fbx_mesh.AddPolygon(int(vertex_index))
            fbx_mesh.EndPolygon()
    
    def _set_mesh_normals(self, fbx_mesh: fbx.FbxMesh, layer: fbx.FbxLayer, normals: np.ndarray):
        """
        Set mesh normals
        
        Args:
            fbx_mesh: FBX mesh
            layer: FBX layer
            normals: Normals array
        """
        normal_layer = fbx.FbxLayerElementNormal.Create(fbx_mesh, "Normals")
        normal_layer.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByControlPoint)
        normal_layer.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
        
        normal_array = normal_layer.GetDirectArray()
        for normal in normals:
            # Normalize the normal vector
            nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
            length = (nx*nx + ny*ny + nz*nz) ** 0.5
            if length > 0:
                nx, ny, nz = nx/length, ny/length, nz/length
            normal_array.Add(fbx.FbxVector4(nx, ny, nz))
        
        layer.SetNormals(normal_layer)
    
    def _set_mesh_uvs(self, fbx_mesh: fbx.FbxMesh, layer: fbx.FbxLayer, uvs: np.ndarray, uv_set_name: str = "UVMap"):
        """
        Set mesh UV coordinates
        
        Args:
            fbx_mesh: FBX mesh
            layer: FBX layer
            uvs: UV coordinates array
            uv_set_name: UV set name (e.g., "UVMap", "UV1")
        """
        uv_layer = fbx.FbxLayerElementUV.Create(fbx_mesh, uv_set_name)
        uv_layer.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByControlPoint)
        uv_layer.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eDirect)
        
        uv_array = uv_layer.GetDirectArray()
        for uv in uvs:
            # Flip V coordinate (some software uses bottom-left origin, FBX uses top-left)
            uv_array.Add(fbx.FbxVector2(float(uv[0]), 1.0 - float(uv[1])))
        
        layer.SetUVs(uv_layer)
    
    def _add_skinning(self, fbx_mesh: fbx.FbxMesh, mesh_node: fbx.FbxNode, 
                      mesh: Mesh, fbx_bone_nodes: Dict[int, fbx.FbxNode]):
        """
        Add skinning data to mesh
        
        Args:
            fbx_mesh: FBX mesh
            mesh_node: FBX mesh node
            mesh: Mesh instance
            fbx_bone_nodes: Dictionary of FBX bone nodes
        """
        bone_weights = mesh.get_vertex_attribute(Mesh.BLEND_WEIGHTS)
        bone_indices = mesh.get_vertex_attribute(Mesh.BLEND_INDICES)
        
        if bone_weights is None or bone_indices is None:
            return
        
        # Create skin deformer
        skin = fbx.FbxSkin.Create(self.scene, f"{mesh.name}_Skin")
        
        # Create clusters for each bone
        clusters = {}
        num_verts = len(bone_weights)
        
        # Assign vertex weights
        for vert_idx in range(num_verts):
            vert_bone_indices = bone_indices[vert_idx]
            vert_bone_weights = bone_weights[vert_idx]
            
            # Process each bone influence (typically up to 4)
            for i in range(len(vert_bone_indices)):
                bone_idx = int(vert_bone_indices[i])
                weight = float(vert_bone_weights[i])
                
                # Skip zero weights or invalid bone indices
                if weight <= 0.0 or bone_idx not in fbx_bone_nodes:
                    continue
                
                # Create cluster for this bone if it doesn't exist
                if bone_idx not in clusters:
                    bone_node = fbx_bone_nodes[bone_idx]
                    cluster = fbx.FbxCluster.Create(self.scene, f"Cluster_{bone_idx}")
                    cluster.SetLink(bone_node)
                    cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eNormalize)
                    
                    # Set transform matrices
                    mesh_transform = mesh_node.EvaluateGlobalTransform()
                    cluster.SetTransformMatrix(mesh_transform)
                    
                    bone_transform = bone_node.EvaluateGlobalTransform()
                    cluster.SetTransformLinkMatrix(bone_transform)
                    
                    clusters[bone_idx] = cluster
                
                # Add vertex weight to cluster
                clusters[bone_idx].AddControlPointIndex(vert_idx, weight)
        
        # Add all clusters to skin
        for cluster in clusters.values():
            skin.AddCluster(cluster)
        
        # Add skin to mesh
        if skin.GetClusterCount() > 0:
            fbx_mesh.AddDeformer(skin)
            
            # Create bind pose
            self._create_bind_pose(mesh_node, clusters, fbx_bone_nodes, mesh.name)
    
    def _create_bind_pose(self, mesh_node: fbx.FbxNode, clusters: Dict[int, fbx.FbxCluster],
                          fbx_bone_nodes: Dict[int, fbx.FbxNode], mesh_name: str):
        """
        Create bind pose for skinned mesh
        
        Args:
            mesh_node: FBX mesh node
            clusters: Dictionary of bone clusters
            fbx_bone_nodes: Dictionary of FBX bone nodes
            mesh_name: Mesh name
        """
        bind_pose = fbx.FbxPose.Create(self.scene, f"{mesh_name}_BindPose")
        bind_pose.SetIsBindPose(True)
        
        # Add mesh node to bind pose
        mesh_transform = mesh_node.EvaluateGlobalTransform()
        bind_pose.Add(mesh_node, fbx.FbxMatrix(mesh_transform))
        
        # Add all bones used by this skin to bind pose
        for bone_idx in clusters.keys():
            if bone_idx in fbx_bone_nodes:
                bone_node = fbx_bone_nodes[bone_idx]
                bone_transform = bone_node.EvaluateGlobalTransform()
                bind_pose.Add(bone_node, fbx.FbxMatrix(bone_transform))
        
        self.scene.AddPose(bind_pose)
    
    def _create_animation(self, anim: Anim, fbx_bone_nodes: Dict[int, fbx.FbxNode], 
                          skeleton: Skeleton):
        """
        Create FBX animation from Anim instance
        
        Args:
            anim: Anim instance
            fbx_bone_nodes: Dictionary of FBX bone nodes
            skeleton: Skeleton instance
        """
        # Create animation stack
        anim_stack = fbx.FbxAnimStack.Create(self.scene, anim.name)
        anim_layer = fbx.FbxAnimLayer.Create(self.scene, f"{anim.name}_Layer")
        anim_stack.AddMember(anim_layer)
        
        # Set time mode to frames per second
        fbx.FbxTime.SetGlobalTimeMode(fbx.FbxTime.eFrames30)
        
        # Process each target (bone)
        for target_name in anim.get_target_names():
            bone = skeleton.get_bone_by_name(target_name)
            if bone is None or bone.index not in fbx_bone_nodes:
                continue
            
            bone_node = fbx_bone_nodes[bone.index]
            tracks = anim.get_tracks_for_target(target_name)
            
            # Create animation curves for each property
            for track in tracks:
                self._create_animation_curve(bone_node, track, anim_layer)
    
    def _create_animation_curve(self, bone_node: fbx.FbxNode, track, anim_layer: fbx.FbxAnimLayer):
        """
        Create animation curve for a track
        
        Args:
            bone_node: FBX bone node
            track: Track instance
            anim_layer: FBX animation layer
        """
        property_path = track.property_path.lower()
        keyframes = track.get_keyframes()
        
        if not keyframes:
            return
        
        # Determine which property to animate
        if 'position' in property_path or 'translation' in property_path:
            self._create_translation_curve(bone_node, keyframes, anim_layer)
        elif 'rotation' in property_path:
            self._create_rotation_curve(bone_node, keyframes, anim_layer)
        elif 'scale' in property_path:
            self._create_scale_curve(bone_node, keyframes, anim_layer)
    
    def _create_translation_curve(self, bone_node: fbx.FbxNode, keyframes, anim_layer: fbx.FbxAnimLayer):
        """Create translation animation curves"""
        curve_x = bone_node.LclTranslation.GetCurve(anim_layer, "X", True)
        curve_y = bone_node.LclTranslation.GetCurve(anim_layer, "Y", True)
        curve_z = bone_node.LclTranslation.GetCurve(anim_layer, "Z", True)
        
        for keyframe in keyframes:
            time = fbx.FbxTime()
            time.SetSecondDouble(keyframe.time)
            
            value = keyframe.value
            if isinstance(value, (list, np.ndarray)) and len(value) >= 3:
                curve_x.KeyModifyBegin()
                key_index = curve_x.KeyAdd(time)[0]
                curve_x.KeySetValue(key_index, float(value[0]))
                curve_x.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_x.KeyModifyEnd()
                
                curve_y.KeyModifyBegin()
                key_index = curve_y.KeyAdd(time)[0]
                curve_y.KeySetValue(key_index, float(value[1]))
                curve_y.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_y.KeyModifyEnd()
                
                curve_z.KeyModifyBegin()
                key_index = curve_z.KeyAdd(time)[0]
                curve_z.KeySetValue(key_index, float(value[2]))
                curve_z.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_z.KeyModifyEnd()
    
    def _create_rotation_curve(self, bone_node: fbx.FbxNode, keyframes, anim_layer: fbx.FbxAnimLayer):
        """Create rotation animation curves"""
        curve_x = bone_node.LclRotation.GetCurve(anim_layer, "X", True)
        curve_y = bone_node.LclRotation.GetCurve(anim_layer, "Y", True)
        curve_z = bone_node.LclRotation.GetCurve(anim_layer, "Z", True)
        
        for keyframe in keyframes:
            time = fbx.FbxTime()
            time.SetSecondDouble(keyframe.time)
            
            value = keyframe.value
            # Assume value is quaternion (x, y, z, w)
            if isinstance(value, (list, np.ndarray)) and len(value) >= 4:
                # Convert quaternion to euler angles
                fbx_quat = fbx.FbxQuaternion(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
                euler = fbx_quat.DecomposeSphericalXYZ()
                
                # Convert radians to degrees
                euler_deg = [euler[0] * 57.2957795131, euler[1] * 57.2957795131, euler[2] * 57.2957795131]
                
                curve_x.KeyModifyBegin()
                key_index = curve_x.KeyAdd(time)[0]
                curve_x.KeySetValue(key_index, euler_deg[0])
                curve_x.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_x.KeyModifyEnd()
                
                curve_y.KeyModifyBegin()
                key_index = curve_y.KeyAdd(time)[0]
                curve_y.KeySetValue(key_index, euler_deg[1])
                curve_y.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_y.KeyModifyEnd()
                
                curve_z.KeyModifyBegin()
                key_index = curve_z.KeyAdd(time)[0]
                curve_z.KeySetValue(key_index, euler_deg[2])
                curve_z.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_z.KeyModifyEnd()
    
    def _create_scale_curve(self, bone_node: fbx.FbxNode, keyframes, anim_layer: fbx.FbxAnimLayer):
        """Create scale animation curves"""
        curve_x = bone_node.LclScaling.GetCurve(anim_layer, "X", True)
        curve_y = bone_node.LclScaling.GetCurve(anim_layer, "Y", True)
        curve_z = bone_node.LclScaling.GetCurve(anim_layer, "Z", True)
        
        for keyframe in keyframes:
            time = fbx.FbxTime()
            time.SetSecondDouble(keyframe.time)
            
            value = keyframe.value
            if isinstance(value, (list, np.ndarray)) and len(value) >= 3:
                curve_x.KeyModifyBegin()
                key_index = curve_x.KeyAdd(time)[0]
                curve_x.KeySetValue(key_index, float(value[0]))
                curve_x.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_x.KeyModifyEnd()
                
                curve_y.KeyModifyBegin()
                key_index = curve_y.KeyAdd(time)[0]
                curve_y.KeySetValue(key_index, float(value[1]))
                curve_y.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_y.KeyModifyEnd()
                
                curve_z.KeyModifyBegin()
                key_index = curve_z.KeyAdd(time)[0]
                curve_z.KeySetValue(key_index, float(value[2]))
                curve_z.KeySetInterpolation(key_index, fbx.FbxAnimCurveDef.eInterpolationLinear)
                curve_z.KeyModifyEnd()
    
    def _export_to_file(self, output_path: str) -> bool:
        """
        Export scene to FBX file
        
        Args:
            output_path: Output file path
            
        Returns:
            True if export successful, False otherwise
        """
        exporter = fbx.FbxExporter.Create(self.scene, "Exporter")
        
        if not output_path.endswith('.fbx'):
            output_path += '.fbx'
        
        if exporter.Initialize(output_path):
            exporter.Export(self.scene)
            exporter.Destroy()
            print(f"Successfully exported to FBX: {output_path}")
            return True
        else:
            print(f"Failed to initialize FBX exporter for: {output_path}")
            return False
    
    def _cleanup(self):
        """Clean up FBX manager"""
        if self.manager is not None:
            self.manager.Destroy()
            self.manager = None
            self.scene = None
    
    def _import_from_file(self, fbx_path: str) -> bool:
        """
        Import FBX file into scene
        
        Args:
            fbx_path: Path to FBX file
            
        Returns:
            True if import successful, False otherwise
        """
        importer = fbx.FbxImporter.Create(self.manager, "Importer")
        
        if not importer.Initialize(fbx_path):
            print(f"Failed to initialize FBX importer for: {fbx_path}")
            importer.Destroy()
            return False
        
        if not importer.Import(self.scene):
            print(f"Failed to import FBX scene from: {fbx_path}")
            importer.Destroy()
            return False
        
        importer.Destroy()
        return True
    
    def _parse_materials(self) -> Dict[str, Material]:
        """
        Parse materials from FBX scene
        
        Returns:
            Dictionary mapping material names to Material instances
        """
        materials_dict = {}
        
        # Iterate through all nodes to find materials
        root_node = self.scene.GetRootNode()
        self._parse_materials_recursive(root_node, materials_dict)
        
        return materials_dict
    
    def _parse_materials_recursive(self, node: fbx.FbxNode, materials_dict: Dict[str, Material]):
        """
        Recursively parse materials from node hierarchy
        
        Args:
            node: FBX node
            materials_dict: Dictionary to store parsed materials
        """
        # Get materials from this node
        material_count = node.GetMaterialCount()
        for i in range(material_count):
            fbx_material = node.GetMaterial(i)
            material_name = fbx_material.GetName()
            
            if material_name not in materials_dict:
                material = self._parse_single_material(fbx_material)
                materials_dict[material_name] = material
        
        # Recurse to children
        for i in range(node.GetChildCount()):
            child = node.GetChild(i)
            self._parse_materials_recursive(child, materials_dict)
    
    def _parse_single_material(self, fbx_material) -> Material:
        """
        Parse a single FBX material
        
        Args:
            fbx_material: FBX material
            
        Returns:
            Material instance
        """
        material = Material(fbx_material.GetName())
        
        # Parse texture parameters from FBX material properties
        self._parse_material_textures(fbx_material.Diffuse, material, TextureType.DIFFUSE)
        self._parse_material_textures(fbx_material.NormalMap, material, TextureType.NORMAL)
        self._parse_material_textures(fbx_material.Shininess, material, TextureType.ROUGHNESS)
        self._parse_material_textures(fbx_material.ReflectionFactor, material, TextureType.METALLIC)
        self._parse_material_textures(fbx_material.Specular, material, TextureType.SPECULAR)
        self._parse_material_textures(fbx_material.Emissive, material, TextureType.EMISSIVE)
        self._parse_material_textures(fbx_material.AmbientFactor, material, TextureType.AMBIENT_OCCLUSION)
        self._parse_material_textures(fbx_material.TransparencyFactor, material, TextureType.OPACITY)
        
        return material
    
    def _parse_material_textures(self, fbx_property, material: Material, texture_type: TextureType):
        """
        Parse textures from FBX material property
        
        Args:
            fbx_property: FBX property that may have textures
            material: Material instance to add textures to
            texture_type: Type of texture
        """
        texture_count = fbx_property.GetSrcObjectCount()
        
        for i in range(texture_count):
            texture = fbx_property.GetSrcObject(i)
            
            if texture and texture.GetClassId().Is(fbx.FbxFileTexture.ClassId):
                # Directly use texture object instead of Cast
                texture_path = texture.GetFileName()
                
                if texture_path:
                    texture_name = f"{texture_type.name.lower()}_texture"
                    texture_param = TextureParam(texture_name, texture_path, texture_type)
                    material.add_parameter(texture_param)
    
    def _parse_meshes(self, model: Model, materials_dict: Dict[str, Material]):
        """
        Parse meshes from FBX scene
        
        Args:
            model: Model instance to add meshes to
            materials_dict: Dictionary of parsed materials
        """
        root_node = self.scene.GetRootNode()
        self._parse_meshes_recursive(root_node, model, materials_dict)
    
    def _parse_meshes_recursive(self, node: fbx.FbxNode, model: Model, materials_dict: Dict[str, Material]):
        """
        Recursively parse meshes from node hierarchy
        
        Args:
            node: FBX node
            model: Model instance to add meshes to
            materials_dict: Dictionary of parsed materials
        """
        # Check if this node has a mesh attribute
        node_attribute = node.GetNodeAttribute()
        
        if node_attribute and node_attribute.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            fbx_mesh = node.GetMesh()
            if fbx_mesh:
                mesh = self._parse_single_mesh(node, fbx_mesh, materials_dict)
                if mesh:
                    model.add_mesh(mesh)
        
        # Recurse to children
        for i in range(node.GetChildCount()):
            child = node.GetChild(i)
            self._parse_meshes_recursive(child, model, materials_dict)
    
    def _parse_single_mesh(self, node: fbx.FbxNode, fbx_mesh: fbx.FbxMesh, 
                           materials_dict: Dict[str, Material]) -> Optional[Mesh]:
        """
        Parse a single FBX mesh
        
        Args:
            node: FBX node containing the mesh
            fbx_mesh: FBX mesh
            materials_dict: Dictionary of parsed materials
            
        Returns:
            Mesh instance or None if parsing failed
        """
        mesh_name = node.GetName()
        mesh = Mesh(mesh_name)
        
        # Parse positions
        positions = self._parse_positions(fbx_mesh)
        if positions is None or len(positions) == 0:
            print(f"  Warning: Mesh {mesh_name} has no positions")
            return None
        
        mesh.set_vertex_attribute(Mesh.POSITION, positions)
        
        # Parse faces
        faces = self._parse_faces(fbx_mesh)
        if faces is None or len(faces) == 0:
            print(f"  Warning: Mesh {mesh_name} has no faces")
            return None
        
        mesh.faces = faces
        
        # Parse normals
        normals = self._parse_normals(fbx_mesh)
        if normals is not None:
            mesh.set_vertex_attribute(Mesh.NORMAL, normals)
        
        # Parse UVs (up to 4 layers)
        UV_ATTR_NAMES = (Mesh.UV0, Mesh.UV1, Mesh.UV2, Mesh.UV3)
        for uv_layer_index in range(Mesh.MAX_UV_LAYERS):
            uvs = self._parse_uvs_from_layer(fbx_mesh, uv_layer_index)
            if uvs is None:
                continue

            mesh.set_vertex_attribute(UV_ATTR_NAMES[uv_layer_index], uvs)
        
        # Parse materials and create mesh sections
        self._parse_mesh_materials(node, fbx_mesh, mesh, materials_dict)
        
        return mesh
    
    def _parse_positions(self, fbx_mesh: fbx.FbxMesh) -> Optional[np.ndarray]:
        """
        Parse positions from FBX mesh
        
        Args:
            fbx_mesh: FBX mesh
            
        Returns:
            Numpy array of positions or None
        """
        control_points = fbx_mesh.GetControlPoints()
        vertex_count = fbx_mesh.GetControlPointsCount()
        
        if vertex_count == 0:
            return None
        
        positions = np.zeros((vertex_count, 3), dtype=np.float32)
        
        for i in range(vertex_count):
            positions[i] = [control_points[i][0], control_points[i][1], control_points[i][2]]
        
        return positions
    
    def _parse_faces(self, fbx_mesh: fbx.FbxMesh) -> Optional[np.ndarray]:
        """
        Parse faces from FBX mesh
        
        Args:
            fbx_mesh: FBX mesh
            
        Returns:
            Numpy array of faces or None
        """
        polygon_count = fbx_mesh.GetPolygonCount()
        
        if polygon_count == 0:
            return None
        
        faces = []
        
        for i in range(polygon_count):
            polygon_size = fbx_mesh.GetPolygonSize(i)
            
            if polygon_size == 3:
                # Triangle
                face = [
                    fbx_mesh.GetPolygonVertex(i, 0),
                    fbx_mesh.GetPolygonVertex(i, 1),
                    fbx_mesh.GetPolygonVertex(i, 2)
                ]
                faces.append(face)
            elif polygon_size == 4:
                # Quad - triangulate
                v0 = fbx_mesh.GetPolygonVertex(i, 0)
                v1 = fbx_mesh.GetPolygonVertex(i, 1)
                v2 = fbx_mesh.GetPolygonVertex(i, 2)
                v3 = fbx_mesh.GetPolygonVertex(i, 3)
                
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
            else:
                # N-gon - simple fan triangulation
                v0 = fbx_mesh.GetPolygonVertex(i, 0)
                for j in range(1, polygon_size - 1):
                    v1 = fbx_mesh.GetPolygonVertex(i, j)
                    v2 = fbx_mesh.GetPolygonVertex(i, j + 1)
                    faces.append([v0, v1, v2])
        
        return np.array(faces, dtype=np.int32)
    
    def _parse_normals(self, fbx_mesh: fbx.FbxMesh) -> Optional[np.ndarray]:
        """
        Parse normals from FBX mesh
        
        Args:
            fbx_mesh: FBX mesh
            
        Returns:
            Numpy array of normals or None
        """
        element_normal = fbx_mesh.GetElementNormal()
        
        if not element_normal:
            return None
        
        vertex_count = fbx_mesh.GetControlPointsCount()
        normals = np.zeros((vertex_count, 3), dtype=np.float32)
        
        mapping_mode = element_normal.GetMappingMode()
        reference_mode = element_normal.GetReferenceMode()
        
        if mapping_mode == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
            if reference_mode == fbx.FbxLayerElement.EReferenceMode.eDirect:
                for i in range(vertex_count):
                    normal = element_normal.GetDirectArray().GetAt(i)
                    normals[i] = [normal[0], normal[1], normal[2]]
            elif reference_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                for i in range(vertex_count):
                    index = element_normal.GetIndexArray().GetAt(i)
                    normal = element_normal.GetDirectArray().GetAt(index)
                    normals[i] = [normal[0], normal[1], normal[2]]
        elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
            # Average normals per vertex
            normal_counts = np.zeros(vertex_count, dtype=np.int32)
            
            vertex_id = 0
            for i in range(fbx_mesh.GetPolygonCount()):
                polygon_size = fbx_mesh.GetPolygonSize(i)
                for j in range(polygon_size):
                    control_point_index = fbx_mesh.GetPolygonVertex(i, j)
                    
                    if reference_mode == fbx.FbxLayerElement.EReferenceMode.eDirect:
                        normal = element_normal.GetDirectArray().GetAt(vertex_id)
                    else:
                        index = element_normal.GetIndexArray().GetAt(vertex_id)
                        normal = element_normal.GetDirectArray().GetAt(index)
                    
                    normals[control_point_index] += [normal[0], normal[1], normal[2]]
                    normal_counts[control_point_index] += 1
                    vertex_id += 1
            
            # Average the normals
            for i in range(vertex_count):
                if normal_counts[i] > 0:
                    normals[i] /= normal_counts[i]
        
        return normals

    def _parse_uvs_from_layer(self, fbx_mesh: fbx.FbxMesh, layer_index: int) -> Optional[np.ndarray]:
        """Parse UV coordinates from FBX mesh for a given layer index.

        Notes:
            - FBX chooses UV sets via mesh layers, not per-texture parameters.
            - This function reads the UVs bound on `FbxLayer[layer_index]`.
            - We store UVs per control point, matching the exporter (`eByControlPoint`).

        Args:
            fbx_mesh: FBX mesh
            layer_index: UV layer index (0..3)

        Returns:
            Numpy array of UVs (vertex_count, 2) or None
        """
        if layer_index < 0 or layer_index >= fbx_mesh.GetLayerCount():
            return None

        layer = fbx_mesh.GetLayer(layer_index)
        if layer is None:
            return None

        element_uv = layer.GetUVs()
        if not element_uv:
            return None

        vertex_count = fbx_mesh.GetControlPointsCount()
        if vertex_count <= 0:
            return None

        uvs = np.zeros((vertex_count, 2), dtype=np.float32)

        mapping_mode = element_uv.GetMappingMode()
        reference_mode = element_uv.GetReferenceMode()

        if mapping_mode == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
            if reference_mode == fbx.FbxLayerElement.EReferenceMode.eDirect:
                direct = element_uv.GetDirectArray()
                direct_count = direct.GetCount()
                count = min(vertex_count, direct_count)
                for i in range(count):
                    uv = direct.GetAt(i)
                    uvs[i] = [uv[0], 1.0 - uv[1]]
            elif reference_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                direct = element_uv.GetDirectArray()
                indices = element_uv.GetIndexArray()
                index_count = indices.GetCount()
                count = min(vertex_count, index_count)
                for i in range(count):
                    index = indices.GetAt(i)
                    uv = direct.GetAt(index)
                    uvs[i] = [uv[0], 1.0 - uv[1]]
            else:
                return None

        elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
            # Reduce polygon-vertex UVs to control-point UVs by averaging.
            direct = element_uv.GetDirectArray()
            indices = element_uv.GetIndexArray()

            uv_counts = np.zeros(vertex_count, dtype=np.int32)
            polygon_vertex_id = 0
            for poly_index in range(fbx_mesh.GetPolygonCount()):
                poly_size = fbx_mesh.GetPolygonSize(poly_index)
                for vert_in_poly in range(poly_size):
                    cp_index = fbx_mesh.GetPolygonVertex(poly_index, vert_in_poly)

                    if reference_mode == fbx.FbxLayerElement.EReferenceMode.eDirect:
                        uv = direct.GetAt(polygon_vertex_id)
                    elif reference_mode == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                        uv_index = indices.GetAt(polygon_vertex_id)
                        uv = direct.GetAt(uv_index)
                    else:
                        return None

                    uvs[cp_index] += [uv[0], 1.0 - uv[1]]
                    uv_counts[cp_index] += 1
                    polygon_vertex_id += 1

            for i in range(vertex_count):
                if uv_counts[i] > 0:
                    uvs[i] /= float(uv_counts[i])

        else:
            return None

        return uvs
    
    def _parse_mesh_materials(self, node: fbx.FbxNode, fbx_mesh: fbx.FbxMesh, 
                              mesh: Mesh, materials_dict: Dict[str, Material]):
        """
        Parse materials and create mesh sections
        
        Args:
            node: FBX node
            fbx_mesh: FBX mesh
            mesh: Mesh instance
            materials_dict: Dictionary of parsed materials
        """
        material_count = node.GetMaterialCount()
        
        if material_count == 0:
            # No materials - create a default section with all faces
            mesh.add_section(0, mesh.get_face_count())
            return
        
        # Get material element to determine face-to-material mapping
        element_material = fbx_mesh.GetElementMaterial()
        
        if not element_material:
            # No material mapping - assign all faces to first material
            if material_count > 0:
                fbx_material = node.GetMaterial(0)
                material_name = fbx_material.GetName()
                if material_name in materials_dict:
                    mesh.add_material(materials_dict[material_name])
            mesh.add_section(0, mesh.get_face_count())
            return
        
        mapping_mode = element_material.GetMappingMode()
        
        if mapping_mode == fbx.FbxLayerElement.EMappingMode.eAllSame:
            # All faces use the same material
            if material_count > 0:
                fbx_material = node.GetMaterial(0)
                material_name = fbx_material.GetName()
                if material_name in materials_dict:
                    mesh.add_material(materials_dict[material_name])
            mesh.add_section(0, mesh.get_face_count())
            
        elif mapping_mode == fbx.FbxLayerElement.EMappingMode.eByPolygon:
            # Each face can have a different material
            # Group consecutive faces with the same material into sections
            polygon_count = fbx_mesh.GetPolygonCount()
            
            # Add all materials to mesh
            for i in range(material_count):
                fbx_material = node.GetMaterial(i)
                material_name = fbx_material.GetName()
                if material_name in materials_dict:
                    mesh.add_material(materials_dict[material_name])
            
            # Build sections based on material indices
            if polygon_count > 0:
                current_material_index = element_material.GetIndexArray().GetAt(0)
                section_start = 0
                
                for i in range(1, polygon_count):
                    material_index = element_material.GetIndexArray().GetAt(i)
                    
                    if material_index != current_material_index:
                        # Material changed - create section
                        section_length = i - section_start
                        mesh.add_section(section_start, section_length, current_material_index)
                        
                        section_start = i
                        current_material_index = material_index
                
                # Add final section
                section_length = polygon_count - section_start
                mesh.add_section(section_start, section_length, current_material_index)
        else:
            # Unsupported mapping mode - create default section
            if material_count > 0:
                fbx_material = node.GetMaterial(0)
                material_name = fbx_material.GetName()
                if material_name in materials_dict:
                    mesh.add_material(materials_dict[material_name])
            mesh.add_section(0, mesh.get_face_count())
    
    def _find_mesh_node_by_name(self, name: str) -> Optional[fbx.FbxNode]:
        """
        Find FBX mesh node by name
        
        Args:
            name: Mesh name to search for
            
        Returns:
            FBX node or None if not found
        """
        result = [None]
        
        def search_recursive(node: fbx.FbxNode):
            if node.GetName() == name:
                node_attribute = node.GetNodeAttribute()
                if node_attribute and node_attribute.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
                    result[0] = node
                    return
            
            for i in range(node.GetChildCount()):
                if result[0]:
                    return
                search_recursive(node.GetChild(i))
        
        search_recursive(self.scene.GetRootNode())
        return result[0]
    
    def _parse_skeleton(self) -> Optional[Skeleton]:
        """
        Parse skeleton from FBX scene
        
        Returns:
            Skeleton instance or None if no skeleton found
        """
        skeleton = Skeleton()
        bone_nodes = []
        
        # Find all skeleton nodes recursively
        self._find_skeleton_nodes_recursive(self.scene.GetRootNode(), bone_nodes)
        
        if not bone_nodes:
            return None
        
        # Create bone map for hierarchy (use node name as key)
        bone_map = {}
        
        # First pass: create all bones
        for bone_index, fbx_node in enumerate(bone_nodes):
            node_name = fbx_node.GetName()
            bone = Bone(node_name, bone_index)
            
            # Get local transform
            translation = fbx_node.LclTranslation.Get()
            rotation = fbx_node.LclRotation.Get()
            scale = fbx_node.LclScaling.Get()
            
            bone.position = np.array([translation[0], translation[1], translation[2]])
            
            # print(f"  {bone_index:3d}-{node_name} rot : {[rotation[0], rotation[1], rotation[2]]}")
            # Convert Euler angles to quaternion
            rot_rad = np.radians([rotation[0], rotation[1], rotation[2]])
            quat = Rotation.from_euler(FbxUtil.FBX_EULER_ORDER, rot_rad).as_quat()
            bone.rotation = quat  # scipy returns [x, y, z, w]
            # print(f"  {bone_index:3d}-{node_name} quat: {quat}")
            
            bone.scale = np.array([scale[0], scale[1], scale[2]])
            
            skeleton.add_bone(bone)
            bone_map[node_name] = bone
        
        # Second pass: set up hierarchy
        for fbx_node in bone_nodes:
            node_name = fbx_node.GetName()
            parent_node = fbx_node.GetParent()
            if parent_node:
                parent_name = parent_node.GetName()
                if parent_name in bone_map:
                    bone = bone_map[node_name]
                    parent_bone = bone_map[parent_name]
                    bone.parent_index = parent_bone.index
        
        # Build hierarchy
        skeleton.build_hierarchy()
        
        skeleton.name = "Skeleton"
        return skeleton
    
    def _find_skeleton_nodes_recursive(self, node: fbx.FbxNode, bone_nodes: List[fbx.FbxNode]):
        """
        Recursively find all skeleton nodes
        
        Args:
            node: Current FBX node
            bone_nodes: List to append skeleton nodes to
        """
        # Check if this node is a skeleton node
        node_attribute = node.GetNodeAttribute()
        if node_attribute:
            attribute_type = node_attribute.GetAttributeType()
            if attribute_type == fbx.FbxNodeAttribute.EType.eSkeleton:
                bone_nodes.append(node)
        
        # Recursively process children
        for i in range(node.GetChildCount()):
            self._find_skeleton_nodes_recursive(node.GetChild(i), bone_nodes)
    
    def _parse_skin_data(self, fbx_mesh: fbx.FbxMesh, mesh: Mesh, skeleton: Skeleton) -> bool:
        """
        Parse skinning data from FBX mesh
        
        Args:
            fbx_mesh: FBX mesh
            mesh: Mesh instance to add skinning data to
            skeleton: Skeleton instance for bone mapping
            
        Returns:
            True if skinning data was found and parsed
        """
        # Get skin deformer count
        skin_count = fbx_mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin)
        if skin_count == 0:
            return False
        
        # Get first skin deformer
        skin = fbx_mesh.GetDeformer(0, fbx.FbxDeformer.EDeformerType.eSkin)
        if not skin:
            return False
        
        vertex_count = mesh.get_vertex_count()
        
        # Initialize arrays for bone weights and indices
        # Support up to 4 bones per vertex
        max_influences = 4
        bone_weights = np.zeros((vertex_count, max_influences), dtype=np.float32)
        bone_indices = np.zeros((vertex_count, max_influences), dtype=np.int32)
        
        # Get cluster count (each cluster represents a bone)
        cluster_count = skin.GetClusterCount()
        
        # Create bone name to index mapping
        bone_name_to_index = {}
        for bone in skeleton.bones:
            bone_name_to_index[bone.name] = bone.index
        
        # Process each cluster
        for cluster_idx in range(cluster_count):
            cluster = skin.GetCluster(cluster_idx)
            if not cluster:
                continue
            
            # Get bone node
            bone_node = cluster.GetLink()
            if not bone_node:
                continue
            
            bone_name = bone_node.GetName()
            if bone_name not in bone_name_to_index:
                continue
            
            bone_idx = bone_name_to_index[bone_name]
            
            # Get control point indices and weights
            control_point_indices = cluster.GetControlPointIndices()
            control_point_weights = cluster.GetControlPointWeights()
            
            # Add weights to vertices
            for i in range(cluster.GetControlPointIndicesCount()):
                vertex_idx = control_point_indices[i]
                weight = control_point_weights[i]
                
                if vertex_idx >= vertex_count:
                    continue
                
                # Find empty slot for this influence
                for influence_idx in range(max_influences):
                    if bone_weights[vertex_idx, influence_idx] == 0:
                        bone_weights[vertex_idx, influence_idx] = weight
                        bone_indices[vertex_idx, influence_idx] = bone_idx
                        break
        
        # Normalize weights
        for i in range(vertex_count):
            weight_sum = np.sum(bone_weights[i])
            if weight_sum > 0:
                bone_weights[i] /= weight_sum
        
        # Set skinning data on mesh
        mesh.set_vertex_attribute(Mesh.BLEND_WEIGHTS, bone_weights)
        mesh.set_vertex_attribute(Mesh.BLEND_INDICES, bone_indices)
        
        return True