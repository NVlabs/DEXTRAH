# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os

import functools

import numpy as np
import torch
from colorsys import hsv_to_rgb
import glob
import torch.distributed as dist
import torch.nn.functional as F
from collections.abc import Sequence
from scipy.spatial.transform import Rotation as R
import random
from pxr import Gf, UsdGeom, UsdShade, Sdf

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaacsim.core.utils.prims import set_prim_attribute_value

from .dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg
from .dextrah_kuka_allegro_utils import (
    assert_equals,
    scale,
    compute_absolute_action,
    to_torch
)
from .dextrah_kuka_allegro_constants import (
    NUM_XYZ,
    NUM_RPY,
    NUM_QUAT,
    NUM_HAND_PCA,
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
#    TABLE_LENGTH_X,
#    TABLE_LENGTH_Y,
#    TABLE_LENGTH_Z,
)

# ADR imports
from .dextrah_adr import DextrahADR

# Fabrics imports
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

class DextrahKukaAllegroEnv(DirectRLEnv):
    cfg: DextrahKukaAllegroEnvCfg

    def __init__(self, cfg: DextrahKukaAllegroEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robot_dofs = self.robot.num_joints
        self.cfg.num_actions = 11

        self.num_actions = self.cfg.num_actions
        self.num_observations = (
            self.cfg.num_student_observations if self.cfg.distillation
            else self.cfg.num_teacher_observations
        )
        self.num_teacher_observations = self.cfg.num_teacher_observations
        self.use_camera = self.cfg.distillation
        self.simulate_stereo = self.use_camera and self.cfg.simulate_stereo
        self.stereo_baseline = self.cfg.stereo_baseline

        # buffers for position targets
        self.robot_dof_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.dof_pos_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.dof_vel_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)

        # Dynamically calculate upper and lower pose action limits
        if self.cfg.max_pose_angle <= 0:
            raise ValueError('Max pose angle must be positive')
        self.PALM_POSE_MINS = PALM_POSE_MINS_FUNC(self.cfg.max_pose_angle)
        self.PALM_POSE_MAXS = PALM_POSE_MAXS_FUNC(self.cfg.max_pose_angle)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))

        # finger bodies
        self.hand_bodies = list()
        for body_name in self.cfg.hand_body_names:
            self.hand_bodies.append(self.robot.body_names.index(body_name))
        self.hand_bodies.sort()
        self.num_hand_bodies = len(self.hand_bodies)

        # joint limits
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        # NOTE: this arranges the limits to be in the same joint order as fabrics
        self.robot_dof_lower_limits = joint_pos_limits[..., 0][:, self.actuated_dof_indices]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1][:, self.actuated_dof_indices]

        # Setting the target position for the object
        # TODO: need to make these goals dynamic, sampled at the start of the rollout
        self.object_goal =\
            torch.tensor([-0.5, 0., 0.75], device=self.device).repeat((self.num_envs, 1))

        # Nominal reset states for the robot
        self.robot_start_joint_pos =\
            torch.tensor([-0.85, -0.0,  0.76,  1.25, -1.76, 0.90, 0.64,
                          0.0,  0.3,  0.3,  0.3,
                          0.0,  0.3,  0.3,  0.3,
                          0.0,  0.3,  0.3,  0.3,
                          1.5,  0.60147215,  0.33795027,  0.60845138], device=self.device)
        self.robot_start_joint_pos =\
            self.robot_start_joint_pos.repeat(self.num_envs, 1).contiguous()
        # Start with zero initial velocities and accelerations
        self.robot_start_joint_vel =\
            torch.zeros(self.num_envs, self.num_robot_dofs, device=self.device)

        # Nominal finger curled config
        self.curled_q =\
            torch.tensor([0.0,  0.,  0.,  0., # NOTE: used to be 0.3 for last 3 joints
                          0.0,  0.,  0.,  0.,
                          0.0,  0.,  0.,  0.,
                          1.5,  0.60147215,  0.33795027,  0.60845138], device=self.device)
        self.curled_q = self.curled_q.repeat(self.num_envs, 1).contiguous()

        # Set up ADR
        self.dextrah_adr =\
            DextrahADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
        self.step_since_last_dr_change = 0
        if self.cfg.distillation:
            self.cfg.starting_adr_increments = self.cfg.num_adr_increments
        self.dextrah_adr.set_num_increments(self.cfg.starting_adr_increments)
        self.local_adr_increment = torch.tensor(
            self.cfg.starting_adr_increments,
            device=self.device,
            dtype=torch.int64
        )
        # The global minimum adr increment across all GPUs. initialized to the starting adr
        self.global_min_adr_increment = self.local_adr_increment.clone()

        # Set up fabrics with cuda graph and everything
        self._setup_geometric_fabrics()

        # Preallocate some reward related signals
        self.hand_to_object_pos_error = torch.ones(self.num_envs, device=self.device) 

        # Track success statistics
        self.in_success_region = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_in_success_region = torch.zeros(self.num_envs, device=self.device)
        
        # Unit tensors - used in creating random object rotations during spawn
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Wrench tensors
        self.object_applied_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Object noise
        self.object_pos_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_bias = torch.zeros(self.num_envs, 1, device=self.device)

        self.object_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_noise_width = torch.zeros(self.num_envs, 1, device=self.device)

        # Robot noise
        self.robot_joint_pos_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_bias = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.robot_joint_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_noise_width = torch.zeros(self.num_envs, 1, device=self.device)

        # For querying 3D points on hand
        robot_dir_name = "kuka_allegro"
        robot_name = "kuka_allegro"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(self.urdf_path, self.cfg.hand_body_names,
                                                            self.num_envs, self.device)

        # markers
        self.pred_pos_markers = VisualizationMarkers(
            self.cfg.pred_pos_marker_cfg
        )
        self.gt_pos_markers = VisualizationMarkers(
            self.cfg.gt_pos_marker_cfg
        )

        # original camera poses
        self.camera_pos_orig = torch.tensor(
            self.cfg.camera_pos
        ).to(self.device).unsqueeze(0)
        self.camera_rot_orig = np.array(self.cfg.camera_rot)
        self.camera_rot_eul_orig = R.from_quat(
            self.camera_rot_orig[[1, 2, 3, 0]]
        ).as_euler('xyz', degrees=True)[None, :]
        # tf = np.array([
        #     9.979802254757542679e-01, 5.805126464282436838e-02, -2.579767882449228097e-02, -6.452117743594977251e-01,
        #     2.867907587635045233e-02, -4.936231931159993508e-02, 9.983691061120923971e-01, -7.328016905360382749e-01,
        #     5.668315593050039097e-02, -9.970924792142141779e-01, -5.092747518000630136e-02, 4.559887081479024329e-01,
        #     0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00
        # ]).reshape(4,4)
        tf = np.array([
            7.416679444534866883e-02,-9.902696855667120213e-01,1.177507386359286923e-01,-7.236400044878017468e-01,
            -1.274026398887237732e-01,1.076995435286611930e-01,9.859864987275952508e-01,-6.886495877727516479e-01,
            -9.890742408692511090e-01,-8.812921292808308105e-02,-1.181752422362273985e-01,6.366771698474239516e-01,
            0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00
        ]).reshape(4,4)
        self.camera_pose = np.tile(
            tf, (self.num_envs, 1, 1)
        )
        self.right_to_left_pose = np.array([
            [-1., 0., 0., 0.065],
            [0., -1., 0., -0.062],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        self.camera_right_pos_orig = torch.tensor(
            self.right_to_left_pose[:3, 3]
        ).to(self.device).unsqueeze(0)
        self.camera_right_rot_orig = R.from_matrix(
            self.right_to_left_pose[:3, :3]
        ).as_quat()
        self.camera_right_rot_eul_orig = R.from_quat(
            self.camera_right_rot_orig
        ).as_euler('xyz', degrees=True)[None, :]
        self.camera_right_pose = np.tile(
            self.right_to_left_pose, (self.num_envs, 1, 1)
        )
        self.intrinsic_matrix = torch.tensor(
            self.cfg.intrinsic_matrix,
            device=self.device, dtype=torch.float64
        )
        
        # self.right_to_left_pose = np.array([
        #     [-0.99990465,  0.00241203,  0.01359653,  0.06590756],
        #     [-0.00238818, -0.99999558,  0.00177028, -0.0547107 ],
        #     [0.01360074,  0.00173764,  0.999906  , -0.00340657],
        #     [0.        ,  0.        ,  0.        ,  1.        ]
        # ])

        self.left_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self.left_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self.right_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self.right_rot = torch.zeros(self.num_envs, 4).to(self.device)


        # Set the starting default joint friction coefficients
        friction_coeff = torch.tensor(self.cfg.starting_robot_dof_friction_coefficients,
                                      device=self.device)
        friction_coeff = friction_coeff.repeat((self.num_envs, 1))
        #self.robot.write_joint_friction_to_sim(friction_coeff, self.actuated_dof_indices, None)
        self.robot.data.default_joint_friction_coeff = friction_coeff

    def find_num_unique_objects(self, objects_dir):
        module_path = os.path.dirname(__file__)
        root_path = os.path.dirname(os.path.dirname(module_path))
        scene_objects_usd_path = os.path.join(root_path, "assets/")

        objects_full_path = scene_objects_usd_path + objects_dir + "/USD"

        # List all subdirectories in the target directory
        sub_dirs = sorted(os.listdir(objects_full_path))

        # Filter out all subdirectories deeper than one level
        sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(
            os.path.join(objects_full_path, object_name))]

        num_unique_objects = len(sub_dirs)

        return num_unique_objects

    def _setup_policy_params(self):
        # Determine number of unique objects in target object dir
        if self.cfg.objects_dir not in self.cfg.valid_objects_dir:
            raise ValueError(f"Need to specify valid directory of objects for training: {self.cfg.valid_objects_dir}")

        num_unique_objects = self.find_num_unique_objects(self.cfg.objects_dir)

        self.cfg.num_student_observations = 159
        self.cfg.num_teacher_observations = 167 + num_unique_objects
        if self.cfg.distillation:
            self.cfg.num_observations = self.cfg.num_student_observations
        else:
            self.cfg.num_observations = self.cfg.num_teacher_observations
        self.cfg.num_states = 214 + num_unique_objects

        self.cfg.state_space = self.cfg.num_states
        self.cfg.observation_space = self.cfg.num_observations
        self.cfg.action_space = self.cfg.num_actions

    def _setup_geometric_fabrics(self) -> None:
        # Set the warp cache directory based on device int
        warp_cache_dir = self.device[-1]
        initialize_warp(warp_cache_dir)

        # This creates a world model that book keeps all the meshes
        # in the world, their pose, name, etc.
        print('Creating fabrics world-------------------------------')
        world_filename = 'kuka_allegro_boxes'
        max_objects_per_env = 20
        self.world_model = WorldMeshesModel(batch_size=self.num_envs,
                                            max_objects_per_env=max_objects_per_env,
                                            device=self.device,
                                            world_filename=world_filename)

        # This reports back handles to the meshes which is consumed
        # by the fabric for collision avoidance
        self.object_ids, self.object_indicator = self.world_model.get_object_ids()

        # Control rate and time settings
        #self.timestep = self.sim.get_physics_dt()
        self.timestep = self.cfg.fabrics_dt

        # Create Kuka-Allegro fabric palm pose and finger PCA action spaces
        self.kuka_allegro_fabric =\
            KukaAllegroPoseFabric(self.num_envs, self.device, self.timestep, graph_capturable=True)
        num_joints = self.kuka_allegro_fabric.num_joints
                    
        # Create integrator for the fabric dynamics.
        self.kuka_allegro_integrator = DisplacementIntegrator(self.kuka_allegro_fabric)

        # Pre-allocate fabrics states
        self.fabric_q = self.robot_start_joint_pos.clone().contiguous() 
        # Start with zero initial velocities and accelerations
        self.fabric_qd = torch.zeros(self.num_envs, num_joints, device=self.device)
        self.fabric_qdd = torch.zeros(self.num_envs, num_joints, device=self.device)

        # Pre-allocate target tensors
        pca_dim = 5
        self.hand_pca_targets = torch.zeros(self.num_envs, pca_dim, device=self.device)
        # Palm target is (origin, Euler ZYX)
        pose_dim = 6
        self.palm_pose_targets = torch.zeros(self.num_envs, pose_dim, device=self.device)

        # Fabric cspace damping gain
        self.fabric_damping_gain =\
            self.dextrah_adr.get_custom_param_value("fabric_damping", "gain") *\
            torch.ones(self.num_envs, 1, device=self.device)

        # Graph capture if enabled
        # NOTE: elements of inputs must be in the same order as expected in the set_features function
        # of the fabric
        if self.cfg.use_cuda_graph:
            # Establish inputs
            self.inputs = [self.hand_pca_targets, self.palm_pose_targets, "euler_zyx", # actions in
                           self.fabric_q.detach(), self.fabric_qd.detach(), # fabric state
                           self.object_ids, self.object_indicator, # world model
                           self.fabric_damping_gain]
            # Capture the forward pass of evaluating the fabric given the inputs and integrating one step
            # in time
            self.g, self.fabric_q_new, self.fabric_qd_new, self.fabric_qdd_new =\
                capture_fabric(self.kuka_allegro_fabric,
                               self.fabric_q,
                               self.fabric_qd,
                               self.fabric_qdd,
                               self.timestep,
                               self.kuka_allegro_integrator,
                               self.inputs,
                               self.device)

        # Preallocate tensors for fabrics state meant to go into obs buffer
        self.fabric_q_for_obs = torch.clone(self.fabric_q)
        self.fabric_qd_for_obs = torch.clone(self.fabric_qd)
        self.fabric_qdd_for_obs = torch.clone(self.fabric_qdd)
    
    def _set_pos_marker(self, pos):
        pos = pos + self.scene.env_origins
        self.pred_pos_markers.visualize(pos, self.object_rot)
    
    def _set_gt_pos_marker(self, pos):
        pos = pos + self.scene.env_origins
        self.gt_pos_markers.visualize(pos, self.object_rot)

    def _setup_scene(self):
        # add robot, objects
        # TODO: add goal objects?
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=True)

        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add cameras
        if self.cfg.distillation:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Determine obs sizes for policies and VF
        self._setup_policy_params()

        # Create the objects for grasping
        # self._setup_metropolis_objects()
        self._setup_objects()
        if self.cfg.distillation:
            import omni.replicator.core as rep
            rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
            table_texture_dir = self.cfg.table_texture_dir
            self.table_texture_files = glob.glob(
                os.path.join(table_texture_dir, "*.png")
            )
            self.stage = omni.usd.get_context().get_stage()

            if not self.cfg.disable_dome_light_randomization:
                dome_light_dir = self.cfg.dome_light_dir
                self.dome_light_files = sorted(glob.glob(
                    os.path.join(dome_light_dir, "*.exr")
                ))
                dome_light_texture = random.choice(self.dome_light_files)
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:texture:file"
                ).Set(dome_light_texture)
            else:
                print("Disabling dome light random initialization")

            UsdGeom.Imageable(
                self.stage.GetPrimAtPath("/World/ground")
            ).MakeInvisible()
            # import omni.replicator.core as rep
            # rep.settings.set_render_rtx_realtime(antialiasing="DLAA")

            self.object_textures = glob.glob(
                os.path.join(
                    self.cfg.metropolis_asset_dir,
                    "**", "*.png"
                ), recursive=True
            )
            try:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath("/Environment/defaultLight")
                ).MakeInvisible()
            except:
                pass

    def _setup_objects(self):
        module_path = os.path.dirname(__file__)
        root_path = os.path.dirname(os.path.dirname(module_path))
        scene_objects_usd_path = os.path.join(root_path, "assets/")

        objects_full_path = scene_objects_usd_path + self.cfg.objects_dir + "/USD"

        # List all subdirectories in the target directory
        sub_dirs = sorted(os.listdir(objects_full_path))

        # Filter out all subdirectories deeper than one level
        sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(
            os.path.join(objects_full_path, object_name))]

        self.num_unique_objects = len(sub_dirs)

        # This creates a 1D tensor array of length self.num_envs with values:
        # [0, 1, ...., num_unique_objects-1, 0, 1, ..., num_unique_objects-1]
        # which provides a unique index for each unique object over all envs
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        # self.multi_object_idx = torch.remainder(
        #     torch.arange(self.num_envs)+self.num_envs*local_rank,
        #     self.num_unique_objects
        # ).to(self.device)
        self.multi_object_idx =\
            torch.remainder(torch.arange(self.num_envs), self.num_unique_objects).to(self.device)

        # Create one-hot encoding of object ID for usage as feature input
        self.multi_object_idx_onehot = F.one_hot(
            self.multi_object_idx, num_classes=self.num_unique_objects).float()

        stage = omni.usd.get_context().get_stage()
        self.object_mat_prims = list()
        self.arm_mat_prims = list()
        # Tensor of scales applied to each object. Setup to do this deterministically...
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
        state = torch.get_rng_state() # get the hidden rng state of torch
        torch.manual_seed(42) # set the rng seed
        scale_range = self.cfg.object_scale_max - self.cfg.object_scale_min
        self.total_object_scales = scale_range * torch.rand(total_gpus * self.num_envs, 1, device=self.device) +\
            self.cfg.object_scale_min
        torch.set_rng_state(state) # reset the rng state of torch

        self.device_index = self.total_object_scales.device.index
        self.object_scale = self.total_object_scales[self.device_index * self.num_envs :
                                                     (self.device_index + 1) * self.num_envs]

#        # Create multiplicitive object scaling factor per GPU device to incur more
#        # object diversity
#        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
#        self.object_scales = torch.linspace(self.cfg.object_scale_min,
#                                            self.cfg.object_scale_max,
#                                            total_gpus,
#                                            device=self.device)
#        self.device_index = self.object_scales.device.index
#        # Find the index of object scale that is closest to 1. and replace
#        # it with 1. This ensures that we train on no additional scaling, i.e.,
#        # a multiplicative object scaling of 1 for one gpu device
#        index_closest_to_one_scaling = torch.abs(self.object_scales - 1.).min(dim=0).indices
#        self.object_scales[index_closest_to_one_scaling] = 1.
#
#        # Save object scale across envs
#        self.object_scale = self.object_scales[self.device_index] *\
#                torch.ones(self.num_envs, 1, device=self.device)

        # If object scaling is deactivated, then just set all the scalings to 1.
        if self.cfg.deactivate_object_scaling:
            self.object_scale = torch.ones_like(self.object_scale)

        for i in range(self.num_envs):
            # TODO: check to see that the below config settings make sense
            object_name = sub_dirs[self.multi_object_idx[i]]
            object_usd_path = objects_full_path + "/" + object_name + "/" + object_name + ".usd"
            print('Object name', object_name)
            print('object usd path', object_usd_path)

            object_prim_name = "object_" + str(i) + "_" + object_name
            prim_path = "/World/envs/" + "env_" + str(i) + "/object/" + object_prim_name
            print('Object prim name', object_prim_name)
            print('Object prim path', prim_path)

            print('Object Scale', self.object_scale[i])

            object_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=False,
                        enable_gyroscopic_forces=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                        sleep_threshold=0.005,
                        stabilization_threshold=0.0025,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        max_depenetration_velocity=1000.0,
                    ),
                    scale=(self.object_scale[i],
                           self.object_scale[i],
                           self.object_scale[i]),
                    #scale=(self.object_scales[self.device_index],
                    #       self.object_scales[self.device_index],
                    #       self.object_scales[self.device_index]),
                    # NOTE: density is that of birchwood. might want to see the effect
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.5, 0., 0.5),
                    rot=(1.0, 0.0, 0.0, 0.0)),
            )
            # add object to scene
            object_for_grasping = RigidObject(object_cfg)

            # remove baseLink
            set_prim_attribute_value(
                prim_path=prim_path+"/baseLink",
                attribute_name="physxArticulation:articulationEnabled",
                value=False
            )

            # Get shaders
            prim = stage.GetPrimAtPath(prim_path)
            self.object_mat_prims.append(prim.GetChildren()[0].GetChildren()[0].GetChildren()[0])

            arm_shader_prims = list()
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/arm_gray/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/arm_orange/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/allegro_black/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/allegro_biotac/Shader"
                )
            )
            self.arm_mat_prims.append(arm_shader_prims)
        # Now create one more RigidObject with regex on existing object prims
        # so that we can add all the above objects into one RigidObject object
        # for batch querying their states, forces, etc.
        regex = "/World/envs/env_.*/object/.*"
        multi_object_cfg = RigidObjectCfg(
            prim_path=regex,
            spawn=None,
        )

        # Add to scene
        self.object = RigidObject(multi_object_cfg)
        self.scene.rigid_objects["object"] = self.object

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Find the current global minimum adr increment
        local_adr_increment = self.local_adr_increment.clone()
        # Query for the global minimum adr increment across all GPUs
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.all_reduce(local_adr_increment, op=dist.ReduceOp.MIN)
        self.global_min_adr_increment = local_adr_increment

        self.actions = actions.clone()

        # Update the palm pose and pca targets based on agent actions
        self.compute_actions(self.actions)

        # Update fabric cspace damping gain based on ADR
        fabric_damping_gain = self.dextrah_adr.get_custom_param_value("fabric_damping", "gain") *\
            torch.ones(self.num_envs, 1, device=self.device)
        self.fabric_damping_gain.copy_(fabric_damping_gain)

        # Evaluate fabric without cuda graph
        if not self.cfg.use_cuda_graph:
            self.inputs = [self.hand_pca_targets, self.palm_pose_targets, "euler_zyx", # actions in
                           self.fabric_q.detach(), self.fabric_qd.detach(), # fabric state
                           self.object_ids, self.object_indicator, # world model
                           self.fabric_damping_gain]
            self.kuka_allegro_fabric.set_features(*self.inputs)
            for i in range(self.cfg.fabric_decimation):
                self.fabric_q, self.fabric_qd, self.fabric_qdd = self.kuka_allegro_integrator.step(
                    self.fabric_q.detach(), self.fabric_qd.detach(), self.fabric_qdd.detach(), self.timestep
                    )
        else:
            # Replay through the fabric graph with the latest action inputs
            for i in range(self.cfg.fabric_decimation):
                # Evaluate the fabric via graph replay
                self.g.replay()

                # Update the fabric states
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)

        # Add F/T wrench to object
        self.apply_object_wrench()

    def _apply_action(self) -> None:
        # Set fabric states to position and velocity targets
        self.dof_pos_targets[:, self.actuated_dof_indices] = torch.clone(self.fabric_q)
        self.dof_vel_targets[:, self.actuated_dof_indices] = torch.clone(self.fabric_qd)

        # Set position target
        self.robot.set_joint_position_target(
            self.dof_pos_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        # Set velocity target
        vel_scale = self.dextrah_adr.get_custom_param_value(
            "pd_targets", "velocity_target_factor"
        )
        self.robot.set_joint_velocity_target(
            vel_scale * self.dof_vel_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        policy_obs = self.compute_policy_observations()
        critic_obs = self.compute_critic_observations()

        if self.use_camera and not self.simulate_stereo:
            depth_map = self._tiled_camera.data.output["depth"].clone()
            mask = depth_map.permute((0, 3, 1, 2)) > self.cfg.d_max
            depth_map[depth_map <= 1e-8] = 10
            depth_map[depth_map > self.cfg.d_max] = 0.
            depth_map[depth_map < self.cfg.d_min] = 0.

            student_policy_obs = self.compute_student_policy_observations()
            teacher_policy_obs = self.compute_policy_observations()
            critic_obs = self.compute_critic_observations()

            aux_info = {
                "object_pos": self.object_pos
            }

            observations = {
                "policy": student_policy_obs,
                # "policy": teacher_policy_obs,
                "img": depth_map.permute((0, 3, 1, 2)),
                "rgb": self._tiled_camera.data.output["rgb"].clone().permute((0, 3, 1, 2)) / 255.,
                "expert_policy": teacher_policy_obs,
                "critic": critic_obs,
                "aux_info": aux_info,
                "mask": mask
            }
        elif self.simulate_stereo:
            left_rgb = self._tiled_camera.data.output["rgb"].clone() / 255.
            left_depth = self._tiled_camera.data.output["depth"].clone()
            left_mask = left_depth > self.cfg.d_max*10
            left_depth[left_depth <= 1e-8] = 10
            left_depth[left_depth > self.cfg.d_max] = 0.
            left_depth[left_depth < self.cfg.d_min] = 0.
            right_to_world = torch.from_numpy(
                np.matmul(self.camera_pose, self.camera_right_pose)
            ).to(self.device)
            right_to_world_rot = torch.tensor(R.from_matrix(
                right_to_world[:, :3, :3].cpu().numpy()
            ).as_quat()[:, [3, 0, 1, 2]]).to(self.device)
            self._tiled_camera.set_world_poses(
                positions=right_to_world[:, :3, 3],
                orientations=right_to_world_rot,
                env_ids=self.robot._ALL_INDICES,
                convention="ros"
            )
            self.sim.render()
            self._tiled_camera.update(0, force_recompute=True)
            object_pos_world = torch.cat(
                [
                    self.object_pos,
                    torch.ones(
                        self.object_pos.shape[0], 1,
                        device=self.device,
                        dtype=right_to_world.dtype
                    )
                ], dim=-1
            )
            # (N, 4, 4)
            T_right_world = torch.eye(4, device=self.device, dtype=right_to_world.dtype).unsqueeze(0).repeat(
                self.num_envs, 1, 1
            )
            T_right_world[:, :3, :3] = right_to_world[:, :3, :3].transpose(1, 2)
            T_right_world[:, :3, 3] = torch.bmm(
                -T_right_world[:, :3, :3],
                right_to_world[:, :3, 3:4] - self.scene.env_origins.unsqueeze(-1)
            ).squeeze(-1)
            obj_pos_right = torch.bmm(
                T_right_world,
                object_pos_world.unsqueeze(-1)
            )[:, :3, :]
            obj_uv_right = torch.matmul(
                self.intrinsic_matrix,
                obj_pos_right
            ).squeeze(-1)
            obj_uv_right[:, :2] /= obj_uv_right[:, 2:3]
            # right image is flipped
            obj_uv_right[:, 0] = self.cfg.img_width - obj_uv_right[:, 0]
            obj_uv_right[:, 1] = self.cfg.img_height - obj_uv_right[:, 1]


            # self.sim.render()
            # self._tiled_camera.update(0, force_recompute=True)
            right_rgb = self._tiled_camera.data.output["rgb"].clone() / 255.
            right_depth = self._tiled_camera.data.output["depth"].clone()
            right_mask = right_depth > self.cfg.d_max*10
            right_depth[right_depth <= 1e-8] = 10
            right_depth[right_depth > self.cfg.d_max] = 0.
            right_depth[right_depth < self.cfg.d_min] = 0.
            self._tiled_camera.set_world_poses(
                positions=self.left_pos,
                orientations=self.left_rot,
                env_ids=self.robot._ALL_INDICES,
                convention="ros"
            )

            object_pos_world = torch.cat(
                [
                    self.object_pos,
                    torch.ones(
                        self.object_pos.shape[0], 1,
                        device=self.device,
                        dtype=right_to_world.dtype
                    )
                ], dim=-1
            )
            T_left_world = torch.eye(4, device=self.device, dtype=right_to_world.dtype).unsqueeze(0).repeat(
                self.num_envs, 1, 1
            )
            T_left_world[:, :3, :3] = torch.from_numpy(self.camera_pose[:, :3, :3]).to(self.device).transpose(1, 2)
            T_left_world[:, :3, 3] = torch.bmm(
                -T_left_world[:, :3, :3],
                torch.from_numpy(self.camera_pose[:, :3, 3:4]).to(self.device) - self.scene.env_origins.unsqueeze(-1)
            ).squeeze(-1)
            obj_pos_left = torch.bmm(
                T_left_world,
                object_pos_world.unsqueeze(-1)
            )[:, :3, :]
            obj_uv_left = torch.matmul(
                self.intrinsic_matrix,
                obj_pos_left
            ).squeeze(-1)
            obj_uv_left[:, :2] /= obj_uv_left[:, 2:3]
            # from PIL import Image
            # img_np_left = left_rgb.cpu().numpy()
            # img_np_right = right_rgb.cpu().numpy()
            # im_left = Image.fromarray(img_np_left[0])
            # im_right = Image.fromarray(img_np_right[0])
            student_policy_obs = self.compute_student_policy_observations()
            teacher_policy_obs = self.compute_policy_observations()
            critic_obs = self.compute_critic_observations()

            # normalize uvs by img dims
            obj_uv_left[:, 0] /= self.cfg.img_width
            obj_uv_left[:, 1] /= self.cfg.img_height
            obj_uv_right[:, 0] /= self.cfg.img_width
            obj_uv_right[:, 1] /= self.cfg.img_height

            aux_info = {
                "object_pos": self.object_pos,
                "left_img_depth": left_depth.permute((0, 3, 1, 2)),
                "obj_uv_left": obj_uv_left[:, :2],
                "obj_uv_right": obj_uv_right[:, :2],
            }

            observations = {
                "policy": student_policy_obs,
                "depth_left": left_depth.permute((0, 3, 1, 2)),
                "depth_right": right_depth.permute((0, 3, 1, 2)),
                "mask_left": left_mask.permute((0, 3, 1, 2)),
                "mask_right": right_mask.permute((0, 3, 1, 2)),
                "img_left": left_rgb.permute((0, 3, 1, 2)),
                "img_right": right_rgb.permute((0, 3, 1, 2)),
                "expert_policy": teacher_policy_obs,
                "critic": critic_obs,
                "aux_info": aux_info,
                "obj_uv_left": obj_uv_left[:, :2],
                "obj_uv_right": obj_uv_right[:, :2],
            }
        else:
            observations = {"policy": policy_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Update signals related to reward
        self.compute_intermediate_reward_values()

        (
            hand_to_object_reward,
            object_to_goal_reward,
            finger_curl_reg,
            lift_reward
        ) = compute_rewards(
                self.reset_buf,
                self.in_success_region,
                self.max_episode_length,
                self.hand_to_object_pos_error,
                self.object_to_object_goal_pos_error,
                self.object_vertical_error,
                self.robot_dof_pos[:, 7:], # NOTE: only the finger joints
                self.curled_q,
                self.cfg.hand_to_object_weight,
                self.cfg.hand_to_object_sharpness,
                self.cfg.object_to_goal_weight,
                self.dextrah_adr.get_custom_param_value("reward_weights", "object_to_goal_sharpness"),
                self.dextrah_adr.get_custom_param_value("reward_weights", "finger_curl_reg"),
                self.dextrah_adr.get_custom_param_value("reward_weights", "lift_weight"),
                self.cfg.lift_sharpness
            )

        # Add reward signals to tensorboard
        self.extras["hand_to_object_reward"] = hand_to_object_reward.mean()
        self.extras["object_to_goal_reward"] = object_to_goal_reward.mean()
        self.extras["finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["lift_reward"] = lift_reward.mean()

        total_reward = hand_to_object_reward + object_to_goal_reward +\
                       finger_curl_reg + lift_reward

        # Log other information
        self.extras["num_adr_increases"] = self.dextrah_adr.num_increments()
        self.extras["in_success_region"] = self.in_success_region.float().mean()

        # print('reach reward', hand_to_object_reward.mean())
        # print('lift reward', lift_reward.mean())

        return total_reward

    def _get_dones(self) -> torch.Tensor:
        # This should be in start
        self._compute_intermediate_values()

        # Determine if the object is out of reach by checking XYZ position
        # XY should be within certain limits on the table to be within
        # the allowable work volume as set by fabrics

        # If Z is too low, then it has probably fallen off
        object_outside_upper_x = self.object_pos[:,0] > (self.cfg.x_center + self.cfg.x_width / 2.)
        object_outside_lower_x = self.object_pos[:,0] < (self.cfg.x_center - self.cfg.x_width / 2.)

        object_outside_upper_y = self.object_pos[:,1] > (self.cfg.y_center + self.cfg.y_width / 2.)
        object_outside_lower_y = self.object_pos[:,1] < (self.cfg.y_center - self.cfg.y_width / 2.)

        z_height_cutoff = 0.2
        object_too_low = self.object_pos[:,2] < z_height_cutoff

        out_of_reach = object_outside_upper_x | \
                       object_outside_lower_x | \
                       object_outside_upper_y | \
                       object_outside_lower_y | \
                       object_too_low

        # Terminate rollout if maximum episode length reached
        if self.cfg.distillation:
            time_out = torch.logical_or(
                self.episode_length_buf >= self.max_episode_length - 1,
                self.time_in_success_region >= self.cfg.success_timeout
            )
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1

        #return out_of_reach, time_out
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self.cfg.disable_out_of_reach_done:
            if env_ids.shape[0] != self.num_envs:
                return

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        num_ids = env_ids.shape[0]

        # Reset object state
        object_start_state = torch.zeros(self.num_envs, 13, device=self.device)
        # Shift and scale the X-Y spawn locations
        object_xy = torch.rand(num_ids, 2, device=self.device) - 0.5 # [-.5, .5]
        x_width_spawn = self.dextrah_adr.get_custom_param_value("object_spawn", "x_width_spawn")
        y_width_spawn = self.dextrah_adr.get_custom_param_value("object_spawn", "y_width_spawn")
        object_xy[:, 0] *= x_width_spawn
        object_xy[:, 0] += self.cfg.x_center
        object_xy[:, 1] *= y_width_spawn
        object_xy[:, 1] += self.cfg.y_center
        object_start_state[env_ids, :2] = object_xy
        # Keep drop height the same
        object_start_state[:, 2] = 0.5

        # Randomize rotation
#        rot_noise = sample_uniform(-1.0, 1.0, (num_ids, 2), device=self.device)  # noise for X and Y rotation
#        object_start_state[env_ids, 3:7] = randomize_rotation(
#            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
#        )
        #object_start_state[:, 3] = 1.
        rotation = self.dextrah_adr.get_custom_param_value("object_spawn", "rotation") 
        rot_noise = sample_uniform(-rotation, rotation, (num_ids, 2), device=self.device)  # noise for X and Y rotation
        object_start_state[env_ids, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state = object_start_state[env_ids]

        # Add the env origin translations
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )

        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # Spawning robot
        joint_pos_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_pos_noise")
        joint_vel_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_vel_noise")

        joint_pos_deltas = 2. * (torch.rand_like(self.robot_start_joint_pos[env_ids]) - 0.5)
        joint_vel_deltas = 2. * (torch.rand_like(self.robot_start_joint_vel[env_ids]) - 0.5)

        # Calculate joint positions
        dof_pos = joint_pos_noise * joint_pos_deltas
        dof_pos[:, self.actuated_dof_indices] += self.robot_start_joint_pos[env_ids].clone()
        # Now clamp
        dof_pos[:, self.actuated_dof_indices] = torch.clamp(dof_pos[:, self.actuated_dof_indices],
                                                            min=self.robot_dof_lower_limits[0],
                                                            max=self.robot_dof_upper_limits[0])

        dof_vel = joint_vel_noise * joint_vel_deltas
        dof_vel[:, self.actuated_dof_indices] += self.robot_start_joint_vel[env_ids].clone()

        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        
        # Reset position and velocity targets to the actual robot position and velocity
        self.robot.set_joint_position_target(dof_pos[:, self.actuated_dof_indices],
            env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        self.robot.set_joint_velocity_target(dof_vel[:, self.actuated_dof_indices],
            env_ids=env_ids, joint_ids=self.actuated_dof_indices)

        # Set the fabric state to the reset position and vel
        self.fabric_start_pos = self.fabric_q.clone()
        self.fabric_start_pos[env_ids, :] = dof_pos[:, self.actuated_dof_indices].clone()
        self.fabric_start_vel = self.fabric_qd.clone()
        self.fabric_start_vel[env_ids, :] = dof_vel[:, self.actuated_dof_indices].clone()

        self.fabric_q.copy_(self.fabric_start_pos)
        self.fabric_qd.copy_(self.fabric_start_vel)

        # Poll robot and object data
        self._compute_intermediate_values()

        # Reset success signals
        self.in_success_region[env_ids] = False
        self.time_in_success_region[env_ids] = 0.

        # Get object mass - this is used in F/T disturbance, etc.
        # NOTE: object mass on the CPU, so we only query infrequently
        self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)

        # Get material properties to give to value function
        # TODO: if you want to get mat props then we have to adjust it's size to factor in that
        # all objects are multiple objects due to convex decomp. Before, they were convex hulls.
        #self.object_material_props =\
        #    self.object.root_physx_view.get_material_properties().to(device=self.device).view(self.num_envs, 3)

        # Get robot properties
        self.robot_dof_stiffness = self.robot.root_physx_view.get_dof_stiffnesses().to(device=self.device)
        self.robot_dof_damping = self.robot.root_physx_view.get_dof_dampings().to(device=self.device)
        self.robot_material_props =\
            self.robot.root_physx_view.get_material_properties().to(device=self.device).view(self.num_envs, -1)

#        if self.cfg.events:
#            if "reset" in self.event_manager.available_modes:
#                term = self.event_manager.get_term_cfg("robot_physics_material")
#                #term.params['static_friction_range'] = (0., 3.)
#                self.event_manager.set_term_cfg("robot_physics_material", term)
#                input(self.event_manager.get_term_cfg("robot_physics_material").params['static_friction_range'])
#                input(self.event_manager.active_terms)
#                input('here')
#                self.event_manager.apply(env_ids=env_ids, mode="reset")

#        # NOTE: use the below to debug DR things if needed
#        if self.cfg.events:
#            if "reset" in self.event_manager.available_modes:
#                # Incrementally increase DR while scoping params from physx
#                for i in range(15):
#                    adr_increments = i
#                    print('adr increments', adr_increments)
#                    # Set the level of DR
#                    self.dextrah_adr.set_num_increments(adr_increments)
#                    # Sample and apply the DR
#                    self.event_manager.apply(env_ids=env_ids, mode="reset", global_env_step_count=0)
#                    # Look at what physx directly reports about the param
#                    input(self.robot.root_physx_view.get_dof_friction_coefficients())

        # OBJECT NOISE---------------------------------------------------------------------------
        # Sample widths of uniform distribtion controlling pose bias
        self.object_pos_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_bias") *\
            torch.rand(num_ids, device=self.device)
        self.object_rot_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_bias") *\
            torch.rand(num_ids, device=self.device)

        # Now sample the uniform distributions for bias
        self.object_pos_bias[env_ids, 0] = self.object_pos_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)
        self.object_rot_bias[env_ids, 0] = self.object_rot_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)

        # Sample width of per-step noise
        self.object_pos_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_noise") *\
            torch.rand(num_ids, device=self.device)
        self.object_rot_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_noise") *\
            torch.rand(num_ids, device=self.device)

        # ROBOT NOISE---------------------------------------------------------------------------
        # Sample widths of uniform distribution controlling robot state bias
        self.robot_joint_pos_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_pos_bias") *\
            torch.rand(num_ids, device=self.device)
        self.robot_joint_vel_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_vel_bias") *\
            torch.rand(num_ids, device=self.device)

        # Now sample the uniform distributions for bias
        self.robot_joint_pos_bias[env_ids, 0] = self.robot_joint_pos_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)
        self.robot_joint_vel_bias[env_ids, 0] = self.robot_joint_vel_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)

        # Sample width of per-step noise
        self.robot_joint_pos_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_pos_noise") *\
            torch.rand(num_ids, device=self.device)
        self.robot_joint_vel_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_vel_noise") *\
            torch.rand(num_ids, device=self.device)

#        # Update whether to apply wrench for the episode
#        self.apply_wrench = torch.where(
#            torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout,
#            True,
#            False)

        # Update DR ranges
        if self.cfg.enable_adr:
            if self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change and \
                (self.in_success_region.float().mean() > self.cfg.success_for_adr) and\
                (self.local_adr_increment == self.global_min_adr_increment):
                self.step_since_last_dr_change = 0
                self.dextrah_adr.increase_ranges(increase_counter=True)
                self.event_manager.reset(env_ids=self.robot._ALL_INDICES)
                self.event_manager.apply(env_ids=self.robot._ALL_INDICES, mode="reset", global_env_step_count=0)
                self.local_adr_increment = torch.tensor(self.dextrah_adr.num_increments(), device=self.device, dtype=torch.int64)
            else:
                #print('not increasing DR ranges')
                self.step_since_last_dr_change += 1

        # randomize camera position
        if self.use_camera:
            rand_rots = np.random.uniform(
                -self.cfg.camera_rand_rot_range,
                self.cfg.camera_rand_rot_range,
                size=(num_ids, 3)
            )
            new_rots = rand_rots + self.camera_rot_eul_orig
            new_rots_quat = R.from_euler('xyz', new_rots, degrees=True).as_quat()
            new_rots_quat = new_rots_quat[:, [3, 0, 1, 2]]
            new_rots_quat = torch.tensor(new_rots_quat).to(self.device).float()
            new_pos = self.camera_pos_orig + torch.empty(
                num_ids, 3, device=self.device
            ).uniform_(
                -self.cfg.camera_rand_pos_range,
                self.cfg.camera_rand_pos_range
            )
            np_env_ids = env_ids.cpu().numpy()
            self.camera_pose[np_env_ids, :3, :3] = R.from_euler(
                'xyz', new_rots, degrees=True
            ).as_matrix()
            self.camera_pose[np_env_ids, :3, 3] = (
                new_pos + self.scene.env_origins[env_ids]
            ).cpu().numpy()
            self.left_pos[env_ids] = new_pos + self.scene.env_origins[env_ids]
            self.left_rot[env_ids] = new_rots_quat
            self._tiled_camera.set_world_poses(
                positions=new_pos + self.scene.env_origins[env_ids],
                orientations=new_rots_quat,
                env_ids=env_ids,
                convention="ros"
            )

            rand_rots = np.random.uniform(
                -2, 2, size=(num_ids, 3)
            )
            new_rots = rand_rots + self.camera_right_rot_eul_orig
            new_rots_quat = R.from_euler('xyz', new_rots, degrees=True).as_quat()
            new_rots_quat = new_rots_quat[:, [3, 0, 1, 2]]
            new_rots_quat = torch.tensor(new_rots_quat).to(self.device).float() 
            new_pos = self.camera_right_pos_orig + torch.empty(
                num_ids, 3, device=self.device
            ).uniform_(
                -3e-3, 3e-3
            )
            self.camera_right_pose[np_env_ids, :3, :3] = R.from_euler(
                'xyz', new_rots, degrees=True
            ).as_matrix()
            self.camera_right_pose[np_env_ids, :3, 3] = new_pos.cpu().numpy()


            if self.cfg.disable_dome_light_randomization:
                dome_light_rand_ratio = 0.0
            else:
                dome_light_rand_ratio = 0.3
            if random.random() < dome_light_rand_ratio:
                dome_light_texture = random.choice(self.dome_light_files)
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:texture:file"
                ).Set(dome_light_texture)
                x, y, z, w = R.random().as_quat()
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "xformOp:orient"
                ).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:intensity"
                ).Set(np.random.uniform(1000., 4000.))
                # # Define hue range for cooler colors (e.g., 180° to 300° in HSV)
                # # Hue in colorsys is between 0 and 1, corresponding to 0° to 360°
                # cool_hue_min = 0.5  # 180°
                # cool_hue_max = 0.833  # 300°

                # # Generate random hue within the cooler range
                # hue = np.random.uniform(cool_hue_min, cool_hue_max)

                # # Generate random saturation and value within desired ranges
                # saturation = np.random.uniform(0.5, 1.0)  # Moderate to high saturation
                # value = np.random.uniform(0.5, 1.0)       # Moderate to high brightness

                # # Convert HSV to RGB
                # r, g, b = hsv_to_rgb(hue, saturation, value)

                # self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                #     "inputs:color"
                # ).Set(
                #     Gf.Vec3f(r, g, b)
                # )

            rand_attributes = [
                "diffuse_texture",
                "project_uvw",
                "texture_scale",
                "diffuse_tint",
                "reflection_roughness_constant",
                "metallic_constant",
                "specular_level",
            ]
            attribute_types = [
                Sdf.ValueTypeNames.Asset,
                Sdf.ValueTypeNames.Bool,
                Sdf.ValueTypeNames.Float2,
                Sdf.ValueTypeNames.Color3f,
                Sdf.ValueTypeNames.Float,
                Sdf.ValueTypeNames.Float,
                Sdf.ValueTypeNames.Float,
            ]
            for env_id in np_env_ids:
                mat_prim = self.object_mat_prims[env_id]
                property_names = mat_prim.GetPropertyNames()
                rand_attribute_vals = [
                    random.choice(self.object_textures),
                    True,
                    tuple(np.random.uniform(0.7, 5, size=(2))),
                    tuple(np.random.rand(3)),
                    np.random.uniform(0., 1.),
                    np.random.uniform(0., 1.),
                    np.random.uniform(0., 1.),
                ]
                for attribute_name, attribute_type, value in zip(
                    rand_attributes,
                    attribute_types,
                    rand_attribute_vals,
                ):
                    disp_name = "inputs:" + attribute_name
                    if disp_name not in property_names:
                        shader = UsdShade.Shader(
                            omni.usd.get_shader_from_material(
                                mat_prim.GetParent(),
                                True
                            )
                        )
                        shader.CreateInput(
                            attribute_name, attribute_type
                        )
                    mat_prim.GetAttribute(
                        disp_name
                    ).Set(value)

            if not self.cfg.disable_arm_randomization:
                with Sdf.ChangeBlock():
                    for idx, arm_shader_prim in enumerate(self.arm_mat_prims):
                        if idx not in env_ids:
                            continue
                        for arm_shader in arm_shader_prim:
                            arm_shader.GetAttribute("inputs:reflection_roughness_constant").Set(
                                np.random.uniform(0.2, 1.)
                            )
                            arm_shader.GetAttribute("inputs:metallic_constant").Set(
                                np.random.uniform(0, 0.8)
                            )
                            arm_shader.GetAttribute("inputs:specular_level").Set(
                                np.random.uniform(0., 1.)
                            )
                    for i in np_env_ids:
                        shader_path = f"/World/envs/env_{i}/table/Looks/OmniPBR/Shader"
                        shader_prim = self.stage.GetPrimAtPath(shader_path)
                        shader_prim.GetAttribute("inputs:diffuse_texture").Set(
                            random.choice(self.table_texture_files)
                        )
                        shader_prim.GetAttribute("inputs:diffuse_tint").Set(
                            Gf.Vec3d(
                                np.random.uniform(0.3, 0.6),
                                np.random.uniform(0.2, 0.4),
                                np.random.uniform(0.1, 0.2)
                            )
                        )
                        shader_prim.GetAttribute("inputs:specular_level").Set(
                            np.random.uniform(0., 1.)
                        )
                        shader_prim.GetAttribute("inputs:reflection_roughness_constant").Set(
                            np.random.uniform(0.3, 0.9)
                        )
                        shader_prim.GetAttribute("inputs:texture_rotate").Set(
                            np.random.uniform(0., 2*np.pi)
                        )

    def _compute_intermediate_values(self):
        # Data from robot--------------------------
        # Robot measured joint position and velocity
        self.robot_dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        self.robot_dof_pos_noisy = self.robot_dof_pos +\
            self.robot_joint_pos_noise_width *\
            2. * (torch.rand_like(self.robot_dof_pos) - 0.5) +\
            self.robot_joint_pos_bias

        self.robot_dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        self.robot_dof_vel_noisy = self.robot_dof_vel +\
            self.robot_joint_vel_noise_width *\
            2. * (torch.rand_like(self.robot_dof_vel) - 0.5) +\
            self.robot_joint_vel_bias
        self.robot_dof_vel_noisy *= self.dextrah_adr.get_custom_param_value(
            "observation_annealing"
            ,"coefficient"
        )

        # Robot fingertip and palm position. NOTE: currently not adding orientation
        self.hand_pos = self.robot.data.body_pos_w[:, self.hand_bodies]
        self.hand_pos -= self.scene.env_origins.repeat((1, self.num_hand_bodies
            )).reshape(self.num_envs, self.num_hand_bodies, 3)

        # Robot fingertip and palm velocity. 6D
        self.hand_vel = self.robot.data.body_vel_w[:, self.hand_bodies]

        # Noisy hand point position and velocity as calculated from fabric taskmap
        self.hand_pos_noisy, hand_points_jac = self.hand_points_taskmap(self.robot_dof_pos_noisy, None)
        self.hand_vel_noisy = torch.bmm(hand_points_jac, self.robot_dof_vel_noisy.unsqueeze(2)).squeeze(2)
        self.hand_vel_noisy *= self.dextrah_adr.get_custom_param_value(
            "observation_annealing"
            ,"coefficient"
        )

        # Query the finger forces
        self.hand_forces =\
            self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.hand_bodies]
        self.hand_forces =\
            self.hand_forces.view(self.num_envs, self.num_hand_bodies * 6)
        # Query the measured torque on the joints
        self.measured_joint_torque =\
            self.robot.root_physx_view.get_dof_projected_joint_forces()

        # Data from objects------------------------
        # Object translational position, 3D
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # NOTE: noise on object pos and rot is per-step sampled uniform noise and sustained
        # bias noise sampled only at start of rollout
        self.object_pos_noisy = self.object_pos +\
            self.object_pos_noise_width *\
            2. * (torch.rand_like(self.object_pos) - 0.5) +\
            self.object_pos_bias

        # Object rotational position, 4D
        self.object_rot = self.object.data.root_quat_w
        self.object_rot_noisy = self.object_rot +\
            self.object_rot_noise_width *\
            2. * (torch.rand_like(self.object_rot) - 0.5) +\
            self.object_rot_bias

        # Object full velocity, 6D
        self.object_vel = self.object.data.root_vel_w

        # Compute table data
        self.table_pos = self.table.data.root_pos_w - self.scene.env_origins
        self.table_pos_z = self.table_pos[:, 2]

        # Update fabric data
        self.fabric_q_for_obs.copy_(self.fabric_q)
        self.fabric_qd_for_obs.copy_(
            self.fabric_qd * self.dextrah_adr.get_custom_param_value(
                "observation_annealing"
                ,"coefficient"
            )
        )
        self.fabric_qdd_for_obs.copy_(
            self.fabric_qdd * self.dextrah_adr.get_custom_param_value(
                "observation_annealing"
                ,"coefficient"
            )
        )

    def compute_intermediate_reward_values(self):
        # Calculate distance between object and its goal position
        self.object_to_object_goal_pos_error =\
            torch.norm(self.object_pos - self.object_goal, dim=-1)

        # Calculate vertical error
        self.object_vertical_error = torch.abs(self.object_goal[:, 2] - self.object_pos[:, 2])

        # Calculate whether object is within success region
        self.in_success_region = self.object_to_object_goal_pos_error < self.cfg.object_goal_tol
        # if not in success region, reset time in success region, else increment
        self.time_in_success_region = torch.where(
            self.in_success_region,
            self.time_in_success_region + self.cfg.sim.dt*self.cfg.decimation,
            0.
        )

        # Object to palm and fingertip distance
        # It is a max over the distances from points on hand to object
        self.hand_to_object_pos_error =\
            torch.norm(self.hand_pos - self.object_pos[:, None, :], dim=-1).max(dim=-1).values

    def compute_actions(self, actions: torch.Tensor) -> None: #torch.Tensor:
        assert_equals(actions.shape, (self.num_envs, self.cfg.num_actions))

        # Slice out the actions for the palm and the hand
        palm_actions = actions[:, : (NUM_XYZ + NUM_RPY)]
        hand_actions = actions[
            :, (NUM_XYZ + NUM_RPY) : (NUM_HAND_PCA + NUM_XYZ + NUM_RPY)
        ]

        # In-place update to palm pose targets
        self.palm_pose_targets.copy_(
            compute_absolute_action(
                raw_actions=palm_actions,
                lower_limits=self.palm_pose_lower_limits,
                upper_limits=self.palm_pose_upper_limits,
            )
        )

        # In-place update to hand PCA targets
        self.hand_pca_targets.copy_(
            compute_absolute_action(
                raw_actions=hand_actions,
                lower_limits=self.hand_pca_lower_limits,
                upper_limits=self.hand_pca_upper_limits,
            )
        )

    def compute_student_policy_observations(self):
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos_noisy, # 0:23
                self.robot_dof_vel_noisy, # 23:46
                self.hand_pos_noisy, # 46:61
                self.hand_vel_noisy, # 61:76
                # object goal
                self.object_goal, # 76:79
                # last action
                self.actions, # 79:90
                # fabric states
                self.fabric_q_for_obs, # 90:113
                self.fabric_qd_for_obs, # 113:136
                self.fabric_qdd_for_obs, # 136:159
            ),
            dim=-1,
        )

        return obs

    def compute_policy_observations(self):
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos_noisy,
                self.robot_dof_vel_noisy,
                #self.hand_pos.view(self.num_envs, self.num_hand_bodies * 3),
                #self.hand_vel.view(self.num_envs, self.num_hand_bodies * 6),
                self.hand_pos_noisy,
                self.hand_vel_noisy,
                # noisy object position, orientation
                self.object_pos_noisy,
                self.object_rot_noisy,
                #self.object_vel, # NOTE: took this out because it's fairly privileged
                # object goal
                self.object_goal,
                # one-hot encoding of object ID
                self.multi_object_idx_onehot,
                # object scales
                self.object_scale,
                # last action
                self.actions,
                # fabric states
                self.fabric_q_for_obs,
                self.fabric_qd_for_obs,
                self.fabric_qdd_for_obs,
            ),
            dim=-1,
        )

        return obs

    def compute_critic_observations(self):
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.hand_pos.view(self.num_envs, self.num_hand_bodies * 3),
                self.hand_vel.view(self.num_envs, self.num_hand_bodies * 6),
                self.hand_forces[:, :3], # 3D forces on fingertips and palm.
                self.measured_joint_torque, # meeasured joint torque
                # object
                self.object_pos,
                self.object_rot,
                self.object_vel,
                # object goal
                self.object_goal,
                # one-hot encoding of object ID
                self.multi_object_idx_onehot,
                # object scale
                self.object_scale,
                # last action
                self.actions,
                # fabric states
                self.fabric_q.clone(),
                self.fabric_qd.clone(),
                self.fabric_qdd.clone(),
                # dr values for robot
                # TODO: should scale dof stiffness and damping if you want them.
                # NOTE: probably don't need them because dynamic response for robot
                # is always available and policy can adjust
#                self.robot_dof_stiffness, 
#                self.robot_dof_damping, # TODO: probably should scale these to be 0, 1
                #self.robot_material_props,
                # dr values for object
                #self.object_mass,
                #self.object_material_props
            ),
            dim=-1,
        )

        return obs

    def apply_object_wrench(self):
        # Update whether to apply wrench based on whether object is at goal
        self.apply_wrench = torch.where(
            self.hand_to_object_pos_error <= self.cfg.hand_to_object_dist_threshold,
            True,
            False
        )

        body_ids = None # targets all bodies
        env_ids = None # targets all envs

        num_bodies = self.object.num_bodies

        # Generates the random wrench
        max_linear_accel = self.dextrah_adr.get_custom_param_value("object_wrench", "max_linear_accel")
        linear_accel = max_linear_accel * torch.rand(self.num_envs, 1, device=self.device)
        max_force = (linear_accel * self.object_mass).unsqueeze(2)
        max_torque = (self.object_mass * linear_accel * self.cfg.torsional_radius).unsqueeze(2)
        forces =\
            max_force * torch.nn.functional.normalize(
                torch.randn(self.num_envs, num_bodies, 3, device=self.device),
                dim=-1
            )
        torques =\
            max_torque * torch.nn.functional.normalize(
                torch.randn(self.num_envs, num_bodies, 3, device=self.device),
                dim=-1
            )
        
        self.object_applied_force = torch.where(
            (self.episode_length_buf.view(-1, 1, 1) % self.cfg.wrench_trigger_every) == 0,
            forces,
            self.object_applied_force
        )

        self.object_applied_force = torch.where(
            self.apply_wrench[:, None, None],
            self.object_applied_force,
            torch.zeros_like(self.object_applied_force)
        )

        self.object_applied_torque = torch.where(
            (self.episode_length_buf.view(-1, 1, 1) % self.cfg.wrench_trigger_every) == 0,
            torques,
            self.object_applied_torque
        )

        self.object_applied_torque = torch.where(
            self.apply_wrench[:, None, None],
            self.object_applied_torque,
            torch.zeros_like(self.object_applied_torque)
        )

        # Set the wrench to the buffers
        self.object.set_external_force_and_torque(
            forces=self.object_applied_force,
            torques=self.object_applied_torque,
            body_ids = body_ids,
            env_ids = env_ids
        )

        # Write wrench data to sim
        self.object.write_data_to_sim()

    @property
    @functools.lru_cache()
    def hand_pca_lower_limits(self) -> torch.Tensor:
        return to_torch(HAND_PCA_MINS, device=self.device)

    @property
    @functools.lru_cache()
    def hand_pca_upper_limits(self) -> torch.Tensor:
        return to_torch(HAND_PCA_MAXS, device=self.device)

    @property
    @functools.lru_cache()
    def palm_pose_lower_limits(self) -> torch.Tensor:
        return to_torch(self.PALM_POSE_MINS, device=self.device)

    @property
    @functools.lru_cache()
    def palm_pose_upper_limits(self) -> torch.Tensor:
        return to_torch(self.PALM_POSE_MAXS, device=self.device)

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    in_success_region: torch.Tensor,
    max_episode_length: float,
    hand_to_object_pos_error: torch.Tensor,
    object_to_object_goal_pos_error: torch.Tensor,
    object_vertical_error: torch.Tensor,
    robot_dof_pos: torch.Tensor,
    curled_q: torch.Tensor,
    hand_to_object_weight: float,
    hand_to_object_sharpness: float,
    object_to_goal_weight: float,
    object_to_goal_sharpness: float,
    finger_curl_reg_weight: float,
    lift_weight: float,
    lift_sharpness: float
):

    # Reward for moving fingertip and palm points closer to object centroid point
    hand_to_object_reward = hand_to_object_weight * torch.exp(-hand_to_object_sharpness * hand_to_object_pos_error)

    # Reward for moving the object to the goal translational position
    object_to_goal_reward =\
        object_to_goal_weight * torch.exp(object_to_goal_sharpness * object_to_object_goal_pos_error)

    # Regularizer on hand joints via the fabric state towards a nominally curled config
    # I brought this in because the fingers seem to curl in a lot to play with the object
    # A good strategy is to approach the object with wider set fingers and then encase the object
    # flexing inwards
    finger_curl_dist = (robot_dof_pos - curled_q).norm(p=2, dim=-1)
    finger_curl_reg =\
        finger_curl_reg_weight * finger_curl_dist ** 2

    # Reward for lifting object off table and towards object goal
    lift_reward = lift_weight * torch.exp(-lift_sharpness * object_vertical_error)

    return hand_to_object_reward, object_to_goal_reward, finger_curl_reg, lift_reward
    
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
