# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pathlib
import numpy as np
import warp as wp
import math
from dextrah_lab.assets.kuka_allegro.kuka_allegro import KUKA_ALLEGRO_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    # NOTE: the below ranges form the initial ranges for the parameters

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1., 1.),
            "damping_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0. , 0.),
             # NOTE: I don't really care about this one
#            "lower_limit_distribution_params": (0.00, 0.01),
#            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1., 1.),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # NOTE: I don't really care about this one
#    # -- scene
#    reset_gravity = EventTerm(
#        func=mdp.randomize_physics_scene_gravity,
#        mode="interval",
#        is_global_time=True,
#        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
#        params={
#            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
#            "operation": "add",
#            "distribution": "gaussian",
#        },
#    )

@configclass
class DextrahKukaAllegroEnvCfg(DirectRLEnvCfg):
    # Placeholder for objects_dir which targets the directory of objects for training
    objects_dir = "replace_me"
    valid_objects_dir = ["visdex_objects"]

    # env
    sim_dt = 1/120.
    fabrics_dt = 1/60.
    decimation = 2 # 60 Hz
    episode_length_s = 10. #10.0
    fabric_decimation = 2 # number of fabric steps per physics step
    num_sim_steps_to_render=2 # renders every 4 sim steps, so 60 Hz
    num_actions = 11 # 6 palm pose + 5 PCA
    success_timeout = 2.
    # num_observations = 94
    distillation = False
    num_student_observations = 0
    num_teacher_observations = 0
    num_observations = 0
    num_states = 0

    state_space = 0
    observation_space = 0
    action_space = 0

    asymmetric_obs = True
    obs_type = "full"
    simulate_stereo = False
    stereo_baseline = 55 / 1000

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 0.,
                "index_joint_(0|1|2|3)": 0.,
                "middle_joint_(0|1|2|3)": 0.,
                "ring_joint_(0|1|2|3)": 0.,
                "thumb_joint_0": 0.5,
                "thumb_joint_(1|2|3)": 0.
            },
        )
    )
    actuated_joint_names = [
        "iiwa7_joint_1",
        "iiwa7_joint_2",
        "iiwa7_joint_3",
        "iiwa7_joint_4",
        "iiwa7_joint_5",
        "iiwa7_joint_6",
        "iiwa7_joint_7",
        "index_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_0",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_0",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_0",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    
    hand_body_names = [
        "palm_link",
        "index_biotac_tip",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    ]


    module_path = os.path.dirname(__file__)
    root_path = os.path.dirname(os.path.dirname(module_path))
    scene_objects_usd_path = os.path.join(root_path, "assets/scene_objects/")

    table_texture_dir = os.path.join(
        root_path, "assets", "curated_table_textures"
    )
    dome_light_dir = os.path.join(
        root_path, "assets", "dome_light_textures"
    )
    metropolis_asset_dir = os.path.join(
        root_path, "assets", "object_textures"
    )

    # table
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_objects_usd_path + "table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.21 - 0.725 / 2,
                 0.668 - 1.16 / 2,
                 0.25 - 0.03 / 2),
            rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # camera pose in the real world
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
    camera_pos = tf[:3, 3].tolist()
    # camera_rot = [0.6887834, -0.7242703, -0.0299371, -0.0106609]
    camera_rot = [ 0.51567701, -0.52073085,  0.53658829,  0.41831759]
    del tf # this is hacky but it needs to be done because omega conf doesn't support np.ndarray as a primitive
    camera_rand_rot_range = 3
    camera_rand_pos_range = 0.03

    # horizontal fov: 48, vertical fov is h:w ratio
    horizontal_aperture = 21.02
    focal_length = 23.59
    img_width = int(160 * 2)
    img_height = int(120 * 2)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length, focus_distance=400.0, horizontal_aperture=horizontal_aperture, clipping_range=(0.01, 2.)
        ),
        width=img_width,
        height=img_height,
    )

    fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
    focal_px = img_width * 0.5 / math.tan(fov / 2)
    a = focal_px
    b = img_width * 0.5
    c = focal_px
    d = img_height * 0.5
    intrinsic_matrix = [
        [a, 0., b],
        [0., c, d],
        [0., 0., 1.]
    ]

    pred_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    gt_pos_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pos_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2., replicate_physics=False)

    # reward weights
    hand_to_object_weight = 1.
    hand_to_object_sharpness = 10.
    object_to_goal_weight = 5.
    in_success_region_at_rest_weight = 10.
    lift_sharpness = 8.5

    # Goal reaching parameters
    object_goal_tol = 0.1 # m
    success_for_adr = 0.4
    #min_steps_for_dr_change = 240 # number of steps
    min_steps_for_dr_change = 5 * int(episode_length_s / (decimation * sim_dt))

    # Lift criteria
    min_num_episode_steps = 60
    object_height_thresh = 0.15

    # Object spawning params
    x_center = -0.55
    x_width = 0.5#0.4
    y_center = 0.1
    y_width = 0.8 #0.5

    # DR Controls
    enable_adr = True
    num_adr_increments = 50
    starting_adr_increments = 0 # 0 for no DR up to num_adr_increments for max DR 

    # Default dof friction coefficients
    # NOTE: these are set based on how far out they will scale multiplicatively
    # with the above robot_joint_friction EventTerm above.
    starting_robot_dof_friction_coefficients = [
        1., 1., 1., 1., 1., 1., 1., # arm joints
        0.01, 0.01, 0.01, 0.01, # index finger joints
        0.01, 0.01, 0.01, 0.01, # middle finger joints
        0.01, 0.01, 0.01, 0.01, # ring finger joints
        0.01, 0.01, 0.01, 0.01, # thumb finger joints
    ]

    # domain randomization config
    events: EventCfg = EventCfg()

    # These serve to set the maximum value ranges for the different physics parameters
    adr_cfg_dict = {
        "num_increments": num_adr_increments, # number of times you can change the parameter ranges
        "robot_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 2.),
            "damping_distribution_params": (0.5, 2.),
        },
        "robot_joint_friction": {
            "friction_distribution_params": (0., 5.),
        },
        "object_physics_material": {
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.8, 1.0)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.5, 3.),
        },
    }

    # Object disturbance wrench fixed params
    wrench_trigger_every = int(1. / (decimation * sim_dt)) # 1 sec
    torsional_radius = 0.01 # m
    hand_to_object_dist_threshold = .3 # m
    #wrench_prob_per_rollout = 0. # NOTE: currently not used

    # Object scaling
    object_scale_max = 1.75
    object_scale_min = 0.5
    deactivate_object_scaling = True

    aux_coeff = 1.

    # Dictionary of custom parameters for ADR
    # NOTE: first number in range is the starting value, second number is terminal value
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0., 10.)
        },
        "object_spawn": {
            "x_width_spawn": (0., x_width),
            "y_width_spawn": (0., y_width),
            "rotation": (0., 1.)
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.03), # m
            "object_pos_bias": (0.0, 0.02), # m
            "object_rot_noise": (0.0, 0.1), # rad
            "object_rot_bias": (0.0, 0.08), # rad
        },
        "robot_spawn": {
            "joint_pos_noise": (0., 0.35),
            "joint_vel_noise": (0., 1.)
        },
        "robot_state_noise": {
            "robot_joint_pos_noise": (0.0, 0.08), # rad
            "robot_joint_pos_bias": (0.0, 0.08), # rad
            "robot_joint_vel_noise": (0.0, 0.18), # rad
            "robot_joint_vel_bias": (0.0, 0.08), # rad
        },
        "reward_weights": {
            "finger_curl_reg": (-0.01, -0.005),
            "object_to_goal_sharpness": (-15., -20.),
            "lift_weight": (5., 0.)
        },
        "pd_targets": {
            "velocity_target_factor": (1., 0.)
        },
        "fabric_damping": {
            "gain": (10., 20.)
        },
        "observation_annealing": {
            "coefficient": (0., 0.)
        },
    }

    # Action space related parameters
    max_pose_angle = -1.

    # depth randomization parameters
    img_aug_type = "rgb"
    aug_depth = True
    cam_matrix = wp.mat44f()
    cam_matrix[0,0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.
    cam_matrix[3, 2] = 1.e-3
    d_min = 0.5
    d_max = 1.3
    depth_randomization_cfg_dict = {
        # Dropout and random noise blob parameters
        "pixel_dropout_and_randu": {
            "p_dropout": 0.0125 / 4,
            "p_randu": 0.0125 / 4,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Random stick parameters
        "sticks": {
            "p_stick":  0.001 / 4,
            "max_stick_len": 18.,
            "max_stick_width": 3.,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Correlated noise parameters
        "correlated_noise": {
            "sigma_s": 1./2,
            "sigma_d": 1./6,
            "d_max": d_min,
            "d_min": d_max,
        },
        # Normal noise parameters
        "normal_noise": {
            "sigma_theta": 0.01,
            "cam_matrix": cam_matrix,
            "d_max": d_min,
            "d_min": d_max,
        }

    }

    # If enabled, the environment will terminate when time out.
    # This is used for data recording.
    disable_out_of_reach_done = False
    disable_dome_light_randomization = False
    disable_arm_randomization = False
