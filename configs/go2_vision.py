# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab).
# SPDX-License-Identifier: BSD-3-Clause
"""NaVILA-Bench Go2 inference scene. Import only after creating AppLauncher."""

from dataclasses import MISSING
import math
from pathlib import Path

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DelayedPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from adahvla.locomotion import base_rpy, height_map_lidar, isaac_camera_data, process_depth_image, project_root


class MatterportUSDImporter(TerrainImporter):
    """Load an already converted scene and retain its original collision setup."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device
        self.meshes, self.warp_meshes, self._terrain_flat_patches = {}, {}, {}
        self.env_origins = self.terrain_origins = None
        scene_path = Path(cfg.obj_filepath)
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
        self._xform_prim = prim_utils.create_prim(
            prim_path=cfg.prim_path + "/Matterport",
            translation=(0.0, 0.0, 0.0),
            usd_path=str(scene_path),
        )
        sim_utils.define_collision_properties(
            self._xform_prim.GetPrimPath(), sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        )
        material_path = cfg.prim_path + "/physicsMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        sim_utils.bind_physics_material(self._xform_prim.GetPrimPath(), material_path)
        stage_utils.update_stage()
        self.configure_env_origins()
        self.set_debug_vis(cfg.debug_vis)


@configclass
class MatterportUSDCfg(TerrainImporterCfg):
    class_type: type = MatterportUSDImporter
    terrain_type: str = "matterport"
    obj_filepath: str = MISSING


UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(project_root() / "assets/robots/go2/go2.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0, ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.0, velocity_limit=30.0, stiffness=40.0, damping=1.0,
            friction=0.0, min_delay=4, max_delay=4,
        )
    },
)


@configclass
class ObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_rpy = ObsTerm(func=base_rpy)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ProprioCfg):
        height_map = ObsTerm(
            func=height_map_lidar, params={"sensor_cfg": SceneEntityCfg("lidar_sensor"), "offset": 0.0},
            clip=(-10.0, 10.0),
        )

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_rpy = ObsTerm(func=base_rpy)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):
        rgb_measurement = ObsTerm(func=isaac_camera_data, params={"sensor_cfg": SceneEntityCfg("rgbd_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VizCameraCfg(CameraCfg):
        rgb_measurement = ObsTerm(func=isaac_camera_data, params={"sensor_cfg": SceneEntityCfg("viz_rgb_camera"), "data_type": "rgb"})

    @configclass
    class DepthCfg(CameraCfg):
        rgb_measurement = None
        depth_measurement = ObsTerm(func=process_depth_image, params={"sensor_cfg": SceneEntityCfg("rgbd_camera"), "data_type": "distance_to_image_plane"})

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: CriticCfg = CriticCfg()
    camera_obs: CameraCfg = CameraCfg()
    viz_camera_obs: VizCameraCfg = VizCameraCfg()
    depth_obs: DepthCfg = DepthCfg()


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    terrain = MatterportUSDCfg(
        prim_path="/World/matterport",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
    )
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 2.0]),
        debug_vis=False, mesh_prim_paths=["/World/matterport"],
    )
    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_lower",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -0.991, 0.0, -0.131)),
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32, vertical_fov_range=(0.0, 90.0), horizontal_fov_range=(-180, 180.0), horizontal_res=4.0,
        ),
        debug_vis=False, mesh_prim_paths=["/World/matterport"],
    )
    rgbd_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/rgbd_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.5), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=54.0),
        width=512, height=512, data_types=["rgb", "distance_to_image_plane"],
    )
    viz_rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/viz_rgb_camera",
        offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 0.8), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=100.0),
        width=512, height=512, data_types=["rgb"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=False)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1000.0))
    disk_1 = AssetBaseCfg(
        prim_path="/World/disk_1", spawn=sim_utils.DiskLightCfg(color=(1.0, 1.0, 1.0), intensity=10000.0, radius=50.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 2.6)),
    )
    disk_2 = AssetBaseCfg(
        prim_path="/World/disk_2", spawn=sim_utils.DiskLightCfg(color=(1.0, 1.0, 1.0), intensity=10000.0, radius=50.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 0.0, 2.6)),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.02,
        rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=0.5, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class Go2MatterportVisionCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = {}

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.episode_length_s = 200000.0
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
        self.scene.height_scanner.update_period = self.scene.lidar_sensor.update_period = 4 * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
            },
        )
        self.viewer.eye = (5, 12, 5)
        self.viewer.lookat = (5, 0, 0.0)
