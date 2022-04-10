from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LionFlatCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'JointFR_abad': 0.,   # [rad]
            'JointFL_abad': 0.,   # [rad]
            'JointHR_abad': 0. ,  # [rad]
            'JointHL_abad': 0.,   # [rad]

            'JointFR_hip': 0.6,     # [rad]
            'JointFL_hip': 0.6,   # [rad]
            'JointHR_hip': 0.6,     # [rad]
            'JointHL_hip': 0.6,   # [rad]

            'JointFR_knee': -1.2,   # [rad]
            'JointFL_knee': -1.2,    # [rad]
            'JointHR_knee': -1.2,  # [rad]
            'JointHL_knee': -1.2    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'Joint': 20.}  # [N*m/rad]
        damping = {'Joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lion/urdf/lion.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.7
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    class env( LeggedRobotCfg.env ):
        num_observations = 48

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False


class LionFlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_lion'