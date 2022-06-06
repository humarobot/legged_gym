from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LionFlatCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'JointFR_abad': 0.,   # [rad]
            'JointFL_abad': 0.,   # [rad]
            'JointHR_abad': 0. ,  # [rad]
            'JointHL_abad': 0.,   # [rad]

            'JointFR_hip': 0.6,     # [rad]5
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
        stiffness = {'Joint': 40.}  # [N*m/rad]
        damping = {'Joint': 1.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lion/urdf/lion.urdf'
        foot_name = "shank"
        penalize_contacts_on = ["thigh","hip","base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  

    class env( LeggedRobotCfg.env ):
        num_observations = 76

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-.6, .3] # min max [m/s]
            lin_vel_y = [-.3, .3]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]    # min max [rad/s]
            heading = [-3.14, 3.14]
    class rewards:
        class scales:   
            tracking_lin_vel = 1.0 
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05 #上面四个是线速度和角速度的奖励和约束
            termination = -30.0
            orientation = -2.
            base_height = -2.
            torques = -0.00001 #-0.00001
            feet_air_time = 1.0
            # dof_vel = -0.2
            dof_acc = -2.5e-7
            action_rate = -0.01
            collision = -5. #指定的零件发生碰撞就惩罚
            imitation = -25.
            # stand_up = -5
            # smoothness = -0.0025
            dof_pos_limits = -10.0
            # internal_contacts = -6 #暂时没有好的方法做内部碰撞检测

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        imit_sigma = 0.25
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.6 #1.0
        max_contact_force = 100. # forces above this value are penalized

class LionFlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        run_name = ''
        experiment_name = 'flat_lion'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    class policy:
        init_noise_std = 1.
        actor_hidden_dims = [512,256,128]
        critic_hidden_dims = [512,256,128]
        # actor_hidden_dims = [256,128,64]
        # critic_hidden_dims = [256,128,64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid