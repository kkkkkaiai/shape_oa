from shape_oa.envs.base.shape_config import ShapeRobotCfg, ShapeRobotCfgPPO

class SwerveFlatCfg( ShapeRobotCfg ):
    class env( ShapeRobotCfg.env ):
        # num_envs是环境的数量，num_actions是动作的数量
        num_envs = 1000
        num_actions = 8
        num_obstalce = 1
        num_observations = 36+4 # 4是观测的数量，包括到障碍物的距离和梯度

    class init_state( ShapeRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.1]
        default_joint_angles = {
            "left_back_steer_joint": 0.0,
            "left_back_wheel_joint": 0.0,
            "left_front_steer_joint": 0.0,
            "left_front_wheel_joint": 0.0,
            "right_back_steer_joint": 0.0,
            "right_back_wheel_joint": 0.0,
            "right_front_steer_joint": 0.0,
            "right_front_wheel_joint": 0.0
        }

    class control( ShapeRobotCfg.control ):
        stiffness = {'steer_joint': 0., 'wheel_joint': 0.}  # [N*m/rad]
        damping = {'steer_joint': 0.0000, 'wheel_joint': 0.15}     # [N*m*s/rad]
        action_scale = 0.5
        decimation = 10

    class asset( ShapeRobotCfg.asset):
        file = "{SHAPE_OA_ROOT_DIR}/resources/robots/swerve_description/holonomic_mm_no_arm.urdf"
        name = "swerve"

        obstacle_enabled = True
        obstacle_root = "{SHAPE_OA_ROOT_DIR}/resources/robots/obstacles"
        num_obstacles = 500
        obstacle_file = ['triangle.urdf', 'pentagon.urdf', 'rect.urdf', 'cylinder.urdf']
        obs_dis_preset = 1.8

        penalize_friction_on = ['wheel']
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filte

    class domain_rand( ShapeRobotCfg.domain_rand ):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]

    class rewards( ShapeRobotCfg.rewards ):
        base_height_target = 0.5
        only_positive_rewards = True
        class scales( ShapeRobotCfg.rewards.scales ):
            obstacle_dis = -1.0
            obstacle_grad = -10.0

class SwerveFlatCfgPPO( ShapeRobotCfgPPO ):
    class policy( ShapeRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class runner( ShapeRobotCfgPPO.runner ):
        run_name = ' '
        experiment_name = 'flat_swerve'
        load_run = -1
        max_iterations = 1000

