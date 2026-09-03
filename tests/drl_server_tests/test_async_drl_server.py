# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import warnings
warnings.filterwarnings('ignore')
import gzdrl as grl
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
unittest.TestLoader.sortTestMethodsUsing = None

NUM_SERVERS = 5
ENVIDS = list(range(NUM_SERVERS))
UAV_MODEL_NAME = ["quadrotor" for _ in range(NUM_SERVERS)  ]
UAV_BASE_LINK_NAME = ["quadrotor/base_link" for _ in range(NUM_SERVERS) ]
HOOP_MODEL_NAME = ["dynamic_hoop" for _ in range(NUM_SERVERS) ]
HOOP_ARM_LINK_NAME = ["arm_link" for _ in range(NUM_SERVERS) ]
CAR_MODEL_NAME = ["car" for _ in range(NUM_SERVERS) ]
CAR_CHASSIS_LINK_NAME = ["chassis" for _ in range(NUM_SERVERS) ]
rl_server = grl.AsyncDRLServerPool(NUM_SERVERS, "_0",
                                  "world_test.sdf",
                                  ["dynamic_hoop",
                                   "quadrotor",
                                   "car"],
                                   False)

class TestDRLServer(unittest.TestCase):

    def test_01_set_cpp_controller_with_rotor_velocity(self):
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME,
                                      UAV_BASE_LINK_NAME,
                                      controller)
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)],
                            [(0.0, 0.0, 0.0)])
        rl_server.run_N(ENVIDS, [10])  # Let it stabilize a bit
        desired_state = [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME,
                                                    UAV_BASE_LINK_NAME,
                                                    desired_state,
                                                    [5000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        self.assertIsNotNone(states)
        self.assertEqual(len(states), NUM_SERVERS)

        for i, state in enumerate(states):
            self.assertEqual(len(state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]), 13)
            self.assertAlmostEqual(state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0][0]-desired_state[0][0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0][1]-desired_state[0][1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0][2]-desired_state[0][2])**2, 0.0, delta=0.1)
            self.assertAlmostEqual(state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0][6]-desired_state[0][6], 0.0,delta=0.1)

        # If no exception is raised, the test passes

    def test_02_set_python_controller_with_rotor_velocity(self):
        class SimplePythonController(grl.UAVController):
            def __init__(self):
                super().__init__()

            def calculate_force(self, state, state_dot,
                                 target_state):

                return [0.0, 0.0, 0.0]  # Force cmd
            def calculate_moments(self, state, state_dot,
                                 target_state, force):
                return [0.0, 0.0, 0.0]  # Moment cmd

        python_controller = [SimplePythonController() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME,
                                      UAV_BASE_LINK_NAME,
                                      python_controller)
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)],
                            [(0.0, 0.0, 0.0)])
        desired_state = [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        rl_server.control_with_rotor_velocity(ENVIDS,
                                                    UAV_MODEL_NAME,
                                                    UAV_BASE_LINK_NAME,
                                                    desired_state,
                                                    [5000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        self.assertIsNotNone(states)
        self.assertEqual(len(states), NUM_SERVERS)
        for i, state in enumerate(states):
            self.assertEqual(len(state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]), 13)
        # No state checks since controller is dummy

    def test_03_set_cpp_controller_with_wrench(self):
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass*1.5/1.2,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS,
                                      UAV_MODEL_NAME,
                                      UAV_BASE_LINK_NAME,
                                      controller)
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)],
                            [(0.0, 0.0, 0.0)])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertAlmostEqual(np.linalg.norm(state[3:6]), 0.0, delta=0.2)
        rl_server.run_N(ENVIDS, [10])  # Let it stabilize a bit
        desired_state = [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        rl_server.control_with_wrench(ENVIDS,
                                                    UAV_MODEL_NAME,
                                                    UAV_BASE_LINK_NAME,
                                                    desired_state,
                                                    [10000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        self.assertIsNotNone(states)
        for i, state in enumerate(states):
            state = state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-desired_state[0][0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-desired_state[0][1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[2]-desired_state[0][2])**2, 0.0, delta=0.1)
            self.assertAlmostEqual(state[6]-desired_state[0][6], 0.0,delta=0.1)
        # If no exception is raised, the test passes

    def test_04_set_python_controller_with_wrench(self):
        class SimplePythonController(grl.UAVController):
            def calculate_force(self, state, state_dot,
                                 target_state):
                return [0.0, 0.0, 0.0]  # Force cmd
            def calculate_moments(self, state, state_dot,
                                 target_state,
                                 force):
                return [0.0, 0.0, 0.0]  # Moment cmd

        python_controller = [SimplePythonController() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME,
                                      UAV_BASE_LINK_NAME,
                                      python_controller)
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)],
                            [(0.0, 0.0, 0.0)])
        desired_state = [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        rl_server.control_with_wrench(ENVIDS,
                                                    UAV_MODEL_NAME,
                                                    UAV_BASE_LINK_NAME,
                                                    desired_state,
                                                    [5000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        self.assertIsNotNone(states)
        for i, state in enumerate(states):
            state = state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
        # No state checks since controller is dummy

    def test_05_reset_pose(self):
        new_pose = ([5.0, 5.0, 5.0],[1.0, 2.0, 3.0])
        rotmat = Rotation.from_euler('xyz', new_pose[1]).as_matrix()
        new_pos = [new_pose[0] for _ in range(NUM_SERVERS)]
        new_ori = [new_pose[1] for _ in range(NUM_SERVERS)]

        for i in range(1000):
            rl_server.set_velocity_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [[1.0, 1.0 ,1.0]])
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = states[ENVIDS[i]][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[3], 1.0,delta=0.1)
            self.assertAlmostEqual(state[4], 1.0,delta=0.1)
            self.assertAlmostEqual(state[5], 1.0,delta=0.1)

        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, new_pos, new_ori)
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        expected_z_vel = -9.82*13*1e-3
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = states[ENVIDS[i]][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-new_pos[i][0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-new_pos[i][1], 0.0,delta=0.1)
            self.assertAlmostEqual(state[2]-new_pos[i][2], 0.0,delta=0.1)
            self.assertAlmostEqual(state[3], 0.0,delta=0.1)
            self.assertAlmostEqual(state[4], 0.0,delta=0.1)
            self.assertAlmostEqual(state[5], expected_z_vel,delta=0.1)
            rotmat_after = Rotation.from_quat(state[6:10], scalar_first=True).as_matrix()
            self.assertAlmostEqual(np.linalg.norm(rotmat - rotmat_after), 0.0, delta=0.1)

        # If no exception is raised, the test passes

    def test_06_set_trajectory_trace(self):
        serv_config =[ grl.DRLServerConfig() for _ in range(NUM_SERVERS)]
        for config in serv_config:
            config.trajectory_viz = True
        rl_server.set_trajectory_trace(ENVIDS,
                                       UAV_MODEL_NAME, UAV_BASE_LINK_NAME, serv_config)
        rl_server.run_N(ENVIDS, [5])
        # If no exception is raised, the test passes


    def test_07_set_wrench(self):
        force = [(0.1, 0.1 , 0.1) for _ in range(NUM_SERVERS)]
        torque = [(0.01, 0.01, 0.01) for _ in range(NUM_SERVERS)]
        rl_server.set_wrench(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, force, torque)
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = state[UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
        # If no exception is raised, the test passes

        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0,0.0,0.0)], [(0.0,0.0,0.0)])
        rl_server.run_N(ENVIDS, [10])
        # If no exception is raised, the test passes

    def test_08_set_rotor_velocity(self):
        rotor_velocities = [[1000.0, 1000.0, 1000.0, 1000.0] for _ in range(NUM_SERVERS)]
        rl_server.set_rotor_velocity_cmd(ENVIDS,
                                              UAV_MODEL_NAME,
                                              UAV_BASE_LINK_NAME,
                                                rotor_velocities)
        rl_server.run_N(ENVIDS, [10 ])
        rl_server.update_control_states(ENVIDS)
        state = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            statei = state[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(statei)
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0,0.0,0.0) for _ in range(NUM_SERVERS)], [(0.0,0.0,0.0) for _ in range(NUM_SERVERS)])
        rl_server.run_N(ENVIDS, [10 ])
        # If no exception is raised, the test passes

    def test_09_contact_sensor_data(self):
        rl_server.request_contact_data(ENVIDS, UAV_MODEL_NAME)
        rl_server.run_N(ENVIDS, [10])
        sensor_data = rl_server.get_contacts(ENVIDS, UAV_MODEL_NAME)
        self.assertIsNotNone(sensor_data)
        for data in sensor_data:
            self.assertIsInstance(data, dict)
        # If no exception is raised, the test passes

    def test_10_velocity_mapping_function(self):
        mapping_func = rl_server.get_thrust_moment_to_rotor_velocity_mapping_function(ENVIDS,
                                                                                      UAV_MODEL_NAME)
        self.assertIsNotNone(mapping_func)
        cmds = [mapping_func[i]([0.0, 0.0, 0.0, 0.0]) for i in range(NUM_SERVERS)]
        rl_server.set_rotor_velocity_cmd(ENVIDS, UAV_MODEL_NAME,
                                              UAV_BASE_LINK_NAME,
                                              cmds)
        rl_server.run_N(ENVIDS, [10])
        # If no exception is raised, the test passes

    def test_11_allocation_matrix(self):
        allocation_matrix = rl_server.get_rotor_thrust_allocation_matrix(ENVIDS, UAV_MODEL_NAME)
        self.assertIsNotNone(allocation_matrix)
        self.assertEqual(len(allocation_matrix), NUM_SERVERS)
        for matrix in allocation_matrix:
            self.assertEqual(len(matrix), 4)  # 4 outputs (thrust + 3 moments)
            self.assertEqual(len(matrix[0]), 4)  # 4 inputs (4 rotors)
        # Test inverse
        inv_allocation_matrix = rl_server.get_inverse_rotor_thrust_allocation_matrix(ENVIDS,
                                                                              UAV_MODEL_NAME)
        self.assertIsNotNone(inv_allocation_matrix)
        self.assertEqual(len(inv_allocation_matrix), NUM_SERVERS)
        for matrix in inv_allocation_matrix:
            self.assertEqual(len(matrix), 4)  # 4 inputs
            self.assertEqual(len(matrix[0]), 4)  # 4
        for i in range(NUM_SERVERS):
            allocation_matrixi = np.array(allocation_matrix[i])
            inv_allocation_matrixi = np.array(inv_allocation_matrix[i])
            self.assertAlmostEqual((inv_allocation_matrixi@allocation_matrixi).ravel().sum(),
                               np.eye((4)).ravel().sum(),delta=0.1)

    def test_12_get_rotor_parameters(self):
        rotor_params = rl_server.get_rotor_parameters(ENVIDS,
                                                      UAV_MODEL_NAME)
        self.assertIsNotNone(rotor_params)
        self.assertEqual(len(rotor_params), NUM_SERVERS)
        for rotor_paramsi in rotor_params:
            self.assertIsInstance(rotor_paramsi.max_rot_velocity, float)
            self.assertEqual(len(rotor_paramsi.thrust_constant_quadratic_params), 3)
            self.assertEqual(len(rotor_paramsi.torque_constant_quadratic_params), 3)
            self.assertIsInstance(rotor_paramsi.ground_effect_constant, float)
            self.assertIsInstance(rotor_paramsi.time_constant_up, float)
            self.assertIsInstance(rotor_paramsi.time_constant_down, float)
            self.assertIsInstance(rotor_paramsi.rotor_drag_coefficient, float)
            self.assertIsInstance(rotor_paramsi.rotor_inertia, float)
            self.assertIsInstance(rotor_paramsi.rolling_moment_coefficient, float)

    def test_13_set_rotor_parameters(self):
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        original_params = rl_server.get_rotor_parameters(ENVIDS, UAV_MODEL_NAME)
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, controller)
        # verify we can track before change
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)

        new_params = [grl.RotorParameters() for _ in range(NUM_SERVERS)]
        for param in new_params:
            param.max_rot_velocity = 15.0
            param.thrust_constant_quadratic_params = [0e-6, 1e-7, 1e-8]
            param.torque_constant_quadratic_params = [0e-7, 2e-8, 2e-9]
            param.ground_effect_constant = 0.05
            param.time_constant_up = 1.5
            param.time_constant_down = 2.5
            param.rotor_drag_coefficient = 0.02
            param.rotor_inertia = 1.5e-5
            param.rolling_moment_coefficient = 0.015
        rl_server.set_rotor_parameters(ENVIDS, UAV_MODEL_NAME, new_params)
        updated_params = rl_server.get_rotor_parameters(ENVIDS, UAV_MODEL_NAME)
        for i in range(NUM_SERVERS):
            self.assertEqual(updated_params[i].max_rot_velocity, 15.0)
            self.assertEqual((updated_params[i].thrust_constant_quadratic_params - np.array([0e-6, 1e-7, 1e-8])).sum(), 0.0)
            self.assertEqual((updated_params[i].torque_constant_quadratic_params - np.array([0e-7, 2e-8, 2e-9])).sum(), 0.0)
            self.assertEqual(updated_params[i].ground_effect_constant, 0.05)
            self.assertEqual(updated_params[i].time_constant_up, 1.5)
            self.assertEqual(updated_params[i].time_constant_down, 2.5)
            self.assertEqual(updated_params[i].rotor_drag_coefficient, 0.02)
            self.assertEqual(updated_params[i].rotor_inertia, 1.5e-5)
            self.assertEqual(updated_params[i].rolling_moment_coefficient, 0.015)

        # verify after change we can't track
        # verify parameters has changed when respawn_model is called
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.run_N(ENVIDS, [10])
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
        # restore original params
        rl_server.set_rotor_parameters(ENVIDS, UAV_MODEL_NAME, original_params)
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        # If no exception is raised, the test passes

    def test_14_set_and_get_inertia(self):
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        original_inertia = rl_server.get_inertia(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME)
        test_inertia = [100*np.array([[0.02, 0.0, 0.0],
                                      [0.0, 0.03, 0.0],
                                      [0.0, 0.0, 0.04]]) for _ in range(NUM_SERVERS)]
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, controller)
        rl_server.set_inertia(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, test_inertia)
        retrieved_inertia = rl_server.get_inertia(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME)
        # verify that inertia has not changed when reset_pos is called
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)
        # verify inertia has changed when respawn_model is called
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
            self.assertTrue(np.allclose(test_inertia[i], retrieved_inertia[i]))
        # set back original param
        rl_server.set_inertia(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, original_inertia)
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        # If no exception is raised, the test passes

    def test_15_set_and_get_mass(self):
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.set_controller(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, controller)
        original_mass = rl_server.get_mass(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME)
        test_mass = [3.5 for _ in range(NUM_SERVERS)]  # kg
        rl_server.set_mass(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, test_mass)
        retrieved_mass = rl_server.get_mass(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME)
        # verify that mass has not changed when reset_pos is called
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
        # verify mass has changed when respawn_model is called
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.control_with_rotor_velocity(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                              [[1,1,1.5, 0,0,0,1,0,0,0,0,0,0]], [3000])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state = states[i][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)
            self.assertAlmostEqual(test_mass[i], retrieved_mass[i])
        # reset mass to original
        rl_server.set_mass(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, original_mass)
        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.05)], [(0.0, 0.0, 0.0)])
        # If no exception is raised, the test passes

    def test_16_control_states(self):
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for statei in states:
            self.assertIn("quadrotor", statei)
            self.assertIn("quadrotor/base_link", statei["quadrotor"])
            state = statei["quadrotor"]["quadrotor/base_link"][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            # check for other models
            self.assertIn("dynamic_hoop", statei)
            self.assertIn("car", statei)
        # If no exception is raised, the test passes

    def test_17_set_ackermann_velocity_cmd(self):
        ackermann_cmd = [[1.0, 0.25] for _ in range(NUM_SERVERS)]  # speed, steering angle rate
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        states0 = rl_server.get_control_states(ENVIDS)
        states_cpy = states0.copy()
        for i in range(1000):
            rl_server.set_ackermann_velocity_cmd(ENVIDS,
                                                 CAR_MODEL_NAME,CAR_CHASSIS_LINK_NAME, ackermann_cmd)
            rl_server.run_N(ENVIDS, [1])
        rl_server.update_control_states(ENVIDS)
        state = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state0 = states_cpy[i]["car"]["chassis"][0]
            statei = state[i]["car"]["chassis"][0]
            self.assertIsNotNone(statei)
            self.assertEqual(len(statei), 13)
            self.assertNotEqual(state0[0], statei[0])  # x should have changed
            self.assertNotEqual(state0[1], statei[1])  # y should have changed
            self.assertNotEqual(state0[6], statei[6])  # yaw should have changed

        # If no exception is raised, the test passes

    def test_18_set_joint_position_cmd(self):
        # Hinge hoop rotates around x axis
        joint_positions = [[0.5, 0.0, 0.0] for _ in range(NUM_SERVERS)]  # radians
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        state0 = rl_server.get_control_states(ENVIDS)
        state_cpy = state0.copy()
        for i in range(1000):
            rl_server.set_joint_position_cmd(ENVIDS,
                                             HOOP_MODEL_NAME, HOOP_ARM_LINK_NAME, joint_positions)
            rl_server.run_N(ENVIDS, [1])
        rl_server.update_control_states(ENVIDS)
        state = rl_server.get_control_states(ENVIDS)

        for i in range(NUM_SERVERS):
            state0 = state_cpy[i]["dynamic_hoop"]["arm_link"][0]
            statei = state[i]["dynamic_hoop"]["arm_link"][0]
            self.assertIsNotNone(statei)
            self.assertEqual(len(statei), 13)  # position and velocity
            self.assertNotEqual(state0[0], statei[0])  # position should have changed
            self.assertNotEqual(state0[6], statei[6])  # quaternion x should have changed
        # If no exception is raised, the test passes

    def test_19_set_angular_velocity_cmd(self):
        angular_velocity_cmd = [[0.0, 0.0, 1.0]  for _ in range(NUM_SERVERS)]  # rad/s
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        state0 = rl_server.get_control_states(ENVIDS)
        state_cpy = state0.copy()
        for i in range(1000):
            rl_server.set_angular_velocity_cmd(ENVIDS,
                                               HOOP_MODEL_NAME, HOOP_ARM_LINK_NAME, angular_velocity_cmd)
            rl_server.run_N(ENVIDS, [1])
        rl_server.update_control_states(ENVIDS)
        state = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state0 = state_cpy[i]["dynamic_hoop"]["arm_link"][0]
            statei = state[i]["dynamic_hoop"]["arm_link"][0]
            self.assertIsNotNone(statei)
            self.assertEqual(len(statei), 13)  # full pose
            self.assertNotEqual(state0[9], statei[9])  # angular velocity z should have changed

        # If no exception is raised, the test passes

    def test_20_set_velocity_cmd(self):
        velocity_cmd = [[1.0, 0.0, 0.0]  for _ in range(NUM_SERVERS)]  # m/s
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        state0 = rl_server.get_control_states(ENVIDS)
        state_cpy = state0.copy()
        for i in range(1000):
            rl_server.set_velocity_cmd(ENVIDS,
                                       CAR_MODEL_NAME, CAR_CHASSIS_LINK_NAME, velocity_cmd)
            rl_server.run_N(ENVIDS, [1])
        rl_server.update_control_states(ENVIDS)
        state = rl_server.get_control_states(ENVIDS)
        for i in range(NUM_SERVERS):
            state0 = state_cpy[i]["car"]["chassis"][0]
            statei = state[i]["car"]["chassis"][0]
            self.assertIsNotNone(statei)
            self.assertEqual(len(statei), 13)  # full pose
            self.assertNotEqual(state0[3], statei[3])  # velocity x should have changed

        # If no exception is raised, the test passes

    def test_21_runtime_performance(self):
        import time
        N = 10000
        start_time = time.time()
        rl_server.run_N(ENVIDS, [N])
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_step = total_time / N
        freq = 1.0 / avg_time_per_step
        print(f"Total time for {N} simulation steps: {total_time:.2f} seconds")
        print(f"Average time per simulation step over {N} steps: {avg_time_per_step:.6f} seconds")
        self.assertLess(avg_time_per_step, 0.001)  # Expect less than 1ms per step
        self.assertGreater(freq, 1000.0)  # Expect more than 1000 Hz
        # If no exception is raised, the test passes

    def test_22_respawn_model(self):
        new_pose = ([-5.0, -5.0, -5.0],[-1.0, -2.0, -3.0])
        rotmat = Rotation.from_euler('xyz', new_pose[1]).as_matrix()
        new_pos = [new_pose[0] for _ in range(NUM_SERVERS)]
        new_ori = [new_pose[1] for _ in range(NUM_SERVERS)]

        for i in range(1000):
            rl_server.set_velocity_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [[1.0, 1.0 ,1.0]])
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = states[ENVIDS[i]][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[3], 1.0,delta=0.1)
            self.assertAlmostEqual(state[4], 1.0,delta=0.1)
            self.assertAlmostEqual(state[5], 1.0,delta=0.1)

        rl_server.respawn_model(ENVIDS, UAV_MODEL_NAME, new_pos, new_ori)
        rl_server.run_N(ENVIDS, [10])
        rl_server.update_control_states(ENVIDS)
        expected_z_vel = 0.0
        states = rl_server.get_control_states(ENVIDS)
        for i, state in enumerate(states):
            state = states[ENVIDS[i]][UAV_MODEL_NAME[i]][UAV_BASE_LINK_NAME[i]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-new_pos[i][0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-new_pos[i][1], 0.0,delta=0.1)
            self.assertAlmostEqual(state[2]-new_pos[i][2], 0.0,delta=0.1)
            self.assertAlmostEqual(state[3], 0.0,delta=0.1)
            self.assertAlmostEqual(state[4], 0.0,delta=0.1)
            self.assertAlmostEqual(state[5], expected_z_vel,delta=0.1)
            rotmat_after = Rotation.from_quat(state[6:10], scalar_first=True).as_matrix()
            self.assertAlmostEqual(np.linalg.norm(rotmat - rotmat_after), 0.0, delta=0.1)

        # If no exception is raised, the test passes

    def test_23_test_ctbt(self):
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.run_N(ENVIDS, [10])  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            cmds = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmds.append(controller[e].calculate_thrust_moments(state, state_dot, desired_state))
            rl_server.set_ctbt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, cmds)
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
            self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)

        # set a time-constant, should'nt track this time
        rl_server.set_ctbt_rate_limiter_time_constants(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [10.0], [20.0], [[0,0,0,0]])
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.15)], [(0.0, 0.0, 0.0)])
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            cmds = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmds.append(controller[e].calculate_thrust_moments(state, state_dot, desired_state))
            rl_server.set_ctbt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, cmds)
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertNotAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertNotAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertNotAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

        # set 0 time-constant, should track this time
        rl_server.set_ctbt_rate_limiter_time_constants(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [0.0], [0.0], [[0,0,0,0]])
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.15)], [(0.0, 0.0, 0.0)])
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            cmds = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmds.append(controller[e].calculate_thrust_moments(state, state_dot, desired_state))
            rl_server.set_ctbt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, cmds)
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

    def test_24_test_srt(self):
        gain_map = grl.GAIN_MAP()
        param_map = grl.PARAMETER_MAP()
        geom_gains = gain_map["qdrone2"]["geometric_controller"]
        geom_params = param_map["qdrone2"]["geometric_controller"]
        make_controller = lambda :  grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
            geom_gains["kp_att"], geom_gains["kd_att"],
            geom_params.max_accel,
            geom_params.gravity_vec,
            geom_params.mass,
            geom_params.inertia)
        controller = [make_controller() for _ in range(NUM_SERVERS)]
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.2)], [(0.0, 0.0, 0.0)])
        rl_server.run_N(ENVIDS, [10])  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ktau = 1.0/68.0
        rotor_links = ["quadrotor/rotor_0", "quadrotor/rotor_1", "quadrotor/rotor_2", "quadrotor/rotor_3"]
        turning_dirs = [1, -1, -1, 1]

        srt_func = rl_server.get_thrust_moment_to_rotor_thrust_mapping_function(ENVIDS, UAV_MODEL_NAME,
                                                                               [rotor_links], [turning_dirs], [ktau])
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            srts = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmd = controller[e].calculate_thrust_moments(state, state_dot, desired_state)
                srts.append(srt_func[e](cmd))
            rl_server.set_srt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                  [rotor_links], [turning_dirs], srts, [ktau])
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
            self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)

        # set a time-constant, should'nt track this time
        rl_server.set_srt_rate_limiter_time_constants(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [10.0], [20.0], [[0,0,0,0]])
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.15)], [(0.0, 0.0, 0.0)])
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            srts = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmd = controller[e].calculate_thrust_moments(state, state_dot, desired_state)
                srts.append(srt_func[e](cmd))
            rl_server.set_srt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                  [rotor_links], [turning_dirs], srts, [ktau])
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertNotAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertNotAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertNotAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

        # set 0 time-constant, should track this time
        rl_server.set_srt_rate_limiter_time_constants(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [0.0], [0.0], [[0,0,0,0]])
        rl_server.reset_pos(ENVIDS, UAV_MODEL_NAME, [(0.0, 0.0, 0.15)], [(0.0, 0.0, 0.0)])
        for _ in range(5000):
            rl_server.update_control_states(ENVIDS)
            states = rl_server.get_control_states(ENVIDS)
            srts = []
            for e in range(NUM_SERVERS):
                state, state_dot = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]]
                cmd = controller[e].calculate_thrust_moments(state, state_dot, desired_state)
                srts.append(srt_func[e](cmd))
            rl_server.set_srt_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME,
                                  [rotor_links], [turning_dirs], srts, [ktau])
            rl_server.run_once(ENVIDS)
        rl_server.update_control_states(ENVIDS)
        states = rl_server.get_control_states(ENVIDS)
        for e in range(NUM_SERVERS):
            state = states[e][UAV_MODEL_NAME[e]][UAV_BASE_LINK_NAME[e]][0]
            self.assertIsNotNone(state)
            self.assertEqual(len(state), 13)
            self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
            self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
            self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

    def test_25_test_ctbr(self):
        # just check if cmds can be applied, no ctbr controller implemented as of now
        for i in range(500):
            if (i%2 == 0):
                rl_server.set_ctbr_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [[10,0,0,0]])
            else:
                rl_server.set_ctbr_cmd(ENVIDS, UAV_MODEL_NAME, UAV_BASE_LINK_NAME, [[10,0,0,0]], [[1,2,3]], [[4,5,6]])
            rl_server.run_once(ENVIDS)

if __name__ == '__main__':
    unittest.main(verbosity=2)
