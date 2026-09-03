# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import warnings
warnings.filterwarnings('ignore')
import gzdrl as grl 
import unittest
import numpy as np
from scipy.spatial.transform import Rotation 
unittest.TestLoader.sortTestMethodsUsing = None

rl_server = grl.DRLServer("_0",
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
        controller = make_controller() 
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      controller)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))
        rl_server.run_N(10)  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rl_server.control_with_rotor_velocity("quadrotor",
                                                    "quadrotor/base_link",
                                                    desired_state,
                                                    5000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
        self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)
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
            
        python_controller = SimplePythonController()
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      python_controller)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rl_server.control_with_rotor_velocity("quadrotor",
                                                    "quadrotor/base_link",
                                                    desired_state,
                                                    5000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
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
        controller =make_controller() 
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      controller)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))
        rl_server.update_control_states()
        state =  rl_server.control_states["quadrotor"]["quadrotor/base_link"]
        self.assertAlmostEqual(np.linalg.norm(state[3:6]), 0.0,  0.2)
        rl_server.run_N(10)  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rl_server.control_with_wrench("quadrotor",
                                                    "quadrotor/base_link",
                                                    desired_state,
                                                    10000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
        self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)
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
            
        python_controller = SimplePythonController()
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      python_controller)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))  
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rl_server.control_with_wrench("quadrotor",
                                                    "quadrotor/base_link",
                                                    desired_state,
                                                    5000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        # No state checks since controller is dummy

    def test_05_reset_pose(self):
        new_pose = ([5.0, 5.0, 5.0],[1.0, 2.0, 3.0])
        rotmat = Rotation.from_euler('xyz', new_pose[1]).as_matrix()
        for i in range(1000):
            rl_server.set_velocity_cmd("quadrotor", "quadrotor/base_link", [1.0, 1.0 ,1.0])
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertAlmostEqual(state[3], 1.0,delta=0.1)
        self.assertAlmostEqual(state[4], 1.0,delta=0.1)
        self.assertAlmostEqual(state[5], 1.0,delta=0.1)
        rl_server.reset_pos("quadrotor", *new_pose)
        rl_server.run_N(10)
        expected_z_vel = -9.82*13*1e-3
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state[0]-new_pose[0][0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-new_pose[0][1], 0.0,delta=0.1)
        self.assertAlmostEqual(state[2]-new_pose[0][2], 0.0,delta=0.1)
        self.assertAlmostEqual(state[3], 0.0,delta=0.1)
        self.assertAlmostEqual(state[4], 0.0,delta=0.1)
        self.assertAlmostEqual(state[5], expected_z_vel,delta=0.1)
        rotmat_after = Rotation.from_quat(state[6:10], scalar_first=True).as_matrix()
        self.assertAlmostEqual(np.linalg.norm(rotmat - rotmat_after), 0.0, delta=0.1)
        # If no exception is raised, the test passes

    def test_06_set_trajectory_trace(self):
        serv_config = grl.DRLServerConfig()
        serv_config.trajectory_viz = True
        rl_server.set_trajectory_trace("quadrotor", "quadrotor/base_link", serv_config)
        rl_server.run_N(5)
        # If no exception is raised, the test passes
    
    # def test_set_headless_mode(self):
    #     rl_server.set_headless_render_mode(True)
    #     rl_server.run_N(5)
    #     rl_server.set_headless_render_mode(False)
    #     rl_server.run_N(5)
    #     # If no exception is raised, the test passes
    

    def test_07_set_wrench(self):
        wrench = ((0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
        rl_server.set_wrench("quadrotor", "quadrotor/base_link", *wrench)
        rl_server.run_N(10)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.0), (0.0,0.0,0.0))
        rl_server.run_N(10)
        # If no exception is raised, the test passes

    def test_08_set_rotor_velocity(self):
        rotor_velocities = [1000.0, 1000.0, 1000.0, 1000.0]
        rl_server.set_rotor_velocity_cmd("quadrotor", 
                                              "quadrotor/base_link",
                                                rotor_velocities)
        rl_server.run_N(10)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.0), (0.0,0.0,0.0))
        rl_server.run_N(10)
        # If no exception is raised, the test passes

    def test_09_contact_sensor_data(self):
        rl_server.request_contact_data("quadrotor")
        rl_server.run_N(10)
        sensor_data = rl_server.get_contacts("quadrotor")                     
        self.assertIsNotNone(sensor_data)
        # If no exception is raised, the test passes

    def test_10_velocity_mapping_function(self):
        mapping_func = rl_server.get_thrust_moment_to_rotor_velocity_mapping_function("quadrotor")
        self.assertIsNotNone(mapping_func)
        rl_server.set_rotor_velocity_cmd("quadrotor", 
                                              "quadrotor/base_link",
                                              mapping_func([0.0, 0.0, 0.0, 0.0]))
        rl_server.run_N(10)
        # If no exception is raised, the test passes
    
    def test_11_allocation_matrix(self):
        allocation_matrix = rl_server.get_rotor_thrust_allocation_matrix("quadrotor")
        self.assertIsNotNone(allocation_matrix)
        self.assertEqual(len(allocation_matrix), 4)  # 4 rotors
        self.assertEqual(len(allocation_matrix[0]), 4)  # 4 inputs (thrust + 3 moments)
        # If no exception is raised, the test passes
        # Test inverse
        inv_allocation_matrix = rl_server.get_inverse_rotor_thrust_allocation_matrix("quadrotor")   
        self.assertIsNotNone(inv_allocation_matrix)
        self.assertEqual(len(inv_allocation_matrix), 4)  # 4 inputs
        self.assertEqual(len(inv_allocation_matrix[0]), 4)  # 4
        self.assertAlmostEqual((inv_allocation_matrix@allocation_matrix).ravel().sum(), 
                               np.eye((4)).ravel().sum(),delta=0.1)
    
    def test_12_get_rotor_parameters(self):
        rotor_params = rl_server.get_rotor_parameters("quadrotor")
        self.assertIsNotNone(rotor_params)
        self.assertEqual(len(rotor_params.thrust_constant_quadratic_params), 3)
        self.assertEqual(len(rotor_params.torque_constant_quadratic_params), 3)
        self.assertIsInstance(rotor_params.ground_effect_constant, float)
        self.assertIsInstance(rotor_params.time_constant_up, float)
        self.assertIsInstance(rotor_params.time_constant_down, float)
        self.assertIsInstance(rotor_params.rotor_drag_coefficient, float)
        self.assertIsInstance(rotor_params.rotor_inertia, float)
        self.assertIsInstance(rotor_params.rolling_moment_coefficient, float)

    def test_13_set_rotor_parameters(self):
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
        original_params =  rl_server.get_rotor_parameters("quadrotor")
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
        controller = make_controller() 
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      controller)
        # verify we can track before change
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)

        new_params = grl.RotorParameters()
        new_params.max_rot_velocity = 15.0
        new_params.thrust_constant_quadratic_params = [0e-6, 1e-7, 1e-8]
        new_params.torque_constant_quadratic_params = [0e-7, 2e-8, 2e-9]
        new_params.ground_effect_constant = 0.05
        new_params.time_constant_up = 1.5
        new_params.time_constant_down = 2.5
        new_params.rotor_drag_coefficient = 0.02
        new_params.rotor_inertia = 1.5e-5
        new_params.rolling_moment_coefficient = 0.015
        rl_server.set_rotor_parameters("quadrotor", new_params)
        updated_params = rl_server.get_rotor_parameters("quadrotor")
        self.assertEqual(updated_params.max_rot_velocity, 15.0)
        self.assertEqual((updated_params.thrust_constant_quadratic_params - np.array([0e-6, 1e-7, 1e-8])).sum(), 0.0)
        self.assertEqual((updated_params.torque_constant_quadratic_params - np.array([0e-7, 2e-8, 2e-9])).sum(), 0.0)
        self.assertEqual(updated_params.ground_effect_constant, 0.05)
        self.assertEqual(updated_params.time_constant_up, 1.5)
        self.assertEqual(updated_params.time_constant_down, 2.5)
        self.assertEqual(updated_params.rotor_drag_coefficient, 0.02)
        self.assertEqual(updated_params.rotor_inertia, 1.5e-5)
        self.assertEqual(updated_params.rolling_moment_coefficient, 0.015)

        # verify after change we can't track
        # verify parameters has changed when respawn_model is called
        rl_server.respawn_model("quadrotor", [0,0, 0.2], [0,0,0])
        rl_server.run_N(10)
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
        # restore original params
        rl_server.set_rotor_parameters("quadrotor", original_params)
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
        # If no exception is raised, the test passes
    
    def test_14_set_and_get_inertia(self):
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
        original_inertia = rl_server.get_inertia("quadrotor", "quadrotor/base_link")
        test_inertia = 100*np.array([[0.02, 0.0, 0.0],
                                 [0.0, 0.03, 0.0],
                                 [0.0, 0.0, 0.04]])
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
        controller = make_controller() 
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      controller)
        rl_server.set_inertia("quadrotor", "quadrotor/base_link", test_inertia)
        retrieved_inertia = rl_server.get_inertia("quadrotor", "quadrotor/base_link")
        # verify that inertia has not changed when rest_pos is called
        rl_server.reset_pos("quadrotor", [0,0, 0.2], [0,0,0])
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)
        # verify inertia has changed when respawn_model is called
        rl_server.respawn_model("quadrotor", [0,0, 0.2], [0,0,0])
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
        self.assertTrue(np.allclose(test_inertia, retrieved_inertia))
        # set back original param
        rl_server.set_inertia("quadrotor", "quadrotor/base_link", original_inertia)
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
        # If no exception is raised, the test passes
    
    def test_15_set_and_get_mass(self):
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
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
        controller = make_controller() 
        rl_server.set_controller("quadrotor", 
                                      "quadrotor/base_link", 
                                      controller)
        original_mass = rl_server.get_mass("quadrotor", "quadrotor/base_link")
        test_mass = 3.5  # kg
        rl_server.set_mass("quadrotor", "quadrotor/base_link", test_mass)
        retrieved_mass = rl_server.get_mass("quadrotor", "quadrotor/base_link")
        # verify that mass has not changed when rest_pos is called
        rl_server.reset_pos("quadrotor", [0,0, 0.2], [0,0,0])
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertAlmostEqual(np.linalg.norm(np.array([1,1,1.5]) - state[:3]), 0.0, delta=0.3)
        # verify mass has changed when respawn_model is called
        rl_server.respawn_model("quadrotor", [0,0, 0.2], [0,0,0])
        rl_server.control_with_rotor_velocity("quadrotor", "quadrotor/base_link", [1,1,1.5, 0,0,0,1,0,0,0,0,0,0], 3000)
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertNotAlmostEqual(np.linalg.norm(np.array([1,1,1.5,]) - state[:3]), 0.0, delta=0.3)
        self.assertAlmostEqual(test_mass, retrieved_mass)
        # If no exception is raised, the test passes
        # reste mass to original
        rl_server.set_mass("quadrotor", "quadrotor/base_link", original_mass)
        rl_server.respawn_model("quadrotor", [0,0,0.05], [0,0,0,])
    
    def test_16_control_states(self):
        rl_server.run_N(10)
        rl_server.update_control_states()
        states = rl_server.control_states
        self.assertIn("quadrotor", states)
        self.assertIn("quadrotor/base_link", states["quadrotor"])
        state = states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        # check for other models
        self.assertIn("dynamic_hoop", states)
        self.assertIn("car", states)
        # If no exception is raised, the test passes
    
    def test_17_set_ackermann_velocity_cmd(self):
        ackermann_cmd = [1.0, 0.25]  # speed, steering angle rate
        rl_server.run_N(10)
        rl_server.update_control_states()
        state0 = rl_server.control_states["car"]["chassis"][0].copy()
        for i in range(1000):
            rl_server.set_ackermann_velocity_cmd("car", "chassis", ackermann_cmd)
            rl_server.run_N(1)
        rl_server.update_control_states()
        state = rl_server.control_states["car"]["chassis"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertNotEqual(state0[0], state[0])  # x should have changed
        self.assertNotEqual(state0[1], state[1])  # y should have changed   
        self.assertNotEqual(state0[6], state[6])  # yaw should have changed

        # If no exception is raised, the test passes
    
    def test_18_set_joint_position_cmd(self):
        # Hinge hoop rotates around x axis
        joint_positions = [0.5, 0.0, 0.0]  # radians
        rl_server.run_N(10)
        rl_server.update_control_states()
        state0 = rl_server.control_states["dynamic_hoop"]["arm_link"][1].copy()
        for i in range(1000):
            rl_server.set_joint_position_cmd("dynamic_hoop", "arm_link", joint_positions)
            rl_server.run_N(1)
        rl_server.update_control_states()
        state = rl_server.control_states["dynamic_hoop"]["arm_link"][1]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)  # position and velocity
        self.assertNotEqual(state0[0], state[0])  # position should have changed
        self.assertNotEqual(state0[6], state[6])  # quaternion x should have changed

        # If no exception is raised, the test passes
    def test_19_set_angular_velocity_cmd(self):
        angular_velocity_cmd = [0.0, 0.0, 1.0]  # rad/s
        rl_server.run_N(10)
        rl_server.update_control_states()
        state0 = rl_server.control_states["dynamic_hoop"]["arm_link"][0].copy()
        for i in range(1000):
            rl_server.set_angular_velocity_cmd("dynamic_hoop", "arm_link", angular_velocity_cmd)
            rl_server.run_N(1)
        rl_server.update_control_states()
        state = rl_server.control_states["dynamic_hoop"]["arm_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)  # full pose
        self.assertNotEqual(state0[9], state[9])  # angular velocity z should have changed

        # If no exception is raised, the test passes
    
    def test_20_set_velocity_cmd(self):
        velocity_cmd = [1.0, 0.0, 0.0]  # m/s
        rl_server.run_N(10)
        rl_server.update_control_states()
        state0 = rl_server.control_states["car"]["chassis"][0].copy()
        for i in range(1000):
            rl_server.set_velocity_cmd("car", "chassis", velocity_cmd)
            rl_server.run_N(1)
        rl_server.update_control_states()
        state = rl_server.control_states["car"]["chassis"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)  # full pose
        self.assertNotEqual(state0[3], state[3])  # velocity x should have changed

        # If no exception is raised, the test passes

    def test_21_runtime_performance(self):
        import time
        N = 10000
        start_time = time.time()
        rl_server.run_N(N)
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
        for i in range(1000):
            rl_server.set_velocity_cmd("quadrotor", "quadrotor/base_link", [1.0, 1.0 ,1.0])
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertAlmostEqual(state[3], 1.0,delta=0.1)
        self.assertAlmostEqual(state[4], 1.0,delta=0.1)
        self.assertAlmostEqual(state[5], 1.0,delta=0.1)
        rl_server.respawn_model("quadrotor", *new_pose)
        rl_server.run_N(10)
        rl_server.update_control_states()
        expected_z_vel = 0
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state[0]-new_pose[0][0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-new_pose[0][1], 0.0,delta=0.1)
        self.assertAlmostEqual(state[2]-new_pose[0][2], 0.0,delta=0.1)
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
        controller = make_controller() 
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))
        rl_server.run_N(10)  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            rl_server.set_ctbt_cmd("quadrotor", "quadrotor/base_link", cmd )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
        self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)

        # set a time-constant, should'nt track this time
        rl_server.set_ctbt_rate_limiter_time_constants("quadrotor", "quadrotor/base_link", 10.0, 20.0,  [0,0,0,0])
        rl_server.reset_pos("quadrotor", [0,0,0.15], [0,0,0])
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            rl_server.set_ctbt_cmd("quadrotor", "quadrotor/base_link", cmd )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertNotAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertNotAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertNotAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

        # set 0 time-constant, should track this time
        rl_server.set_ctbt_rate_limiter_time_constants("quadrotor", "quadrotor/base_link", 0.0, 0.0,  [0,0,0,0])
        rl_server.reset_pos("quadrotor", [0,0,0.15], [0,0,0])
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            rl_server.set_ctbt_cmd("quadrotor", "quadrotor/base_link", cmd )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
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
        controller = make_controller() 
        rl_server.reset_pos("quadrotor", (0.0,0.0,0.2), (0.0,0.0,0.0))
        rl_server.run_N(10)  # Let it stabilize a bit
        desired_state = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ktau = 1.0/68.0
        rotor_links=["quadrotor/rotor_0", "quadrotor/rotor_1", "quadrotor/rotor_2", "quadrotor/rotor_3"]
        turning_dirs = [1, -1, -1, 1]

        srt_func = rl_server.get_thrust_moment_to_rotor_thrust_mapping_function("quadrotor", rotor_links,
                                                                                turning_dirs, ktau
                                                                                )
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            srt = srt_func(cmd)
            rl_server.set_srt_cmd("quadrotor", "quadrotor/base_link", rotor_links, turning_dirs, srt, ktau )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)
        self.assertAlmostEqual(state[6]-desired_state[6], 0.0,delta=0.1)
        # set a time-constant, should'nt track this time
        rl_server.set_srt_rate_limiter_time_constants("quadrotor", "quadrotor/base_link", 10.0, 20.0,  [0,0,0,0])
        rl_server.reset_pos("quadrotor", [0,0,0.15], [0,0,0])
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            srt = srt_func(cmd)
            rl_server.set_srt_cmd("quadrotor", "quadrotor/base_link", rotor_links, turning_dirs, srt, ktau )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertNotAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertNotAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertNotAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)

        # set 0 time-constant, should track this time
        rl_server.set_srt_rate_limiter_time_constants("quadrotor", "quadrotor/base_link", 0.0, 0.0,  [0,0,0,0])
        rl_server.reset_pos("quadrotor", [0,0,0.15], [0,0,0])
        for i in range(5000):
            rl_server.update_control_states()
            state, state_dot = rl_server.control_states["quadrotor"]["quadrotor/base_link"]
            cmd = controller.calculate_thrust_moments(state, state_dot, desired_state)
            srt = srt_func(cmd)
            rl_server.set_srt_cmd("quadrotor", "quadrotor/base_link", rotor_links, turning_dirs, srt, ktau )
            rl_server.run_once()
        rl_server.update_control_states()
        state = rl_server.control_states["quadrotor"]["quadrotor/base_link"][0]
        self.assertIsNotNone(state)
        self.assertEqual(len(state), 13)
        self.assertAlmostEqual(state[0]-desired_state[0], 0.0,delta=0.1)
        self.assertAlmostEqual(state[1]-desired_state[1], 0.0,delta=0.1)
        self.assertAlmostEqual((state[2]-desired_state[2])**2, 0.0, delta=0.1)


    def test_25_test_ctbr(self):
        # just check if cmds can be applied, no ctbr controller implemented as of now
        for i in range(500):
            if (i%2 == 0):
                rl_server.set_ctbr_cmd("quadrotor", "quadrotor/base_link", [10,0,0,0])
            else:
                rl_server.set_ctbr_cmd("quadrotor", "quadrotor/base_link", [10,0,0,0], [1,2,3],[4,5,6])
            rl_server.run_once()
                
if __name__ == '__main__':
    unittest.main(verbosity=2)
