# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import warnings
warnings.filterwarnings('ignore')
import gzdrl as grl 
import unittest
import numpy as np
from scipy.spatial.transform import Rotation 
unittest.TestLoader.sortTestMethodsUsing = None

rl_server = grl.DRLServer("_0", "world_sdcs.sdf",
                                  ["quadrotor"],
                                   True)
class TestSensorsDRLServer(unittest.TestCase):

    def test01_get_images(self):
        cmaera_sensors = rl_server.camera_sensor_names()
        for sensor in cmaera_sensors:
            img = rl_server.get_sensor_image(sensor)
            # self.assertEqual(img.dtype, np.uint8)
            self.assertNotEqual(np.abs(img.abs).sum(), 0)
