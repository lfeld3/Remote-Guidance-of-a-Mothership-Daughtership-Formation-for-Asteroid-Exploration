import numpy as np
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.asteroid import Asteroid


def test_Asteroid_class():
    config_file = 'config_for_testing.ini'
    asteroid = Asteroid(config_file)
    actual_accel, actual_sa = asteroid.newpolygrav_vec(np.array([1,2,3]))
    desired_accel = np.array([[-2.67602833e-07],[-5.34235395e-07],[-8.03000719e-07]])/1000
    desired_sa = -2.8644621397067027e-16

    np.testing.assert_allclose(actual_accel, desired_accel, rtol=1e-4)
    np.testing.assert_allclose(actual_sa, desired_sa, rtol=1e-4)
