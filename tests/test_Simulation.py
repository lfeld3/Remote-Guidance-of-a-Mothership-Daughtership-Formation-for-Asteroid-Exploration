import numpy as np
import sys, os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.asteroid import Asteroid
from src.simulation import Simulation
from src.Spacecraft import Spacecraft

def test_Simulation_class():
    config_file = 'config_for_testing.ini'
    asteroid = Asteroid(config_file)
    spacecraft = Spacecraft(asteroid, config_file, None, 0)
    simulation = Simulation(asteroid, spacecraft, config_file, None)
    actual_focal_length = simulation.focal_length
    desired_focal_length = 12
    np.testing.assert_allclose(actual_focal_length, desired_focal_length, rtol=1e-4)
