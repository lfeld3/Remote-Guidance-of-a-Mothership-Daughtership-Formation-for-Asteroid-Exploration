import configparser
import math
import numpy as np
import os

from src.Spacecraft import Spacecraft
from src.asteroid import Asteroid

class Simulation:
    """Class containing all parameters pertaining to the simulation setup

    Args:
        focal_length: focal length of the camera in millimeters
        camera_width: image width in pixels
        camera_height: image height in pixels
        pixel_pitch: camera pixel pitch in pixels/millimeter
        num_spacecraft: number of spacecraft in the simulation in total
        model: string for what meas model is used for daughtership state estimation in the BLS. Currently only supports image_range
        dt_MPC_multiplier: number of time steps between prediction horizon epochs in the MPC
        mothership_index: index in list of spacecraft that corresponds to the mothership
    """

    def __init__(self, asteroid, spacecraft, config_file, monte_carlo_s_0):
        config = configparser.ConfigParser()
        config.read('input/' + config_file)

        self.focal_length = int(config.get('Simulation','f'))  # value representing the focal length of the camera
        self.camera_width = int(config.get('Simulation','width'))  # value representing width of the camera in pixels
        self.camera_height = int(config.get('Simulation', 'height'))  # value representing height of the camera in pixels
        self.pixel_pitch = float(config.get('Simulation', 'pixel_pitch'))
        self.num_spacecraft = len(spacecraft)
        self.model = config.get('Simulation','model')  # string containing the measurement model for the daughterships
        self.dt_MPC_multiplier = float(config.get('Simulation', 'dt_MPC_multiplier'))

        self.mothership_index = 0
        for i in range(self.num_spacecraft):
            if self.num_spacecraft > 1:
                if spacecraft[i].designation == 'mothership':
                    self.mothership_index = i
            else:
                if spacecraft[i].designation == 'mothership':
                    self.mothership_index = i





def main():
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    spacecraft = Spacecraft(asteroid, config_file, None, 0)
    simulation = Simulation(asteroid, spacecraft, config_file, None)
    print(simulation.cost_function_to_use)
    help(Simulation)


if __name__ == '__main__':
    main()
