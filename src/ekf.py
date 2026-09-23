import math
import sys

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from src.Spacecraft import Spacecraft
from src.asteroid import Asteroid
from src.simulation import Simulation
from scipy.integrate import odeint

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True, linewidth=100000)

class EKF:
    """Class contains all the methods for spacecraft state propagation, measurements, and the EKF itself"""

    def __init__(self, time_elapsed):
        self.time_elapsed = time_elapsed
        self.seen = np.empty(0)


    def features_seen(self, simulation, asteroid, spacecraft):
        """Method checks which features on the asteroid are within the camera field of view (FOV)
           and on the "correct" side of the asteroid (done via dot products)

        Args:
            simulation: instance of the Simulation class
            asteroid: instance of the Asteroid class
            spacecraft: instance of the Spacecraft class

        Returns:
            seen: [3 x ?]: matrix of all the feature positions in the camera fov, km
        """

        features = np.copy(asteroid.feature_verts)
        surface_normal = np.copy(asteroid.facet_normals[asteroid.feature_indices, :])
        num = features.shape[0]
        angle_to_rotate = asteroid.asteroid_ang_vel * self.time_elapsed * spacecraft.dt
        c = math.cos(angle_to_rotate)
        s = math.sin(angle_to_rotate)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        features_inertial = features @ R.T
        r_sun = spacecraft.r_sun[:, self.time_elapsed]
        r_sun_mat = np.repeat(r_sun[:, np.newaxis], num, axis=1).T
        r_sun_norm = np.linalg.norm(r_sun)
        surface_normal_inertial = surface_normal @ R.T

        fov = spacecraft.fov

        s_vis = (np.array([0, 0, 0]) - spacecraft.s[0:3])  # use estimated camera attitude once it is implemented!
        s_vis_norm = np.linalg.norm(s_vis)

        s_vis_mat = np.repeat(s_vis[:, np.newaxis], num, axis=1).T
        s_mat = np.repeat(spacecraft.s[0:3, np.newaxis], num, axis=1).T
        meas = features_inertial - s_mat

        features_inertial_norm = np.linalg.norm(features_inertial, axis=1)
        print('s_vis')
        print(s_vis)
        #print(meas)
        print('meas mag')
        print(np.linalg.norm(meas, axis=1))
        ang_wrt_sc = np.sum(meas * s_vis_mat, axis=1) / np.linalg.norm(meas, axis=1) / s_vis_norm
        print(ang_wrt_sc)
        ang_wrt_feat = np.sum(features_inertial * s_vis_mat, axis=1) / features_inertial_norm / s_vis_norm
        print(ang_wrt_feat)
        ang_wrt_sun = np.sum(surface_normal_inertial * r_sun_mat, axis=1) / r_sun_norm

        feature_vec1 = [i for i, v in enumerate(ang_wrt_sc) if v >= math.cos(math.radians(fov)/2)]
        feature_vec2 = [i for i, v in enumerate(ang_wrt_feat) if v <= 0]
        feature_vec3 = [i for i, v in enumerate(ang_wrt_sun) if v <= 0]
        feature_vec = np.intersect1d(feature_vec1, np.intersect1d(feature_vec2, feature_vec3).astype(int))
        print(feature_vec1)
        print(feature_vec2)
        print(feature_vec3)

        if feature_vec.shape[0] == 0:
            self.seen = np.empty(0)
        else:
            self.seen = features_inertial[feature_vec, :]

        return self.seen

    def pixel_line_dy(self, simulation, spacecraft, asteroid):
        """Method computes the measurements for use in the EKF measurement update

        Args:
            simulation: instance of the Simulation class
            spacecraft: instnace of the Spacecraft class

        Returns:
            dy: [a]: vector containing the measurement differentials (a being number of seen features)
        """

        v = spacecraft.v_cov
        features_inertial = np.copy(self.seen)

        num = features_inertial.shape[0]
        f = simulation.focal_length
        f_x = f / simulation.pixel_pitch
        f_y = f / simulation.pixel_pitch
        w = simulation.camera_width
        h = simulation.camera_height
        K_camera = np.array([[f_x, 0, w/2], [0, f_y, h/2]])

        # true state vectors/arrays
        s = np.copy(spacecraft.s[0:6])
        s_mat = np.repeat(s[0:3, np.newaxis], num, axis=1).T

        # estimated state vectors/arrays
        shat = np.copy(spacecraft.shat[0:6])
        shat_mat = np.repeat(shat[0:3, np.newaxis], num, axis=1).T

        # get estimated rotation matrix
        R_hat = -shat[0:3] / np.linalg.norm(shat[0:3])  # radial
        C_hat = np.cross(shat[0:3], shat[3:6])  # cross track
        C_hat /= np.linalg.norm(C_hat)
        I_hat = np.cross(C_hat, R_hat)  # in track
        I_hat /= np.linalg.norm(I_hat)

        rotation_camera_to_inertial = np.vstack([I_hat, C_hat, R_hat])

        meas = features_inertial - s_mat
        camera_observed = np.dot(meas, rotation_camera_to_inertial.T)
        dividend = np.tile(camera_observed[:, 2].reshape([-1, 1]), (1, 3))
        camera_focal_plane_observed = np.divide(camera_observed, dividend).T
        noises = (np.diag(v ** 0.5) @ np.random.normal(loc=0, scale=1, size=(2, num))).reshape([num * len(v)], order='F')
        pixel_line_observed = np.dot(K_camera, camera_focal_plane_observed).reshape([num * len(v)], order='F') + noises

        meas_hat = features_inertial - shat_mat
        camera_computed = np.dot(meas_hat, rotation_camera_to_inertial.T)
        dividend = np.tile(camera_computed[:, 2].reshape([-1, 1]), (1, 3))
        camera_focal_plane_computed = np.divide(camera_computed, dividend).T

        pixel_line_computed = np.dot(K_camera, camera_focal_plane_computed).reshape([num * len(v)], order='F')

        self.dy = pixel_line_observed - pixel_line_computed
        print('residuals')
        print(self.dy)
        #print('test')


    def pixel_line_H(self, simulation, asteroid, spacecraft):
        """Method computes the H for use in the EKF measurement update

        Args:
            simulation: instance of the Simulation class
            asteroid: instance of the Asteroid class
            spacecraft: instnace of the Spacecraft class

        Returns:
            H: [a x 7] matrix of the observation model within the EKF
        """
        shat = np.copy(spacecraft.shat[0:6])
        num = self.seen.shape[0]
        H_temp = np.zeros([len(spacecraft.v_cov) * num, spacecraft.num_states])

        v = np.array(spacecraft.v_cov)
        features_inertial = np.copy(self.seen)

        num = features_inertial.shape[0]
        f = simulation.focal_length
        f_x = f / simulation.pixel_pitch
        f_y = f / simulation.pixel_pitch
        w = simulation.camera_width
        h = simulation.camera_height
        K_camera = np.array([[f_x, 0, w / 2], [0, f_y, h / 2]])

        # estimated state vectors/arrays
        shat_mat = np.repeat(shat[0:3, np.newaxis], num, axis=1).T

        # get estimated rotation matrix
        R_hat = -shat[0:3] / np.linalg.norm(shat[0:3])  # radial
        C_hat = np.cross(shat[0:3], shat[3:6])  # cross track
        C_hat /= np.linalg.norm(C_hat)
        I_hat = np.cross(C_hat, R_hat)  # in track
        I_hat /= np.linalg.norm(I_hat)

        rotation_camera_to_inertial = np.vstack([I_hat, C_hat, R_hat])
        #print(rotation_camera_to_inertial)
        # H MAY NEED THE PARTIALS WRT STATE VECTOR OF THE ROTATION MATRIX!!!
        for i in range(num):
            meas_RIC = rotation_camera_to_inertial @ (features_inertial[i, :].reshape(3) - shat[0:3])
            H_temp[2 * i:2 * (i + 1), 0:3] = np.array([[f_x / meas_RIC[2], 0, -f_x * meas_RIC[0] / meas_RIC[2] ** 2], [0, f_y / meas_RIC[2],-f_y * meas_RIC[1] /meas_RIC[2] ** 2]]) @ -rotation_camera_to_inertial
        #print(H_temp)
        self.H = H_temp


    def meas_update(self, spacecraft):
        """Method performs the EKF measurement update

        Args:
            simulation: instance of the Simulation class
            asteroid: instance of the Asteroid class
            spacecraft: instance of the Spacecraft class

        Returns:
            self.shat: [7] updated vector for the estimated state (km, km/s, ln(kg))
            Pnew: [7 x 7] matrix for the updated covariance matrix (km^2, km^2/s^2, ln(kg)^2)

        """

        n = spacecraft.P.shape[0]
        m = self.H.shape[0]
        v = spacecraft.v_cov
        R = np.diag(np.repeat(v, m/len(v)))
        K = np.matmul(spacecraft.P, np.matmul(self.H.T, np.linalg.inv(np.matmul(self.H, np.matmul(spacecraft.P, self.H.T)) + R)))
        spacecraft.shat += np.matmul(K, self.dy)
        spacecraft.P = np.matmul(np.matmul((np.eye(n) - np.matmul(K, self.H)), spacecraft.P), (np.eye(n) - np.matmul(K, self.H)).T) + np.matmul(K, np.matmul(R, K.T))

    def run_EKF(self, simulation, asteroid, spacecraft):
        """Method performs the FULL EKF! check individual functions for more info.

        Args:
            simulation: instance of the Simulation class
            asteroid: instance of the Asteroid class
            spacecraft: instance of the Spacecraft class

        Returns:
            s: [7] vector for the updated true state (m, m/s, ln(kg), unitless)
            shat: [7] vector for the updated estimated state (m, m/s, ln(kg), unitless)
            Pnew: [7 x 7] matrix for the updated covariance matrix (m^2, m^2/s^2, ln(kg)^2, unitless)
            sa: scalar solid angle for the true state (steradians)
            sahat: scalar solid angle for the estimated state (steradians)
            seen: [3 x ?]: matrix of all the feature positions in the camera fov, km
        """

        if all(element != 0 for element in spacecraft.v_cov) != 0:
            if spacecraft.designation == 'mothership':
                self.seen = self.features_seen(simulation, asteroid, spacecraft)
            else:
                self.seen = []

            if not np.shape(self.seen)[0] == 0 and self.time_elapsed % 1 == 0:
                self.pixel_line_H(simulation, asteroid, spacecraft)
                self.pixel_line_dy(simulation, spacecraft, asteroid)
                self.meas_update(spacecraft)
                self.pixel_line_dy(simulation, spacecraft, asteroid)
            else:
                spacecraft.P = np.copy(spacecraft.P)
        return self.seen


def main():
    print('unit test')

    # run EKF for a state and estimate
    s = np.array([1000, 600, 800, 0.1, 0.2, 0.3, math.log(1400)])
    shat = np.array([950, 550, 850, 0.125, 0.18, 0.33, math.log(1400)])
    P = np.diag([2500, 2500, 2500, 10**-1, 10**-1, 10**-1, 0])
    u = np.array([0.1, 0.2, 0.05])
    time_elapsed = 1
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    spacecraft = Spacecraft(asteroid, config_file, None)
    simulation = Simulation(asteroid, spacecraft, config_file, None)
    ekf = EKF(s, u, shat, P, time_elapsed, None, None, None)

    print('starting P')
    print(ekf.P)
    print('starting state')
    print(ekf.s)
    print('starting estimated state')
    print(ekf.shat)

    ekf.s, ekf.shat, ekf.Pnew, ekf.sa, ekf.sahat, ekf.seen, dt_updated, _ = ekf.run_EKF(simulation, asteroid, spacecraft)

    print('post time update P')
    print(ekf.Pnew)
    print('post time update state')
    print(ekf.s)
    print('post time update estimated state')
    print(ekf.shat)

    print('dt_updated')
    print(dt_updated)

    # generate surface features and plot them
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(asteroid.verts[:, 0], asteroid.verts[:, 1], asteroid.verts[:, 2], color='blue', zorder=1, alpha=0.1, marker='.')
    ax.scatter(asteroid.feature_verts[1:, 0], asteroid.feature_verts[1:, 1], asteroid.feature_verts[1:, 2], color='red', zorder=2, alpha=0.8, marker='o')
    ax.scatter(asteroid.feature_verts[0, 0], asteroid.feature_verts[0, 1], asteroid.feature_verts[0, 2], color='green', zorder=5, alpha=1, marker='s')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend(['Asteroid Vertices', 'Asteroid Features', 'Target Landing Site'], loc='center left')


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.scatter(asteroid.feature_verts[:,0], asteroid.feature_verts[:,1], asteroid.feature_verts[:,2], zorder=1, alpha=0.5, marker='.')

    # set up state and camera fov
    featuresseen = ekf.features_seen(simulation, asteroid, spacecraft)
    if featuresseen.shape[0] > 0:
        ax1.scatter(featuresseen[:, 0], featuresseen[:, 1], featuresseen[:, 2], color='red', zorder=2, alpha=1)

    ax1.scatter(s[0]/1000, s[1]/1000, s[2]/1000, color='green', zorder=3)
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    #  cProfile.run('main()')
    main()