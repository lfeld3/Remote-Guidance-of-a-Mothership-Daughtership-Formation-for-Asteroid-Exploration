import configparser
import math

import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from src.MinDisSegmentTime_Vectorized import MinDisSegmentPoint_Vectorized
from src.asteroid import Asteroid
from scipy.integrate import odeint, solve_ivp
from interpax import interp1d


class Spacecraft:
    """Class containing the state, estimated state, parameter, and meas model info for a single spacecraft

    Args:
        thrust_min: min thrust allowed in the simulation, Newtons
        thrust_max: max thrust allowed in the simulation, Newtons
        wet_mass: wet mass of the lander, kg
        dry_mass: dry mass of the lander, kg
        num_subproblems: number of sub-problems/time steps, unitless
        dt: time for one time step, seconds
        num_states: dimension of state vector, unitless
        num_controls: dimension of control vector, unitless
        isp: specific impulse of lander thrusters, seconds
        x_i: initial true x position in simulation, kilometers
        y_i: initial true y position in simulation, kilometers
        z_i: initial true z position in simulation, kilometers
        vx_i: initial true x velocity in simulation, kilometers/sec
        vy_i: initial true y velocity in simulation, kilometers/sec
        vz_i: initial true z velocity in simulation, kilometers/sec
        dxhat_i: initial estimated x position in simulation, kilometers
        dyhat_i: initial estimated y position in simulation, kilometers
        dzhat_i: initial estimated z position in simulation, kilometers
        dvxhat_i: initial estimated x velocity in simulation, kilometers/sec
        dvyhat_i: initial estimated y velocity in simulation, kilometers/sec
        dvzhat_i: initial estimated z velocity in simulation, kilometers/sec
        p_x: variance of x in simulation, meters^2
        p_y: variance of y in simulation, meters^2
        p_z: variance of z in simulation, meters^2
        p_vx: variance of vx in simulation, meters^2/sec^2
        p_vy: variance of vy in simulation, meters^2/sec^2
        p_vz: variance of vz in simulation, meters^2/sec^2
        p_q: variance of q in simulation, ln(kg)^2, should always be zero
        a_r: largest unmodeled radial accel in RTN frame, kilometers/s^2
        a_t: largest unmodeled radial accel in RTN frame, kilometers/s^2
        a_n: largest unmodeled radial accel in RTN frame, kilometers/s^2
        v_cov: variance of measurement noise in simulation, various units
        fov: full field of view of camera in sim assuming circular camera, degrees
        landing_loc_index_overwrite: if overwrite_lli is True, this OVERWRITES landinglocindex. Otherwise, use landinglocindex, boolean
        exhaust_velocity: scalar for the exhaust velocity of the spacecraft thrusters
        fov: scalar for the half angle field of view of the spacecraft, degrees
        P: assembly of p_[] values
        S: surface area of spacecraft, meters^2
        C_R: coefficient of reflection
        gates_sigma_1: Gates model sigma 1 - fixed magnitude error standard deviation in millimeters / sec
        gates_sigma_2: Gates model sigma 2 - proportional magnitude error standard deviation as a percentage
        gates_sigma_3: Gates model sigma 3 - fixed pointing error standard deviation in millimeters / sec
        gates_sigma_4: Gates model sigma 4 - proportional pointing error standard deviation in milliradians
        orbit: string for type of orbit used, halo or terminator
        s: current spacecraft state
        shat: current spacecraft estimated state
        s_hist: true spacecraft state history [7, duration]
        shat_hist: setimated spacecraft state history [7, duration]
        delta_v_num_in_batch: integer for what local number in the BLS window a deltaV is at
        delta_v_hist: [3, duration] history of commanded deltaVs (without error) for the spacecraft
        delta_v_error_hist: [3, duration] history of deltaV erroneous components for the spacecraft
        delta_v_hat_hist: [3, duration] history of estimated deltaVs for the spacecraft
        shat_post_BLS_hist: etimated spacecraft state history AFTER RUNNING BLS [7, duration]
        dmin_hist: [duration] list of min distances to nearest ref traj segment
        vmin_hist: [duration] list of velocity diff from nearest ref traj segment's velocity
        true_accel_hist: [3, duration] polyhedral gravity accel history over time
        pred_accel_hist: [3, duration] 2-body/low res gravity accel history over time
        s_star: [6] current nominal state in the BLS
        s_observed: [6, duration] states used for computing observations in meas models
        s_observed[:, 0] = np.copy(s[0:6])
        cov_hist: [7, 7, duration] state covariance history over time
        delta_v_cov_hist: [3, 3, duration] deltaV covariance history over time
    """

    def __init__(self, asteroid, config_file, monte_carlo_s_0, spacecraft_num):
        config = configparser.ConfigParser()
        config.read('input/' + config_file)
        self.thrust_min = int(config.get('Simulation', 'Tmin').split(',')[spacecraft_num])  # Newtons, minimum thrust
        self.thrust_max = int(config.get('Simulation', 'Tmax').split(',')[spacecraft_num])  # Newtons, maximum thrust
        self.surface_area = float(config.get('Simulation', 'S').split(',')[spacecraft_num])  # m^2, surface area for illumination
        self.wet_mass = int(config.get('Simulation', 'm_wet').split(',')[spacecraft_num])  # kg, wet mass of the s/c
        self.dry_mass = int(config.get('Simulation', 'm_dry').split(',')[spacecraft_num])  # kg, dry mass of the s/c
        self.num_states = int(config.get('Simulation', 'numstates'))  # number of states
        self.num_controls = int(config.get('Simulation', 'm'))  # number of controls
        designation_vec = config.get('Simulation', 'designations')  # string, designations for the spacecraft
        self.num_spacecraft = len(designation_vec.split(','))  # scalar, number of spacecraft in the simulation
        self.designation = designation_vec.split(',')[spacecraft_num]  # designation of the specific spacecraft for reference trajectory selection
        self.isp = int(config.get('Simulation', 'isp').split(',')[spacecraft_num])  # s, specific impulse of lander thrusters
        self.exhaust_velocity = self.isp * 9.805 / 1000  # km/s, exhaust velocity
        self.ref_traj_location = config.get('Simulation','folder')
        self.dt = int(config.get('Simulation', 'dt'))  # time step duration (sec)
        self.dt_ref = int(config.get('Simulation', 'dt_ref'))  # ref time step duration (sec)
        self.duration = int(config.get('Simulation', 'N'))  # number of time steps in the sim
        self.ref_duration = int(config.get('Simulation', 'N_ref'))  # number of time steps of ref traj data to load
        self.N_batch = int(config.get('Simulation', 'N_batch'))  # number of range meas per daughterhsip batch
        self.gates_sigma_1_factor = float(config.get('Simulation', 'gates_1_factor'))  # multiple of gates_sigma_1 that is the min deltaV applied
        self.k = int(config.get('Simulation', 'k').split(',')[spacecraft_num])  # integer value for what order the BSpline interpolation should be
        self.meas_times = np.linspace(0, (self.N_batch - 1) * self.dt, self.N_batch)

        self.ref_traj_setting = int(config.get('Simulation', 'ref_traj_setting').split(',')[spacecraft_num])  # reference trajectory setting
        self.ics_gen_setting = int(config.get('Simulation', 'ics_gen_setting').split(',')[spacecraft_num])  # initial condition setting
        self.r_0 = float(config.get('Simulation', 'r_0').split(',')[spacecraft_num])  # km, radius for generated ic orbit
        self.ta_0 = float(config.get('Simulation', 'ta_0').split(',')[spacecraft_num])  # rad, true anomaly for generated ic orbit
        self.t_0 = int(config.get('Simulation', 't_0').split(',')[spacecraft_num])  # integer value for what epoch in sun ephemeris data to base ics around
        self.orbit = config.get('Simulation', 'orbit').split(',')[spacecraft_num]  # orbit type of spacecraft

        # sun ephemeris
        self.r_sun = np.zeros([3, self.ref_duration])
        self.v_sun = np.zeros([3, self.ref_duration])
        file = open('data/trajdata/' + self.ref_traj_location + '/' + asteroid.asteroid_name + '.txt')
        lines = file.readlines()
        epoch1 = lines[2].split(',')[0][12:25].split(':')
        time1 = float(epoch1[0]) * 24 * 60 + float(epoch1[1]) * 60 + float(epoch1[2])
        epoch2 = lines[3].split(',')[0][12:25].split(':')
        time2 = float(epoch2[0]) * 24 * 60 + float(epoch2[1]) * 60 + float(epoch2[2])
        dt_sun = time2 - time1
        for line in range(self.ref_duration):  # len(lines[2:])
            self.r_sun[:, line] = np.array(
                [float(lines[line + 2].split(',')[1]), float(lines[line + 2].split(',')[2]),
                 float(lines[line + 2].split(',')[3])])
            self.v_sun[:, line] = np.array(
                [float(lines[line + 2].split(',')[4]), float(lines[line + 2].split(',')[5]),
                 float(lines[line + 2].split(',')[6])])

        file.close()

        spl_sun_x = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.r_sun[0, :], self.k)
        spl_sun_y = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.r_sun[1, :], self.k)
        spl_sun_z = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.r_sun[2, :], self.k)
        spl_sun_vx = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.v_sun[0, :], self.k)
        spl_sun_vy = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.v_sun[1, :], self.k)
        spl_sun_vz = make_interp_spline(np.arange(0, dt_sun * self.ref_duration, dt_sun), self.v_sun[2, :], self.k)

        self.spline_sun_x = BSpline(spl_sun_x.t, spl_sun_x.c, spl_sun_x.k)
        self.spline_sun_y = BSpline(spl_sun_y.t, spl_sun_y.c, spl_sun_y.k)
        self.spline_sun_z = BSpline(spl_sun_z.t, spl_sun_z.c, spl_sun_z.k)
        self.spline_sun_vx = BSpline(spl_sun_vx.t, spl_sun_vx.c, spl_sun_vx.k)
        self.spline_sun_vy = BSpline(spl_sun_vy.t, spl_sun_vy.c, spl_sun_vy.k)
        self.spline_sun_vz = BSpline(spl_sun_vz.t, spl_sun_vz.c, spl_sun_vz.k)

        self.sun_times = jnp.arange(0.0, dt_sun * self.ref_duration, dt_sun)

        # earth-moon barycenter ephemeris
        self.r_EMB = np.zeros([3, self.ref_duration])
        file = open('data/trajdata/' + self.ref_traj_location + '/earth-barycenter-traj.txt')
        lines = file.readlines()
        epoch1 = lines[2].split(',')[0][12:25].split(':')
        time1 = float(epoch1[0]) * 24 * 60 + float(epoch1[1]) * 60 + float(epoch1[2])
        epoch2 = lines[3].split(',')[0][12:25].split(':')
        time2 = float(epoch2[0]) * 24 * 60 + float(epoch2[1]) * 60 + float(epoch2[2])
        dt_EMB = time2 - time1
        for line in range(self.ref_duration):  # len(lines[2:])
            self.r_EMB[:, line] = np.array(
                [float(lines[line + 2].split(',')[1]), float(lines[line + 2].split(',')[2]),
                 float(lines[line + 2].split(',')[3])])

        file.close()

        spl_EMB_x = make_interp_spline(np.arange(0, dt_EMB * self.ref_duration, dt_EMB), self.r_EMB[0, :], self.k)
        spl_EMB_y = make_interp_spline(np.arange(0, dt_EMB * self.ref_duration, dt_EMB), self.r_EMB[1, :], self.k)
        spl_EMB_z = make_interp_spline(np.arange(0, dt_EMB * self.ref_duration, dt_EMB), self.r_EMB[2, :], self.k)

        self.spline_EMB_x = BSpline(spl_EMB_x.t, spl_EMB_x.c, spl_EMB_x.k)
        self.spline_EMB_y = BSpline(spl_EMB_y.t, spl_EMB_y.c, spl_EMB_y.k)
        self.spline_EMB_z = BSpline(spl_EMB_z.t, spl_EMB_z.c, spl_EMB_z.k)

        self.EMB_times = jnp.arange(0.0, dt_EMB * self.ref_duration, dt_EMB)

        # jupiter ephemeris
        self.r_jup = np.zeros([3, self.ref_duration])
        file = open('data/trajdata/' + self.ref_traj_location + '/jupiter-barycenter-traj.txt')
        lines = file.readlines()
        epoch1 = lines[2].split(',')[0][12:25].split(':')
        time1 = float(epoch1[0]) * 24 * 60 + float(epoch1[1]) * 60 + float(epoch1[2])
        epoch2 = lines[3].split(',')[0][12:25].split(':')
        time2 = float(epoch2[0]) * 24 * 60 + float(epoch2[1]) * 60 + float(epoch2[2])
        dt_jup = time2 - time1
        for line in range(self.ref_duration):  # len(lines[2:])
            self.r_jup[:, line] = np.array(
                [float(lines[line + 2].split(',')[1]), float(lines[line + 2].split(',')[2]),
                 float(lines[line + 2].split(',')[3])])

        file.close()

        spl_jup_x = make_interp_spline(np.arange(0, dt_jup * self.ref_duration, dt_jup), self.r_jup[0, :], self.k)
        spl_jup_y = make_interp_spline(np.arange(0, dt_jup * self.ref_duration, dt_jup), self.r_jup[1, :], self.k)
        spl_jup_z = make_interp_spline(np.arange(0, dt_jup * self.ref_duration, dt_jup), self.r_jup[2, :], self.k)

        self.spline_jup_x = BSpline(spl_jup_x.t, spl_jup_x.c, spl_jup_x.k)
        self.spline_jup_y = BSpline(spl_jup_y.t, spl_jup_y.c, spl_jup_y.k)
        self.spline_jup_z = BSpline(spl_jup_z.t, spl_jup_z.c, spl_jup_z.k)

        self.jup_times = jnp.arange(0.0, dt_jup * self.ref_duration, dt_jup)

        self.C_R = float(
            config.get('Simulation', 'C_R').split(',')[spacecraft_num])  # reflectivity coefficient

        if self.ref_traj_setting == 1:
            print('test')
            # extract initial conditions from text files at data/trajdata/self.ref_traj_location/[designation].txt
            file = open('data/trajdata/' + self.ref_traj_location + '/' + self.designation + '.txt')
            lines = file.readlines()

            # get time step and duration of reference
            epoch1 = lines[2].split(',')[0][12:25].split(':')
            time1 = float(epoch1[0])*24*60 + float(epoch1[1])*60 + float(epoch1[2])
            epoch2 = lines[3].split(',')[0][12:25].split(':')
            time2 = float(epoch2[0])*24*60 + float(epoch2[1])*60 + float(epoch2[2])
            self.dt = time2 - time1
            print(self.dt)
            #self.duration = len(lines)-2


            # extract state history
            self.s_d = np.zeros([self.ref_duration, self.num_states])  # len(lines[2:])
            for line in range(self.ref_duration):  # len(lines[2:])
                self.s_d[line] = np.array([float(lines[line + 2].split(',')[1]), float(lines[line + 2].split(',')[2]), float(lines[line + 2].split(',')[3]), float(lines[line + 2].split(',')[4]), float(lines[line + 2].split(',')[5]), float(lines[line + 2].split(',')[6]), math.log(self.wet_mass)])

            file.close()
            self.s_d = self.s_d.T

            self.x_0 = self.s_d[0, 0]  # km, initial true x position in sim
            self.y_0 = self.s_d[1, 0]  # km, initial true y position in sim
            self.z_0 = self.s_d[2, 0]  # km, initial true z position in sim
            self.vx_0 = self.s_d[3, 0]  # km/s, initial true x velocity in sim
            self.vy_0 = self.s_d[4, 0]  # km/s, initial true y velocity in sim
            self.vz_0 = self.s_d[5, 0]  # km/s, initial true z velocity in sim

        else:
            print('Generating reference trajectory! This will take a couple of minutes.')
            self.s_d = np.zeros([self.num_states, self.ref_duration])  # * self.dt
            if self.ics_gen_setting == 1:

                self.x_0 = float(config.get('Simulation', 'x_0').split(',')[spacecraft_num])
                self.y_0 = float(config.get('Simulation', 'y_0').split(',')[spacecraft_num])
                self.z_0 = float(config.get('Simulation', 'z_0').split(',')[spacecraft_num])
                self.vx_0 = float(config.get('Simulation', 'vx_0').split(',')[spacecraft_num])
                self.vy_0 = float(config.get('Simulation', 'vy_0').split(',')[spacecraft_num])
                self.vz_0 = float(config.get('Simulation', 'vz_0').split(',')[spacecraft_num])

                self.s_d[0, 0] = self.x_0  # km, initial true x position in sim
                self.s_d[1, 0] = self.y_0  # km, initial true y position in sim
                self.s_d[2, 0] = self.z_0  # km, initial true z position in sim
                self.s_d[3, 0] = self.vx_0  # km/s, initial true x velocity in sim
                self.s_d[4, 0] = self.vy_0  # km/s, initial true y velocity in sim
                self.s_d[5, 0] = self.vz_0  # km/s, initial true z velocity in sim
                self.s_d[6, 0] = np.log(self.wet_mass)

            else:
                #s_d_0_frozen, offset_frozen, period_frozen = self.frozen_ic(asteroid, self.t_0, self.r_0, self.ta_0, math.pi/2, -math.pi/2, self.dt, self.ref_duration)
                self.s_d[:, 0], self.offset, self.orbit_period = self.terminator_ic(asteroid, self.t_0, self.r_0, self.ta_0)
                self.x_0 = self.s_d[0, 0]
                self.y_0 = self.s_d[1, 0]
                self.z_0 = self.s_d[2, 0]
                self.vx_0 = self.s_d[3, 0]
                self.vy_0 = self.s_d[4, 0]
                self.vz_0 = self.s_d[5, 0]
                print('generated initial conditions')
                print(self.s_d[:, 0])
                #print(s_d_0_frozen)
                t_frozen = np.arange(0, self.dt*self.ref_duration, self.dt)
                #print(t_frozen)


            A = np.array([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])

            ref_num_pts = int(np.ceil(self.orbit_period / self.dt))
            t = np.linspace(0, self.orbit_period, ref_num_pts)

            # get terminator x, y coords
            self.s_d = self.circle_points_6d(self.r_0, ref_num_pts, self.ta_0).T

            # get terminator z coords
            self.local_offset_d = np.zeros(ref_num_pts)
            for j in range(ref_num_pts):
                n_sun = np.array([self.spline_sun_x(t[j]), self.spline_sun_y(t[j]), self.spline_sun_z(t[j])])
                n_sun /= np.linalg.norm(n_sun)
                R = self.get_rotation(n_sun)
                self.local_offset_d[j] = self.get_local_offset(asteroid, t[j], R.T @ self.s_d[0:3, j])

            self.s_d[2, :] = self.local_offset_d

            # get terminator velocities
            print(self.local_offset_d)
            self.s_d[3:6, :] = self.positions_to_velocities(self.s_d[0:3, :].T, self.dt).T
            self.s_d[2, :] = np.zeros(ref_num_pts)

        self.max_segment_length = np.linalg.norm(np.array([self.vx_0, self.vy_0, self.vz_0])) * self.dt  # THIS IS A GROSS APPROXIMATION!

        print(self.s_d)

        self.xhat_0 = self.x_0 + float(config.get('Simulation', 'dxhat_0').split(',')[
                                           spacecraft_num])  # km, initial estimated x position in sim
        self.yhat_0 = self.y_0 + float(config.get('Simulation', 'dyhat_0').split(',')[
                                           spacecraft_num])  # km, initial estimated y position in sim
        self.zhat_0 = self.z_0 + float(config.get('Simulation', 'dzhat_0').split(',')[
                                           spacecraft_num])  # km, initial estimated z position in sim
        self.vxhat_0 = self.vx_0 + float(config.get('Simulation', 'dvxhat_0').split(',')[
                                             spacecraft_num])  # km/s, initial estimated x velocity in sim
        self.vyhat_0 = self.vy_0 + float(config.get('Simulation', 'dvyhat_0').split(',')[
                                             spacecraft_num])  # km/s, initial estimated y velocity in sim
        self.vzhat_0 = self.vz_0 + float(config.get('Simulation', 'dvzhat_0').split(',')[
                                             spacecraft_num])  # km/s, initial estimated z velocity in sim

        # covariance
        self.p_x = float(config.get('Simulation', 'Px').split(',')[spacecraft_num])  # km^2, Px in sim
        self.p_y = float(config.get('Simulation', 'Py').split(',')[spacecraft_num])  # km^2, Py in sim
        self.p_z = float(config.get('Simulation', 'Pz').split(',')[spacecraft_num])  # km^2, Pz in sim

        self.p_vx = float(config.get('Simulation', 'Pvx').split(',')[spacecraft_num])  # km^2/s^2, Pvx in sim
        self.p_vy = float(config.get('Simulation', 'Pvy').split(',')[spacecraft_num])  # km^2/s^2, Pvy in sim
        self.p_vz = float(config.get('Simulation', 'Pvz').split(',')[spacecraft_num])  # km^2/s^2, Pvz in sim

        self.p_q = float(config.get('Simulation', 'Pq').split(',')[spacecraft_num])  # kg^2, Pq in sim

        self.P = np.diag([self.p_x, self.p_y, self.p_z, self.p_vx, self.p_vy, self.p_vz, self.p_q])

        # process noise spectral density
        self.a_r = float(config.get('Simulation', 'a_r').split(',')[spacecraft_num])  # km/s^2, a_r in sim
        self.a_t = float(config.get('Simulation', 'a_t').split(',')[spacecraft_num])  # km/s^2, a_t in sim
        self.a_n = float(config.get('Simulation', 'a_n').split(',')[spacecraft_num])  # km/s^2, a_n in sim

        self.q_r = self.dt * self.a_r ** 2
        self.q_t = self.dt * self.a_t ** 2
        self.q_n = self.dt * self.a_n ** 2
        self.Q = np.diag([self.q_r, self.q_t, self.q_n])  # in RTN frame. NEED to rotate into inertial to use!

        self.v_start = int(config.get('Simulation', 'v_start').split(',')[spacecraft_num])
        self.v_end = int(config.get('Simulation', 'v_end').split(',')[spacecraft_num])

        self.v_cov = np.array(config.get('Simulation', 'v').split(',')[self.v_start:self.v_end]).astype(float)  # km^2, v in sim
        print(self.v_cov)
        print(self.v_cov.dtype)

        self.gates_sigma_1 = float(config.get('Gates', 's1').split(',')[spacecraft_num])  # Gates model sigma 1
        self.gates_sigma_2 = float(config.get('Gates', 's2').split(',')[spacecraft_num])  # Gates model sigma 2
        self.gates_sigma_3 = float(config.get('Gates', 's3').split(',')[spacecraft_num])  # Gates model sigma 3
        self.gates_sigma_4 = float(config.get('Gates', 's4').split(',')[spacecraft_num])  # Gates model sigma 4
        width = int(config.get('Simulation','width'))
        f_x = int(config.get('Simulation','f')) / float(config.get('Simulation', 'pixel_pitch'))
        self.fov = 2 * np.arctan2(width, 2*f_x) * 180. / math.pi  # degrees, fov of camera in sim
        print("HERE IS THE COMPUTED FIELD OF VIEW")
        print(self.fov)

        self.horizon_steps = int(config.get('Simulation', 'horizon').split(',')[spacecraft_num])  # number of timesteps in prediction horizon

        self.s_0 = np.array([self.x_0, self.y_0, self.z_0, self.vx_0, self.vy_0, self.vz_0, np.log(self.wet_mass)])
        self.shat_0 = np.array([self.xhat_0, self.yhat_0, self.zhat_0, self.vxhat_0, self.vyhat_0, self.vzhat_0, np.log(self.wet_mass)])

        # state and control histories
        self.s = np.copy(self.s_0)
        self.shat = np.copy(self.shat_0)
        self.s_hist = np.zeros([len(self.s_0), self.duration])
        self.s_hist[:, 0] = self.s_0
        self.shat_hist = np.zeros([len(self.s_0), self.duration])
        self.shat_hist[:, 0] = self.shat_0
        self.delta_v = np.array([0, 0, 0])
        self.delta_v_num_in_batch = -1*np.ones(self.duration)
        self.delta_v_hist = np.zeros([len(self.delta_v), self.duration])
        self.delta_v_error_hist = np.zeros([len(self.delta_v), self.duration])
        self.delta_v_hat_hist = np.zeros([len(self.delta_v), self.duration])
        self.shat_post_BLS_hist = np.zeros([len(self.s_0), self.duration])
        self.shat_post_BLS_hist[:, 0] = self.shat_0
        self.dmin_hist = np.zeros(self.duration)
        self.vmin_hist = np.zeros(self.duration)
        self.true_accel_hist = np.zeros([3, self.duration])
        self.pred_accel_hist = np.zeros([3, self.duration])
        self.s_star = np.copy(self.shat_0)
        self.s_observed = np.zeros([len(self.s_0)-1, self.duration])
        self.s_observed[:, 0] = np.copy(self.s[0:6])
        self.cov_hist = np.zeros([self.num_states, self.num_states, self.duration])
        self.cov_hist[:, :, 0] = np.copy(self.P)
        self.delta_v_cov_hist = np.zeros([self.num_controls - 1, self.num_controls - 1, self.duration])

        # if meas model takes images
        self.feature_array = []

        # Assemble p1, p2 for minimum distance line segment code (n points by m dimensions)
        self.p1 = np.transpose(self.s_d[0:3, 0:-1])
        self.p2 = np.transpose(self.s_d[0:3, 1:])
        self.vvec = np.transpose(self.s_d[3:6, 1:])

        # orbital element history of reference
        self.a_hist, self.e_hist, self.i_hist, self.raan_hist, self.aop_hist = self.compute_elements_over_time(asteroid.mu_grav_param)

    def accel_at(self, asteroid, t, pos_vec):
        rest_of_state = np.array([0.0, 0.0, 0.0, np.log(self.wet_mass)])
        state = np.hstack((pos_vec, rest_of_state))

        a_grav, _ = asteroid.pt_mass_grav(pos_vec)
        r_sun_to_apophis = np.array([self.spline_sun_x(t), self.spline_sun_y(t), self.spline_sun_z(t)])
        r_sun_to_EMB = np.array([self.spline_EMB_x(t), self.spline_EMB_y(t), self.spline_EMB_z(t)])
        r_sun_to_jup = np.array([self.spline_jup_x(t), self.spline_jup_y(t), self.spline_jup_z(t)])

        a_SRP = self.solar_rad_pressure(state, np.copy(r_sun_to_apophis))[3:6]
        a_3rd_sun = self.third_body_effect(np.copy(state), gm=1.327e11, r_third_body=np.copy(-r_sun_to_apophis))[3:6]
        a_3rd_EMB = self.third_body_effect(np.copy(state), gm=3.986e5 + 4.902e3, r_third_body=np.copy(-r_sun_to_apophis) + np.copy(r_sun_to_EMB))[3:6]
        a_3rd_jupiter = self.third_body_effect(np.copy(state), gm=1.26686534e8, r_third_body=np.copy(-r_sun_to_apophis) + np.copy(r_sun_to_jup))[3:6]

        return a_SRP + a_grav.reshape(-1) + a_3rd_sun + a_3rd_EMB + a_3rd_jupiter

    def positions_to_velocities(self, pos, dt):
        """
        pos : (N, 3) array of 3D positions
        dt  : constant timestep

        returns (N, 3) array of 3D velocities using 4th‑order accurate finite differences
        """

        N = pos.shape[0]
        vel = np.zeros_like(pos)

        # forward differences
        vel[0] = (-25 * pos[0] + 48 * pos[1] - 36 * pos[2] + 16 * pos[3] - 3 * pos[4]) / (12 * dt)
        vel[1] = (-3 * pos[0] - 10 * pos[1] + 18 * pos[2] - 6 * pos[3] + pos[4]) / (12 * dt)

        # central differences
        vel[2:-2] = (-pos[4:] + 8 * pos[3:-1] - 8 * pos[1:-3] + pos[0:-4]) / (12 * dt)

        # backward differences
        vel[-2] = (-pos[-5] + 6 * pos[-4] - 18 * pos[-3] + 10 * pos[-2] + 3 * pos[-1]) / (12 * dt)
        vel[-1] = (3 * pos[-5] - 16 * pos[-4] + 36 * pos[-3] - 48 * pos[-2] + 25 * pos[-1]) / (12 * dt)

        return vel

    def get_local_offset(self, asteroid, t, r):
        """Method computes local offset from a terminator orbit defined by the sun->body vector at epoch t that has the halo orbit in plane

        Args:
            asteroid: instance of the asteroid class
            t [1]: integer scalar for what epoch in the Sun ephemeris data to take as the initial epoch for orbit IC
            r [3]: km, vector for current position of spacecraft in inertial frame

        Returns:
            offset: [1] z-component offset to add to reference trajectory in terminator frame to put into local halo orbit
        """

        # get vector from sun -> body
        n_sun = np.array([self.spline_sun_x(t), self.spline_sun_y(t), self.spline_sun_z(t)])
        n_sun /= np.linalg.norm(n_sun)

        pos_plane = np.copy(r)
        # print('pos_plane')
        # print(pos_plane)
        # get offset to go from terminator to halo orbit

        def f(s):
            pos = pos_plane + s * n_sun
            a = self.accel_at(asteroid, t, pos)
            # print('a np')
            # print(a)
            # print(s)
            # print(np.dot(a, n_sun))
            return np.dot(a, n_sun)/np.linalg.norm(a)

        # 1D bracket + bisection on s
        s_lo, s_hi = -0.5, 0.5  # km
        f_lo, f_hi = f(s_lo), f(s_hi)
        for _ in range(100):
            s_mid = 0.5 * (s_lo + s_hi)
            f_mid = f(s_mid)
            if f_lo * f_mid <= 0:
                s_hi, f_hi = s_mid, f_mid
            else:
                s_lo, f_lo = s_mid, f_mid
            if abs(f_mid) < 1e-16:
                break

        offset = s_mid * n_sun
        pos = pos_plane + offset

        # print(pos)
        # print(s_mid)
        R = self.get_rotation(n_sun)
        z_current = (R @ pos_plane)[2]
        z_equilibrium = z_current + s_mid

        return z_equilibrium

    def terminator_ic(self, asteroid, t, r, true_anomaly):
        """Method computes initial conditions (IC) for a terminator orbit defined by the sun->body vector at epoch t

        Args:
            asteroid: instance of the asteroid class
            t [1]: integer scalar for what epoch in the Sun ephemeris data to take as the initial epoch for orbit IC
            r [1]: km, scalar for what radius the circular orbit should have
            true_anomaly [1]: rad, scalar value [0, 2pi) for what true anomaly angle CCW relative to +z the spacecraft IC is

        Returns:
            ics: [7] vector of the ICs at epoch t
            offset: [3] inertial frame offset to add to the position to get into halo orbit (if in one)
            period: [1] scalar period of the orbit in seconds
        """

        # get vector from sun -> body
        n_sun = np.copy(self.r_sun[:, t])
        n_sun /= np.linalg.norm(n_sun)

        # define a vector p1 in the terminator plane
        if abs(n_sun[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        p1 = np.cross(n_sun, ref)
        p1 /= np.linalg.norm(p1)

        # define a vector in plane, perpendicular to p1
        p1_perp = np.cross(n_sun, p1)
        p1_perp /= np.linalg.norm(p1_perp)

        # get position of spacecraft in terminator plane
        pos_plane = r * (np.cos(true_anomaly) * p1 + np.sin(true_anomaly) * p1_perp)

        # get offset to go from terminator to halo orbit
        s = self.get_local_offset(asteroid, t, pos_plane)
        offset = s * n_sun
        pos = pos_plane + offset

        # radial unit vector in the terminator plane
        r_hat = pos / np.linalg.norm(pos)

        # tangential direction, perpendicular to r_hat, in the terminator plane
        vel_dir = np.cross(n_sun, r_hat)
        vel_dir /= np.linalg.norm(vel_dir)

        # net non-centrifugal acceleration at pos
        a_net = self.accel_at(asteroid, t, pos)

        # radial component
        a_rad = -np.dot(a_net, r_hat)
        r_norm = np.linalg.norm(pos)

        v_mag = np.sqrt(max(a_rad * r_norm, 0.0))
        vel = v_mag * vel_dir

        ics = np.hstack((pos, vel, np.log(self.wet_mass)))
        # print("OFFSET !!!!")
        # print(offset)
        # print(np.dot(n_sun, offset)/np.linalg.norm(offset)/np.linalg.norm(n_sun))
        period = 2 * math.pi * np.sqrt(np.linalg.norm(pos) ** 3 / asteroid.mu_grav_param)
        return ics, offset, period

    def frozen_ic(self, asteroid, t, r, true_anomaly, raan, aop, dt, duration):
        """Method computes initial conditions (IC) for a terminator orbit defined by the sun->body vector at epoch t

        Args:
            asteroid: instance of the asteroid class
            t [1]: integer scalar for what epoch in the Sun ephemeris data to take as the initial epoch for orbit IC
            r [1]: km, scalar for what radius the circular orbit should have
            true_anomaly [1]: rad, scalar value [0, 2pi) for what true anomaly angle CCW relative to +z the spacecraft IC is
            raan [1]: rad, raan in the SAM frame the orbit ellipse is
            aop [1]: rad, aop in the SAM frame the orbit ellipse is
            dt [1]: sec, time step size
            duration [1]: scalar, number of time steps in reference

        Returns:
            ics: [7] vector of the ICs at epoch t
            offset: [3] inertial frame offset to add to the position to get into halo orbit (if in one)
            period: [1] scalar period of the orbit in seconds
        """

        times = t + dt * np.arange(duration)

        r_sun_to_apophis = np.vstack([
            self.spline_sun_x(times),
            self.spline_sun_y(times),
            self.spline_sun_z(times)
        ]).T

        r_sun_sc = r_sun_to_apophis * 1000.0
        r_mag = np.linalg.norm(r_sun_sc, axis=1)
        r_AU = r_mag / 1.495978707e11

        mags = 4.56e-6 * self.C_R * self.surface_area / self.wet_mass * (1.0 / r_AU ** 2)
        a_SRP_mean = mags.mean() / 1000.0

        mu_sun = 1.327e11
        e_apophis = 0.1911492279663492
        a_apophis = 0.9223592206975018 * 1.496e8
        n_apophis = 1.112638115271892 * np.pi / 180 / 86400
        nu_dot = np.sqrt(mu_sun * a_apophis * (1 - e_apophis ** 2)) / (r_mag.mean() / 1000) ** 2
        a = r

        perturbation_angle = np.arctan2(3 * a_SRP_mean, 2 * n_apophis * a * nu_dot)
        e = np.cos(perturbation_angle)
        inc = math.pi/2  # always this for terminator plane solution

        # Distance
        p = a * (1 - e ** 2)
        r = p / (1 + e * np.cos(true_anomaly))

        # Perifocal position
        x_p = r * np.cos(true_anomaly)
        y_p = r * np.sin(true_anomaly)
        z_p = 0.0

        # Perifocal velocity
        h = np.sqrt(asteroid.mu_grav_param * p)
        vx_p = -asteroid.mu_grav_param / h * np.sin(true_anomaly)
        vy_p = asteroid.mu_grav_param / h * (e + np.cos(true_anomaly))
        vz_p = 0.0

        # Rotation matrices
        R3_w = np.array([
            [np.cos(aop), -np.sin(aop), 0],
            [np.sin(aop), np.cos(aop), 0],
            [0, 0, 1]
        ])

        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)]
        ])

        R3_O = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])

        # Full rotation: SAM = R3(raan) * R1(inc) * R3(aop) * perifocal
        R = R3_O @ R1_i @ R3_w

        r_SAM = R @ np.array([x_p, y_p, z_p])
        v_SAM = R @ np.array([vx_p, vy_p, vz_p])

        R_I2SAM = self.get_rotation_sam(np.array([self.r_sun[0, t], self.r_sun[1, t], self.r_sun[2, t], self.v_sun[0, t], self.v_sun[1, t], self.v_sun[2, t]]))
        n_sun = np.array([self.spline_sun_x(t), self.spline_sun_y(t), self.spline_sun_z(t)])
        n_sun /= np.linalg.norm(n_sun)
        R_I2T = self.get_rotation(n_sun)

        r_T = R_I2SAM.T @ r_SAM
        v_T = R_I2SAM.T @ v_SAM

        s = self.get_local_offset(asteroid, t, r_T)
        offset = s * n_sun
        pos = (r_T + offset)
        period = 2 * math.pi * np.sqrt(np.linalg.norm(pos) ** 3 / asteroid.mu_grav_param)
        ics = np.hstack((pos, v_T, np.log(self.wet_mass)))
        return ics, offset, period

    def get_rotation_sam(self, n_sun):
        """
        Compute rotation matrix from inertial frame to Sun Anti-Momentum (SAM) frame
        for a single 6-element vector.

        SAM frame:
            x_hat = normalized Sun->body position
            z_hat = -normalized( r × v )   (anti-momentum direction)
            y_hat = z_hat × x_hat

        Parameters
        ----------
        n_sun : array-like, shape (6,)
            [x, y, z, vx, vy, vz] of the body w.r.t. the Sun.

        Returns
        -------
        R : (3, 3) rotation matrix (inertial → SAM)
        """

        # Extract position and velocity
        r = np.asarray(n_sun[0:3])
        v = np.asarray(n_sun[3:6])

        # X-axis: Sun→body direction
        x = r / np.linalg.norm(r)

        # Z-axis: negative heliocentric angular momentum direction
        h = np.cross(r, v)
        z = -h / np.linalg.norm(h)

        # Y-axis: z × x
        y = np.cross(z, x)
        y /= np.linalg.norm(y)

        # Assemble rotation matrix
        R = np.vstack((x, y, z))

        return R

    def circle_points_6d(self, r, n, start_angle):
        """
        r: (1) scalar radius of the circle
        n: (1) number of points
        start_angle: (1) angle (radians), CCW relative to +Z axis,
                     defining where the first point lies on the circle.

        Returns:
            (n, 6) array of points [x, y, z, vx, vy, vz]
            lying in the local terminator frame.
        """

        # Uniform angles with phase shift
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + start_angle + np.pi

        # circle in XY-plane
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)

        # zero velocity components
        vx = np.zeros_like(theta)
        vy = np.zeros_like(theta)
        vz = np.zeros_like(theta)

        return np.vstack((x, y, z, vx, vy, vz)).T

    def get_rotation(self, n_sun):
        """
        n_sun: (N, 3) array of unit Sun->body vectors

        Returns: R (N, 3, 3) rotation matrices
        """

        update_R_shape = False

        n_sun = np.asarray(n_sun)
        if n_sun.ndim == 1:
            n_sun = n_sun.reshape(1, 3)
            update_R_shape = True

        z = n_sun

        # choose reference vector r for each row
        r = np.where(np.abs(z[:, 0:1]) < 0.9,
                     np.array([1.0, 0.0, 0.0]),
                     np.array([0.0, 1.0, 0.0]))

        # x = normalized(r × z)
        x = np.cross(r, z)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # y = z × x
        y = np.cross(z, x)

        # stack rotation matrices
        R = np.stack([x, y, z], axis=1)

        if update_R_shape:
            R = R.reshape(3, 3)

        return R

    def get_RTN_to_inertial(self, r, v):
        """
        r: (3,) position vector
        v: (3,) velocity vector

        Returns: R (3, 3) rotation matrix from RTN -> inertial
        """

        r = np.asarray(r)
        v = np.asarray(v)

        # R-axis: radial direction
        R_axis = r / np.linalg.norm(r)

        # N-axis: normal to orbital plane
        h = np.cross(r, v)
        N_axis = h / np.linalg.norm(h)

        # T-axis: along-track direction
        T_axis = np.cross(N_axis, R_axis)

        # Stack rotation matrix
        R = np.stack([R_axis, T_axis, N_axis], axis=0)

        return R

    def orbital_elements_from_rv(self, r, v, mu):
        """
        Computes the classical orbital elements from r, v in 3D, assumes r and v lie in terminator frame
        """
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)

        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)

        # semimajor axis
        energy = v_norm ** 2 / 2 - mu / r_norm
        a = -mu / (2 * energy)

        # eccentricity vector
        e_vec = np.cross(v, h) / mu - r / r_norm
        e = np.linalg.norm(e_vec)

        # inclination
        hx, hy, hz = h
        sin_i = np.linalg.norm(np.array([hx, hy]))
        i = np.arctan2(sin_i, hz)

        # node
        n = np.cross([0, 0, 1], h)
        n_norm = np.linalg.norm(n)

        # RAAN
        if n_norm < 1e-12:
            RAAN = 0.0
        else:
            RAAN = np.arctan2(n[1], n[0])

        # argument of periapsis (undefined if e=0 or n_norm=0)
        if e < 1e-12 or n_norm < 1e-12:
            argp = 0.0
        else:
            argp = np.arctan2(np.cross(n, e_vec)[2] / (n_norm * e), np.dot(n, e_vec) / (n_norm * e))

        return a, e, i, RAAN, argp

    def compute_elements_over_time(self, mu):
        """
        computes orbital elements in the Sun-terminator frame for spacecraft.s_d
        """
        N = self.s_d.shape[1]

        a_arr = np.zeros(N)
        e_arr = np.zeros(N)
        i_arr = np.zeros(N)
        RAAN_arr = np.zeros(N)
        argp_arr = np.zeros(N)

        for k in range(N):
            # shift into terminator frame
            s_d = self.s_d[:, k]
            r_shift = s_d[0:3] - np.array([0, 0, s_d[2]])
            v_shift = s_d[3:6]

            print(k)
            print(r_shift*1e3)
            print(v_shift*1e6)

            # orbital elements
            a, e, inc, RAAN, argp = self.orbital_elements_from_rv(r_shift, v_shift, mu)

            a_arr[k] = a
            e_arr[k] = e
            i_arr[k] = inc
            RAAN_arr[k] = RAAN
            argp_arr[k] = argp

        return a_arr, e_arr, i_arr, RAAN_arr, argp_arr

    def safe_norm_jax(self, x, eps=1e-20):
        return jnp.sqrt(jnp.sum(x ** 2) + eps)

    def ode_dynamics(self, state, t, A, asteroid, epoch_of_sim, gravity_model):
        time_since_beginning_of_sim = epoch_of_sim * self.dt + t

        # get polyhedral gravity acceleration
        angle_to_rotate = -asteroid.asteroid_ang_vel * time_since_beginning_of_sim
        c = math.cos(angle_to_rotate)
        s = math.sin(angle_to_rotate)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        state_body = R @ state[0:3].reshape([3, 1])
        if gravity_model == 'poly':
            accel, _ = asteroid.newpolygrav_vec(state_body[0:3])
        else:
            accel, _ = asteroid.pt_mass_grav(state_body[0:3])
        accel = accel.reshape([3, 1])
        accel_inertial = (R.T @ accel).reshape(3)
        C = np.array([0, 0, 0, accel_inertial[0], accel_inertial[1], accel_inertial[2], 0])

        # print(self.designation)
        # print('gravity')
        # print(C)
        # print(np.linalg.norm(C))

        # use cubic spline interpolation to determine coordinates of Sun, EMB, and Jupiter
        r_sun_to_apophis = np.array([self.spline_sun_x(time_since_beginning_of_sim), self.spline_sun_y(time_since_beginning_of_sim), self.spline_sun_z(time_since_beginning_of_sim)])
        r_sun_to_EMB = np.array([self.spline_EMB_x(time_since_beginning_of_sim), self.spline_EMB_y(time_since_beginning_of_sim), self.spline_EMB_z(time_since_beginning_of_sim)])
        r_sun_to_jup = np.array([self.spline_jup_x(time_since_beginning_of_sim), self.spline_jup_y(time_since_beginning_of_sim), self.spline_jup_z(time_since_beginning_of_sim)])

        # get accelerations for SRP and 3rd body effects
        C_SRP = self.solar_rad_pressure(np.copy(state), np.copy(r_sun_to_apophis))
        C_3rd_sun = self.third_body_effect(np.copy(state), gm=1.327e11, r_third_body=np.copy(-r_sun_to_apophis))
        C_3rd_earth_moon = self.third_body_effect(np.copy(state), gm=3.986e5+4.902e3, r_third_body=np.copy(-r_sun_to_apophis)+np.copy(r_sun_to_EMB))
        C_3rd_jupiter = self.third_body_effect(np.copy(state), gm=1.26686534e8, r_third_body=np.copy(-r_sun_to_apophis)+np.copy(r_sun_to_jup))

        # print('SRP / Sun 3rd' + str(t))
        # print(np.divide(C_SRP[3:6], C_3rd_sun[3:6]))
        # print(180/math.pi*np.arccos(np.dot(C_SRP[3:6], C_3rd_sun[3:6])/np.linalg.norm(C_3rd_sun[3:6])/np.linalg.norm(C_SRP[3:6])))
        #print(np.linalg.norm(C_SRP))

        return np.matmul(A, state) + C + C_SRP + C_3rd_sun + C_3rd_earth_moon + C_3rd_jupiter

    def solar_rad_pressure(self, state, r_sun_to_body):
        """Method computes acceleration from SRP

        Args:
            state [7]: vector containing the state from the central body to the spacecraft
            r_sun_to_body [3]: vector from the sun to the central body

        Returns:
            [7] vector of the SRP acceleration
        """
        r_sun_sc = (state[0:3] + r_sun_to_body)*1000  # sun to s/c, meters
        r_sun_sc_mag = np.linalg.norm(r_sun_sc)
        r_sun_sc_norm = r_sun_sc / r_sun_sc_mag
        r_sun_sc_AU = r_sun_sc_mag / 1.495978707e11
        a_SRP = 4.56*10**-6 * self.C_R * self.surface_area / np.exp(state[6]) * (1 / r_sun_sc_AU)**2 * r_sun_sc_norm
        return np.array([0, 0, 0, a_SRP[0], a_SRP[1], a_SRP[2], 0])/1000

    def third_body_effect(self, state, gm, r_third_body):
        """Method computes acceleration from third body effects

        Args:
            state: vector containing the state from the central body to the spacecraft
            gm: gravitational parameter of the third body
            r_third_body: position vector from the central body to the third body

        Returns:
            [7] vector of the 3rd body acceleration
        """
        r_third_minus_sc = r_third_body - state[0:3]
        r_third_minus_sc_norm = np.linalg.norm(r_third_minus_sc)
        r_third_norm = np.linalg.norm(r_third_body)
        a_3rd = gm * (r_third_minus_sc / r_third_minus_sc_norm ** 3 - r_third_body / r_third_norm ** 3)
        return np.array([0, 0, 0, a_3rd[0], a_3rd[1], a_3rd[2], 0])

    def state_prop(self, asteroid, time_elapsed):
        """Propagates true and estimated states and the covariance forward for use in EKF and BLS

        Args:
            asteroid: instance of Asteroid class

        Returns:
            accel [3]: vector containing the polyhedral gravitational acceleration at the state before propagation
        """

        delta_v = np.array([self.delta_v[0], self.delta_v[1], self.delta_v[2], np.linalg.norm(np.array([self.delta_v[0], self.delta_v[1], self.delta_v[2]]))]).reshape(4)

        s = np.copy(self.s)
        print('current true state')
        print(s)
        shat = np.copy(self.shat)

        w_ast = asteroid.asteroid_ang_vel
        v_ex = self.exhaust_velocity
        dt = self.dt

        accel, sa = asteroid.newpolygrav_vec(s[0:3])
        accel = accel.reshape(3)

        # true state
        A = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
        #print('mass change')
        #print(-np.exp(delta_v[3]/v_ex))
        #print('current mass')
        #print(np.exp(s[6]))

        if delta_v[3] > 0:
            delta_v_error = self.gates(delta_v[0:3])
            delta_v_error_mag = np.linalg.norm(delta_v_error)
        else:
            delta_v_error = np.array([0, 0, 0])
            delta_v_error_mag = 0

        print('delta_v_error (mm/s)')
        print(delta_v_error * 1e6)
        self.delta_v_error_hist[:, time_elapsed+1] = delta_v_error

        s += np.array([0, 0, 0, delta_v[0] + delta_v_error[0], delta_v[1] + delta_v_error[1], delta_v[2] + delta_v_error[2], -(delta_v[3] + delta_v_error_mag)/v_ex])
        shat += np.array([0, 0, 0, delta_v[0], delta_v[1], delta_v[2], -delta_v[3]/v_ex])

        t = np.array([0, dt])

        sol_s = odeint(self.ode_dynamics, np.copy(s), t, args=(A, asteroid, time_elapsed, 'poly'), rtol=1e-12, atol=1e-12)

        sol_shat = odeint(self.ode_dynamics, np.copy(shat), t, args=(A, asteroid, time_elapsed, 'poly'), rtol=1e-12, atol=1e-12)

        obs_count = time_elapsed + 2  # total observations collected
        N = self.N_batch
        t_batch = np.arange(N) * dt

        if obs_count <= N:
            local_j = obs_count - 1
        else:
            phase = (obs_count - N) % (N - 1)
            local_j = phase

        t_current = local_j * dt

        print("t_batch:", t_batch)
        print("t_current:", t_current)
        print("local_j:", local_j)

        self.s = sol_s[-1, :]
        self.shat = sol_shat[-1, :]

        self.s_observed[:, time_elapsed+1] = sol_s[-1, 0:6]

        print(self.designation)

        # covariance propagation for spacecraft
        gg = asteroid.gravgrad_vec(s[0:3])
        dCdx = np.pad(gg, ((3, 1), (0, 4)), mode='constant')
        F = np.eye(self.num_states) + (A + dCdx)*dt

        R_RTN_to_I = self.get_RTN_to_inertial(self.s[0:3], self.s[3:6])

        Q_tilde = R_RTN_to_I @ self.Q @ R_RTN_to_I.T
        S_i = np.block([[Q_tilde * self.dt**3 / 3, Q_tilde * self.dt**2 / 2, np.zeros([3, 1])],
                        [Q_tilde * self.dt**2 / 2, Q_tilde * self.dt, np.zeros([3, 1])],
                        [np.zeros([1, 3]), np.zeros([1, 3]), 1e-20]])

        #print(S_i)

        self.P = np.matmul(F, np.matmul(self.P, F.T)) + S_i

        return accel

    def gates(self, delta_v):
        """Method implements the Gates model to simulate delta V errors

        Args:
            delta_v: [3] vector of the delta V to noise

        Returns:
            delta_v_error: [3] vector of the erroneous additional delta V in the same frame as delta_v
        """

        y = np.linalg.norm(delta_v)

        s1 = self.gates_sigma_1
        s2 = self.gates_sigma_2 / 100  # convert percentage to proportion
        s3 = self.gates_sigma_3 / 1000 / 1000  # convert mm/s to km/s
        s4 = self.gates_sigma_4 / 1000  # convert mrad to rad

        P_gates_diag = np.array([s1 ** 2 + y ** 2 * s2 ** 2, s3 ** 2 + y ** 2 * s4 ** 2, s3 ** 2 + y ** 2 * s4 ** 2])

        # get rotation matrix to go from inertial to delta_v frame
        delta_v_frame = np.array([1, 0, 0])
        delta_v_direction = delta_v / y
        delta_v_cross = np.cross(delta_v_direction, delta_v_frame)
        delta_v_dot = np.dot(delta_v_direction, delta_v_frame)
        axis = delta_v_cross / np.linalg.norm(delta_v_cross)
        angle = np.arccos(delta_v_dot)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        delta_v_rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        delta_v_error = np.linalg.inv(delta_v_rotation_matrix) @ (P_gates_diag ** 0.5 * np.random.normal(loc=0, scale=1, size=len(P_gates_diag))).reshape([3, 1])

        return delta_v_error.reshape(3)


def main():
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    spacecraft = Spacecraft(asteroid, config_file, None, 0)




if __name__ == '__main__':
    main()