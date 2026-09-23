# Polyhedral Gravity MPC Algorithm
# Logan Feld
import io
import pstats
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from typing import NamedTuple
import math
from datetime import datetime
import cProfile

import functools

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src import ekf
from src.Spacecraft import Spacecraft

matplotlib.use('Agg')  # TkAgg
plt.rcParams["figure.figsize"] = (16, 10)
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from src.MinDisSegmentTime_Vectorized import MinDisSegmentPoint_Vectorized
from src.MinDisSegmentTime_Vectorized import MinDisSegmentPoint_Vectorized_jax
from src.asteroid import Asteroid
from src.simulation import Simulation
from src.ekf import EKF
from src.BLS import BLS
from scipy.integrate import odeint
from scipy.io import savemat

from interpax import interp1d

#jax.config.update("jax_log_compiles", True)
#jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_enable_x64", True)

plt.rcParams.update({'font.size': 14})
np.set_printoptions(precision=17)


class MPC:
    """Class for running the Model Predictive Control algorithm for proximity operations around Apophis
    """

    def __init__(self, simulation, asteroid, spacecraft):
        self.H = spacecraft[0].horizon_steps
        self.num_states = spacecraft[0].num_states
        self.num_controls = spacecraft[0].num_controls
        self.run_MPC(simulation, asteroid, spacecraft)


    def make_grids(self, spacecraft, t0, tn, N):
        """Precompute sun, EMB, jupiter position grid over the integration interval.

        Args:
            spacecraft: an instance of the spacecraft class
            t0: scalar for initial epoch in the full ephemeris data to start grid at
            tn: scalar for final epoch in the full ephemeris to extract
            N: number of points between t0 and tn to interpolate

        Returns:
            times_grid: [1, N] grid of the times in the grid in seconds from the start of the FULL ephemeris data
            r_sun_grid: [3, N] grid of sun->apophis inertial positions at the times in times_grid
            r_EMB_grid: [3, N] grid of sun->EMB inertial positions at the times in times_grid
            r_jup_grid: [3, N] grid of sun->jupiter inertial positions at the times in times_grid
        """
        times_grid = jnp.linspace(t0, tn, N)

        r_sun_grid = jnp.stack([interp1d(times_grid, spacecraft.sun_times, spacecraft.r_sun[i, :], method="cubic") for i in range(spacecraft.r_sun.shape[0])], axis=0)
        r_EMB_grid = jnp.stack([interp1d(times_grid, spacecraft.EMB_times, spacecraft.r_EMB[i, :], method="cubic") for i in range(spacecraft.r_EMB.shape[0])], axis=0)
        r_jup_grid = jnp.stack([interp1d(times_grid, spacecraft.jup_times, spacecraft.r_jup[i, :], method="cubic") for i in range(spacecraft.r_jup.shape[0])], axis=0)

        return times_grid, r_sun_grid, r_EMB_grid, r_jup_grid

    def get_rotation(self, n_sun):
        """Method computes the rotation matrix R to express the daughtership states in the sun terminator plane frame

        Args:
            n_sun: [3] unit vector of the Sun->body direction

        Returns:
            R: [3, 3] rotation matrix such that R*n_sun = [0, 0, 1]
        """
        # z-axis
        z = n_sun

        # reference vector not parallel to z
        if abs(z[0]) < 0.9:
            r = np.array([1.0, 0.0, 0.0])
        else:
            r = np.array([0.0, 1.0, 0.0])

        # x and y-axes
        x = np.cross(r, z)
        x /= np.linalg.norm(x)

        y = np.cross(z, x)

        # compute rotation matrix
        R = np.vstack((x, y, z))
        return R

    def max_vec(self, vec, max):
        """Method clips vector magnitude to not exceed the scalar max value

        Args:
            vec: [3] vector being clipped
            max: [1] scalar magnitude to cap the vector magnitude with

        Returns:
            vec: vec or clipped vec
        """
        norm = np.linalg.norm(vec)
        if norm > max:
            return vec * (max/norm)
        return vec

    def zero_vec(self, vec, value, min):
        """Method zeros vector magnitude if a value is below some minimum value

        Args:
            vec: [3] vector being possibly zeroed
            value: [1] scalar to determine the zero-ing
            min: [1] scalar that sets the threshold for zero-ing

        Returns:
            vec: possibly zeroed vec
        """
        if value <= min:
            return 0*vec
        return vec

    def scipy_fun(self, dv_flat):
        """Method takes in a flattened version of the decision variables, shapes them properly, applies a scale factor, and obtains the optimal cost and jacobians

        Args:
            dv_flat: [H*2*3] vector of initial deltaV guess

        Returns:
            tot: total scalar cost from cost function
            grad_scaled: [H*2*3] vector of the jacobians of the scalar cost with respect to the deltaV components
        """
        scale = 1e-7
        dv_phys = dv_flat.reshape(self.H, 2, 3) * scale
        tot, grad_phys = cost_and_grad_jitted(dv_phys, self.const_args)
        grad_scaled = grad_phys * scale
        return float(tot), grad_scaled.ravel()

    def run_MPC(self, simulation, asteroid, spacecraft):
        """Method runs the model predictive control (MPC) algorithm and converts all outputs into MATLAB .mat arrays for good plotting

        Args:
            simulation: an instance of the Simulation class
            asteroid: an instance of the Asteroid class
            spacecraft: an instance of the Spacecraft class
        """
        time_elapsed = 0  # initialize time step count

        self.last_costs = {"total": 0, "pos_cost": 0, "dot_cost": 0, "dv_cost_1": 0, "dv_cost_2": 0, "a_cost": 0, "e_cost": 0, "i_cost": 0}
        num_spacecraft = len(spacecraft)
        mothership = spacecraft[simulation.mothership_index]
        has_done_a_batch = False

        duration = 20000  # dummy value for sim duration

        # compute number of decision variables
        longest_control_dim = 0
        for i in range(num_spacecraft):
            duration_test = spacecraft[i].duration
            control_test = spacecraft[i].num_controls - 1
            if duration_test < duration:
                duration = duration_test
            if control_test > longest_control_dim:
                longest_control_dim = control_test

        for i in range(num_spacecraft):
            spacecraft[i].duration = duration

        # initialize cost term histories
        self.cost_hist = np.zeros(spacecraft[0].duration)
        self.pos_cost_hist = np.zeros(spacecraft[0].duration)
        self.dot_cost_hist = np.zeros(spacecraft[0].duration)
        self.dv_cost_1_hist = np.zeros(spacecraft[0].duration)
        self.dv_cost_2_hist = np.zeros(spacecraft[0].duration)
        self.a_cost_hist = np.zeros(spacecraft[0].duration)
        self.e_cost_hist = np.zeros(spacecraft[0].duration)
        self.cov_cost_hist = np.zeros(spacecraft[0].duration)
        self.i_cost_hist = np.zeros(spacecraft[0].duration)
        self.bistatic_angle_true = np.zeros(spacecraft[0].duration)
        self.bistatic_angle_hat = np.zeros(spacecraft[0].duration)

        self.horizon_dv_hist = np.zeros([spacecraft[0].horizon_steps*longest_control_dim*(num_spacecraft-1), spacecraft[0].duration])

        # initialize reference trajectory segments for use in cost
        delta_v_opt = np.zeros([num_spacecraft, longest_control_dim])
        tof_1 = spacecraft[1].s_d.shape[1] - 1
        p1_1_np = np.transpose(spacecraft[1].s_d[0:3, 0:tof_1])
        p2_1_np = np.transpose(spacecraft[1].s_d[0:3, 1:tof_1+1])
        tof_2 = spacecraft[2].s_d.shape[1] - 1
        p1_2_np = np.transpose(spacecraft[2].s_d[0:3, 0:tof_2])
        p2_2_np = np.transpose(spacecraft[2].s_d[0:3, 1:tof_2 + 1])

        # convert cost parameters into JAX arrays
        dt_MPC_multiplier = jnp.array(simulation.dt_MPC_multiplier)
        mu = jnp.array(asteroid.mu_grav_param)
        dt = jnp.array(spacecraft[0].dt)
        exhaust_velocity = jnp.array([spacecraft[1].exhaust_velocity, spacecraft[2].exhaust_velocity])
        Q = jnp.stack([spacecraft[1].Q, spacecraft[2].Q])
        max_segment_length = jnp.array([spacecraft[1].max_segment_length, spacecraft[2].max_segment_length])
        gates_sigma_1 = jnp.array(spacecraft[0].gates_sigma_1)
        gates_sigma_1_factor = jnp.array(spacecraft[0].gates_sigma_1_factor)
        r_0 = jnp.array(spacecraft[1].r_0)
        C_R = jnp.array([spacecraft[1].C_R, spacecraft[2].C_R])
        surface_area = jnp.array([spacecraft[1].surface_area, spacecraft[2].surface_area])
        p1_1 = jnp.array(p1_1_np)
        p2_1 = jnp.array(p2_1_np)
        p1_2 = jnp.array(p1_2_np)
        p2_2 = jnp.array(p2_2_np)

        s = np.array([np.copy(spacecraft[1].shat), np.copy(spacecraft[2].shat)]).T
        P = np.zeros([7, 7, 2])
        P[:, :, 0] = np.copy(spacecraft[1].P)
        P[:, :, 1] = np.copy(spacecraft[2].P)
        offset = np.array([np.copy(spacecraft[1].offset), np.copy(spacecraft[2].offset)]).T

        # initialize MPC ephemeris grids for first simulation ephemeris window
        time_passed = time_elapsed * spacecraft[0].dt
        times_grid, r_sun_grid, r_EMB_grid, r_jup_grid = self.make_grids(spacecraft[0], time_passed, time_passed + spacecraft[0].dt * simulation.dt_MPC_multiplier * spacecraft[0].horizon_steps, 50*spacecraft[0].horizon_steps)

        self.const_args = make_const_args(s, P, dt_MPC_multiplier, mu, dt, exhaust_velocity, Q, max_segment_length, gates_sigma_1, gates_sigma_1_factor, r_0, C_R, surface_area, p1_1, p2_1, p1_2, p2_2, time_elapsed, times_grid, r_sun_grid, r_EMB_grid, r_jup_grid, offset)

        control_guess_length = 0
        for i in [1, 2]:
            control_guess_length += (spacecraft[i].num_controls - 1) * spacecraft[i].horizon_steps

        delta_v_ig = jnp.zeros(control_guess_length)

        self.jacobian_hist = np.zeros([spacecraft[0].duration, control_guess_length])

        print("============================================================================")

        while duration > time_elapsed + 1:  # while we still have some simulation left to run

            delta_v_ig = jnp.zeros(control_guess_length)  # initial guess for deltaV control along horizon

            x0 = jnp.copy(jnp.float32(delta_v_ig))

            # update states, covariances for current MPC time step/iteration
            s = np.array([np.copy(spacecraft[1].shat), np.copy(spacecraft[2].shat)]).T
            P = np.zeros([7, 7, 2])
            P[:, :, 0] = np.copy(spacecraft[1].P)
            P[:, :, 1] = np.copy(spacecraft[2].P)

            if has_done_a_batch:
                time_passed = time_elapsed * spacecraft[0].dt
                times_grid, r_sun_grid, r_EMB_grid, r_jup_grid = self.make_grids(spacecraft[0], time_passed, time_passed + spacecraft[0].dt * simulation.dt_MPC_multiplier * spacecraft[0].horizon_steps, 50*spacecraft[0].horizon_steps)
                self.const_args = self.const_args._replace(s0=jnp.array(s), P0=jnp.array(P), time_elapsed=jnp.array(time_elapsed), times_grid=times_grid, r_sun_grid=r_sun_grid, r_EMB_grid=r_EMB_grid, r_jup_grid=r_jup_grid)

                sol = fmin_l_bfgs_b(func=lambda x: self.scipy_fun(x)[0], x0=x0, fprime=lambda x: self.scipy_fun(x)[1], maxiter=10000, factr=100, pgtol=1e-4)
            else:  # if we havent done a batch, continue accruing observations until we can update Dship state estimates w/ BLS and don't burn until we have updated Dship state estimates
                sol = [x0*1e7, 0]

            print(sol)

            self.horizon_dv_hist[:, time_elapsed] = sol[0]  # commanded deltaV from prediction horizon

            costs = cost_breakdown_jitted(jnp.float32(sol[0]).reshape(self.H, 2, 3) / 1e7, self.const_args)  # get individual cost components (not just scalar total)

            print('Computed minimum cost function value = ' + str(sol[1]))
            self.cost_hist[time_elapsed] = costs["total"]

            if has_done_a_batch:
                print('pos')
                print(costs["pos_cost"])
                print('dot')
                print(costs["dot_cost"])
                print('dv')
                print(costs["dv_cost_1"])
                print(costs["dv_cost_2"])
                print('a')
                print(costs["a_cost"])
                print('e')
                print(costs["e_cost"])
                print('i')
                print(costs["i_cost"])
                print('jacobians')
                print(sol[2]['grad'])
                print('-----------------------------')
                self.pos_cost_hist[time_elapsed] = costs["pos_cost"]
                self.dot_cost_hist[time_elapsed] = costs["dot_cost"]
                self.dv_cost_1_hist[time_elapsed] = costs["dv_cost_1"]
                self.dv_cost_2_hist[time_elapsed] = costs["dv_cost_2"]
                self.a_cost_hist[time_elapsed] = costs["a_cost"]
                self.e_cost_hist[time_elapsed] = costs["e_cost"]
                self.i_cost_hist[time_elapsed] = costs["i_cost"]
                self.jacobian_hist[time_elapsed, :] = sol[2]['grad'].flatten()  # C order
            else:
                self.pos_cost_hist[time_elapsed] = 0
                self.dot_cost_hist[time_elapsed] = 0
                self.dv_cost_1_hist[time_elapsed] = 0
                self.dv_cost_2_hist[time_elapsed] = 0
                self.a_cost_hist[time_elapsed] = 0
                self.e_cost_hist[time_elapsed] = 0
                self.i_cost_hist[time_elapsed] = 0
                self.jacobian_hist[time_elapsed, :] = np.zeros(control_guess_length)

            # compute local true and estimated bistatic/antipodal angle between daughterships
            n_sun_to_apophis = np.array([spacecraft[0].spline_sun_x(time_elapsed*spacecraft[0].dt), spacecraft[0].spline_sun_y(time_elapsed*spacecraft[0].dt), spacecraft[0].spline_sun_z(time_elapsed*spacecraft[0].dt)])
            n_sun_to_apophis /= np.linalg.norm(n_sun_to_apophis)
            print(time_elapsed*spacecraft[0].dt)
            print(n_sun_to_apophis)
            R = self.get_rotation(n_sun_to_apophis)
            dot_vec_d1 = np.matmul(R, np.copy(spacecraft[1].s[0:3]))
            dot_vec_d2 = np.matmul(R, np.copy(spacecraft[2].s[0:3]))
            dot_vec_d1_hat = np.matmul(R, np.copy(spacecraft[1].shat[0:3]))
            dot_vec_d2_hat = np.matmul(R, np.copy(spacecraft[2].shat[0:3]))
            offset_d1 = np.matmul(R, np.copy(offset[:, 0]))
            offset_d2 = np.matmul(R, np.copy(offset[:, 1]))
            offset_avg = 0.5*(offset_d1 + offset_d2)
            print(offset_avg)
            print((dot_vec_d1-offset_avg)/np.linalg.norm(dot_vec_d1-offset_avg))
            print((dot_vec_d2-offset_avg)/np.linalg.norm(dot_vec_d2-offset_avg))
            print('true bistatic angle (deg)')
            bistatic_true = 180 / math.pi * np.arccos(np.dot((dot_vec_d1 - offset_avg)/np.linalg.norm(dot_vec_d1 - offset_avg), (dot_vec_d2 - offset_avg)/np.linalg.norm(dot_vec_d2 - offset_avg)))
            print(bistatic_true)
            self.bistatic_angle_true[time_elapsed] = bistatic_true
            print('estimated bistatic angle (deg)')
            bistatic_hat = 180 / math.pi * np.arccos(np.dot((dot_vec_d1_hat - offset_avg) / np.linalg.norm(dot_vec_d1_hat - offset_avg), (dot_vec_d2_hat - offset_avg) / np.linalg.norm(dot_vec_d2_hat - offset_avg)))
            print(bistatic_hat)
            self.bistatic_angle_hat[time_elapsed] = bistatic_hat

            print('DeltaV (mm/s)')  # /1e7 to remove scaling and return to km/s, *1e6 to put into mm/s
            print(np.linalg.norm(sol[0][0:(spacecraft[1].num_controls - 1)]) / 1e7 * 1e6)
            print(np.linalg.norm(sol[0][(spacecraft[1].num_controls - 1):2*(spacecraft[1].num_controls - 1)]) / 1e7 * 1e6)

            # get the count of how many observations we have collected in total and how many that is for the current BLS window
            obs_count = time_elapsed + 2
            if obs_count <= spacecraft[0].N_batch:
                iter_in_batch = obs_count - 1
            else:
                iter_in_batch = (obs_count - spacecraft[0].N_batch) % (spacecraft[0].N_batch - 1)

            print('iter in batch')
            print(iter_in_batch)
            print('----')
            for i in [1, 2]:  # for each daughtership

                # if the first burn in the horizon is below the burn threshold, zero out the deltaV
                if np.linalg.norm(sol[0][(i-1)*(spacecraft[1].num_controls - 1):i*(spacecraft[1].num_controls - 1)]) * 1e-7 < spacecraft[i].gates_sigma_1_factor*spacecraft[i].gates_sigma_1:
                    delta_v_opt = np.zeros(spacecraft[i].num_controls - 1)
                else:  # if the first burn in the horizon is above the deltaV threshold, store the deltaV
                    delta_v_opt = sol[0][(i-1)*(spacecraft[1].num_controls - 1):i*(spacecraft[1].num_controls - 1)] * 1e-7

                delta_v_opt = self.max_vec(np.copy(delta_v_opt), spacecraft[i].thrust_max / np.exp(spacecraft[i].s[6]) * 0.0005)  # if the burn is above our max thruster thrust, cap it
                delta_v_opt = self.zero_vec(np.copy(delta_v_opt), np.exp(spacecraft[i].s[6]), spacecraft[i].dry_mass)  # if we have no more propellant, zero out the deltaV
                print('dv mag')
                print(np.linalg.norm(delta_v_opt)*1e6)
                print('MPC Optimal Delta V for ' + spacecraft[i].designation + ' the timestep (Delta V_X, V_Y, V_Z, mm/s) = ' + str(delta_v_opt*10**6))
                spacecraft[i].delta_v = np.copy(delta_v_opt)
                spacecraft[i].delta_v_hist[:, time_elapsed+1] = spacecraft[i].delta_v
                if np.linalg.norm(delta_v_opt) > 0:
                    spacecraft[i].delta_v_num_in_batch[time_elapsed+1] = iter_in_batch

            for i in range(len(spacecraft)):
                true_accel = spacecraft[i].state_prop(asteroid, time_elapsed)  # propagate true and estimated states with the commanded deltaV
                pred_accel = -asteroid.mu_grav_param / np.linalg.norm(spacecraft[i].s[0:3]) ** 3 * spacecraft[i].s[0:3]  # low res gravity model used within MPC for comparison purposes
                EKF = ekf.EKF(time_elapsed)  # initialize EKF class
                seen = EKF.run_EKF(simulation, asteroid, spacecraft[i])  # update mothership estimated state with EKF
                print('observations total:')
                print(time_elapsed+2)
                print('Estimated State (pre BLS):')
                print(spacecraft[i].shat)
                if obs_count >= spacecraft[i].N_batch:  # check if the current number of observations is enough for a new BLS
                    if (obs_count - spacecraft[i].N_batch) % (spacecraft[i].N_batch - 1) == 0:
                        if spacecraft[i].designation != 'mothership':
                            leftover = 0
                            bls = BLS(simulation, asteroid, spacecraft[i], mothership, time_elapsed, leftover)  # perform BLS to update state estimates and maneuver reconstruction for Dships
                            has_done_a_batch = True  # comment this out to only propagate dynamics, no thrust

                    elif obs_count == duration:
                        if spacecraft[i].designation != 'mothership':
                            leftover = (obs_count - spacecraft[i].N_batch) % (spacecraft[i].N_batch - 1)
                            bls = BLS(simulation, asteroid, spacecraft[i], mothership, time_elapsed, leftover)

                if spacecraft[i].designation == 'mothership':
                    spacecraft[i].feature_array.append(seen)
                    print('Features in view: ' + str(seen.shape[0]))

                print('True State:')
                print(spacecraft[i].s)
                print('Estimated State (post BLS):')
                print(spacecraft[i].shat)
                print('s - s_hat:')
                print(spacecraft[i].s-spacecraft[i].shat)
                print('Covariance diagonals:')
                print(np.diag(spacecraft[i].P))
                spacecraft[i].s_hist[:, time_elapsed+1] = spacecraft[i].s
                spacecraft[i].shat_hist[:, time_elapsed+1] = spacecraft[i].shat
                spacecraft[i].true_accel_hist[:, time_elapsed+1] = true_accel
                spacecraft[i].pred_accel_hist[:, time_elapsed+1] = pred_accel
                self.time_of_flight = (time_elapsed+1) * spacecraft[i].dt
                print('Time elapsed (s) = ' + str(self.time_of_flight))

                tout, dmin, p1s, p2s, sindex = MinDisSegmentPoint_Vectorized(spacecraft[i].p1, spacecraft[i].p2, R @ spacecraft[i].s[0:3])
                vmin = np.linalg.norm(R @ spacecraft[i].s[3:6] - spacecraft[i].vvec[sindex])
                print("Minimum distance from true pos to nearest ref line segment: " + str(dmin * 1000) + " m")
                print("Index of reference segment ship is closest to: " + str(sindex))
                print("Velocity difference from true vel to nearest ref line segment vel: " + str(vmin * 1e6) + " mm/s")
                print("---------------------------------------------------------------------")

                spacecraft[i].dmin_hist[time_elapsed+1] = dmin
                spacecraft[i].vmin_hist[time_elapsed+1] = vmin
                spacecraft[i].cov_hist[:, :, time_elapsed + 1] = np.copy(spacecraft[i].P)

            time_elapsed += 1

        spacecraft[0].shat_post_BLS_hist = spacecraft[0].shat_hist

        # save all the plotting stuff as MATLAB .mat files for better plots
        savemat('output/MATLAB_mats/s_hist_m.mat', {'myArray': spacecraft[0].s_hist})
        savemat('output/MATLAB_mats/s_hist_1.mat', {'myArray': spacecraft[1].s_hist})
        savemat('output/MATLAB_mats/s_hist_2.mat', {'myArray': spacecraft[2].s_hist})
        savemat('output/MATLAB_mats/s_d_m.mat', {'myArray': spacecraft[0].s_d})
        savemat('output/MATLAB_mats/s_d_1.mat', {'myArray': spacecraft[1].s_d})
        savemat('output/MATLAB_mats/s_d_2.mat', {'myArray': spacecraft[2].s_d})
        savemat('output/MATLAB_mats/dmin_hist_m.mat', {'myArray': spacecraft[0].dmin_hist})
        savemat('output/MATLAB_mats/dmin_hist_1.mat', {'myArray': spacecraft[1].dmin_hist})
        savemat('output/MATLAB_mats/dmin_hist_2.mat', {'myArray': spacecraft[2].dmin_hist})
        savemat('output/MATLAB_mats/vmin_hist_m.mat', {'myArray': spacecraft[0].vmin_hist})
        savemat('output/MATLAB_mats/vmin_hist_1.mat', {'myArray': spacecraft[1].vmin_hist})
        savemat('output/MATLAB_mats/vmin_hist_2.mat', {'myArray': spacecraft[2].vmin_hist})
        savemat('output/MATLAB_mats/delta_v_hist_m.mat', {'myArray': spacecraft[0].delta_v_hist})
        savemat('output/MATLAB_mats/delta_v_hist_1.mat', {'myArray': spacecraft[1].delta_v_hist})
        savemat('output/MATLAB_mats/delta_v_hist_2.mat', {'myArray': spacecraft[2].delta_v_hist})
        savemat('output/MATLAB_mats/delta_v_error_hist_m.mat', {'myArray': spacecraft[0].delta_v_error_hist})
        savemat('output/MATLAB_mats/delta_v_error_hist_1.mat', {'myArray': spacecraft[1].delta_v_error_hist})
        savemat('output/MATLAB_mats/delta_v_error_hist_2.mat', {'myArray': spacecraft[2].delta_v_error_hist})
        savemat('output/MATLAB_mats/cost_hist.mat', {'myArray': self.cost_hist})
        savemat('output/MATLAB_mats/shat_post_BLS_hist_m.mat', {'myArray': spacecraft[0].shat_post_BLS_hist})
        savemat('output/MATLAB_mats/shat_post_BLS_hist_1.mat', {'myArray': spacecraft[1].shat_post_BLS_hist})
        savemat('output/MATLAB_mats/shat_post_BLS_hist_2.mat', {'myArray': spacecraft[2].shat_post_BLS_hist})
        savemat('output/MATLAB_mats/shat_hist_m.mat', {'myArray': spacecraft[0].shat_hist})
        savemat('output/MATLAB_mats/shat_hist_1.mat', {'myArray': spacecraft[1].shat_hist})
        savemat('output/MATLAB_mats/shat_hist_2.mat', {'myArray': spacecraft[2].shat_hist})
        savemat('output/MATLAB_mats/true_accel_hist_m.mat', {'myArray': spacecraft[0].true_accel_hist})
        savemat('output/MATLAB_mats/true_accel_hist_1.mat', {'myArray': spacecraft[1].true_accel_hist})
        savemat('output/MATLAB_mats/true_accel_hist_2.mat', {'myArray': spacecraft[2].true_accel_hist})
        savemat('output/MATLAB_mats/pred_accel_hist_m.mat', {'myArray': spacecraft[0].pred_accel_hist})
        savemat('output/MATLAB_mats/pred_accel_hist_1.mat', {'myArray': spacecraft[1].pred_accel_hist})
        savemat('output/MATLAB_mats/pred_accel_hist_2.mat', {'myArray': spacecraft[2].pred_accel_hist})
        savemat('output/MATLAB_mats/pos_cost_hist.mat', {'myArray': self.pos_cost_hist})
        savemat('output/MATLAB_mats/dot_cost_hist.mat', {'myArray': self.dot_cost_hist})
        savemat('output/MATLAB_mats/dv_cost_1_hist.mat', {'myArray': self.dv_cost_1_hist})
        savemat('output/MATLAB_mats/dv_cost_2_hist.mat', {'myArray': self.dv_cost_2_hist})
        savemat('output/MATLAB_mats/a_cost_hist.mat', {'myArray': self.a_cost_hist})
        savemat('output/MATLAB_mats/e_cost_hist.mat', {'myArray': self.e_cost_hist})
        savemat('output/MATLAB_mats/cov_cost_hist.mat', {'myArray': self.cov_cost_hist})
        savemat('output/MATLAB_mats/i_cost_hist.mat', {'myArray': self.i_cost_hist})
        savemat('output/MATLAB_mats/bistatic_angle_true_hist.mat', {'myArray': self.bistatic_angle_true})
        savemat('output/MATLAB_mats/bistatic_angle_hat_hist.mat', {'myArray': self.bistatic_angle_hat})
        savemat('output/MATLAB_mats/delta_v_hat_hist_m.mat', {'myArray': spacecraft[0].delta_v_hat_hist})
        savemat('output/MATLAB_mats/delta_v_hat_hist_1.mat', {'myArray': spacecraft[1].delta_v_hat_hist})
        savemat('output/MATLAB_mats/delta_v_hat_hist_2.mat', {'myArray': spacecraft[2].delta_v_hat_hist})
        savemat('output/MATLAB_mats/feature_array.mat', {'myArray': spacecraft[0].feature_array})
        savemat('output/MATLAB_mats/jacobian_hist.mat', {'myArray': self.jacobian_hist})
        savemat('output/MATLAB_mats/a_hist_1.mat', {'myArray': spacecraft[1].a_hist})
        savemat('output/MATLAB_mats/a_hist_2.mat', {'myArray': spacecraft[2].a_hist})
        savemat('output/MATLAB_mats/e_hist_1.mat', {'myArray': spacecraft[1].e_hist})
        savemat('output/MATLAB_mats/e_hist_2.mat', {'myArray': spacecraft[2].e_hist})
        savemat('output/MATLAB_mats/i_hist_1.mat', {'myArray': spacecraft[1].i_hist})
        savemat('output/MATLAB_mats/i_hist_2.mat', {'myArray': spacecraft[2].i_hist})
        savemat('output/MATLAB_mats/raan_hist_1.mat', {'myArray': spacecraft[1].raan_hist})
        savemat('output/MATLAB_mats/raan_hist_2.mat', {'myArray': spacecraft[2].raan_hist})
        savemat('output/MATLAB_mats/aop_hist_1.mat', {'myArray': spacecraft[1].aop_hist})
        savemat('output/MATLAB_mats/aop_hist_2.mat', {'myArray': spacecraft[2].aop_hist})
        savemat('output/MATLAB_mats/offset_1.mat', {'myArray': spacecraft[1].offset})
        savemat('output/MATLAB_mats/offset_2.mat', {'myArray': spacecraft[2].offset})
        savemat('output/MATLAB_mats/horizon_dv_hist.mat', {'myArray': self.horizon_dv_hist})
        savemat('output/MATLAB_mats/dt_multiplier.mat', {'myArray': simulation.dt_MPC_multiplier})
        savemat('output/MATLAB_mats/cov_hist_m.mat', {'myArray': spacecraft[0].cov_hist})
        savemat('output/MATLAB_mats/cov_hist_1.mat', {'myArray': spacecraft[1].cov_hist})
        savemat('output/MATLAB_mats/cov_hist_2.mat', {'myArray': spacecraft[2].cov_hist})
        savemat('output/MATLAB_mats/delta_v_cov_hist_1.mat', {'myArray': spacecraft[1].delta_v_cov_hist})
        savemat('output/MATLAB_mats/delta_v_cov_hist_2.mat', {'myArray': spacecraft[2].delta_v_cov_hist})

    def plot_MPC_and_SCvx(self, simulation, asteroid, spacecraft):
        """Method to plot the MPC and SCvx trajectories together

        Args:
            simulation: an instance of the Simulation class
            asteroid: an instance of the Asteroid class
            spacecraft: an instnace of the Spacecraft class
        """
        num_spacecraft = len(spacecraft)
        colors = ['yellow', 'purple', 'blue', 'orange', 'red', 'green']

        # Plot 0: 3D plot of mission
        fig0, ax0 = asteroid.plotpoly()
        legend_list = [asteroid.title]
        for i in range(num_spacecraft):
            ax0.plot3D(spacecraft[i].s_hist[0, :], spacecraft[i].s_hist[1, :], spacecraft[i].s_hist[2, :], linewidth=4, zorder=3, color=colors[2*i+1])
            legend_list.append(spacecraft[i].designation + ' true trajectory')

        ax0.axis('equal')
        print(legend_list)
        ax0.legend(legend_list, loc='upper left', prop={"size": 8})
        ax0.view_init(elev=25, azim=40)
        ax0.set_xlim([-4, 4])
        ax0.set_ylim([-4, 4])
        ax0.set_zlim([-4, 4])

        # fuel burned by each spacecraft
        fuel_burned = np.zeros(num_spacecraft)
        for i in range(num_spacecraft):
            fuel_burned[i] = spacecraft[i].wet_mass - np.exp(spacecraft[i].s_hist[6, -1])
            print('Fuel burned by ' + spacecraft[i].designation + ': ' + str(fuel_burned[i]) + ' kg')

        fig1, ax1 = plt.subplots(2, 2)
        fig2, ax2 = plt.subplots(2, 2)
        fig3, ax3 = plt.subplots(2, 2)
        fig4, ax4 = plt.subplots(1, 2)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        fig6, ax6 = plt.subplots(2, 3)
        fig7, ax7 = plt.subplots(2, 3)
        fig8, ax8 = plt.subplots(1, 3)
        fig9 = plt.figure()
        ax9 = fig9.add_subplot(111)
        fig10 = plt.figure()
        ax10 = fig10.add_subplot(111)
        fig11, ax11 = plt.subplots(1, 2)

        fig1.suptitle('Position Data (km)')
        fig2.suptitle('Velocity Data (mm/s)')
        fig3.suptitle('Delta V data (mm/s)')
        fig5.suptitle('Cost Over Time')
        fig6.suptitle('True and Estimated State Data')
        fig7.suptitle('True and Estimated State Differences')

        # Plot 1: Position Data
        # x, y, z vs time
        legend_list = []
        for i in range(num_spacecraft):
            ax1[0, 0].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[0, :], label='x (km)', color=colors[2*i+1])
            ax1[0, 1].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[1, :], label='y (km)', color=colors[2*i+1])
            ax1[1, 0].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[2, :], label='z (km)', color=colors[2*i+1])
            ax1[1, 1].plot(range(spacecraft[i].duration), spacecraft[i].dmin_hist[:], label='d (km)', color=colors[2*i])
            legend_list.append(spacecraft[i].designation + ' true trajectory')

        print(legend_list)

        ax1[0, 0].set_ylabel('Position (km)')
        ax1[0, 1].set_ylabel('Position (km)')
        ax1[1, 0].set_ylabel('Position (km)')
        ax1[1, 1].set_ylabel('Magnitude (km)')
        ax1[1, 0].set_xlabel('Epoch')
        ax1[1, 1].set_xlabel('Epoch')
        ax1[0, 0].legend(legend_list, prop={"size": 8})
        ax1[0, 0].set_title('X Position vs Time')
        ax1[0, 1].set_title('Y Position vs Time')
        ax1[1, 0].set_title('Z Position vs Time')
        ax1[1, 1].set_title('Position Difference Magnitude vs Time')
        ax1[1, 1].legend(['Mothership', 'Daughtership1', 'Daughtership2'], prop={"size": 8})

        # Plot 2: Velocity Data
        legend_list = []
        for i in range(num_spacecraft):
            ax2[0, 0].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[3, :]*1e6, label='x (km/s)', color=colors[2*i+1])
            ax2[0, 1].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[4, :]*1e6, label='y (km/s)', color=colors[2*i+1])
            ax2[1, 0].plot(np.arange(spacecraft[i].duration), spacecraft[i].s_hist[5, :]*1e6, label='z (km/s)', color=colors[2*i+1])
            ax2[1, 1].plot(range(spacecraft[i].duration), spacecraft[i].vmin_hist[:]*1e6, label='d (km/s)', color=colors[2*i])
            legend_list.append(spacecraft[i].designation + ' true trajectory')

        ax2[0, 0].set_ylabel('Velocity (mm/s)')
        ax2[0, 1].set_ylabel('Velocity (mm/s)')
        ax2[1, 0].set_ylabel('Velocity (mm/s)')
        ax2[1, 1].set_ylabel('Magnitude (mm/s)')
        ax2[1, 0].set_xlabel('Epoch')
        ax2[1, 1].set_xlabel('Epoch')
        ax2[0, 0].legend(legend_list, prop={"size": 8})
        ax2[0, 0].set_title('X Velocity vs Time')
        ax2[0, 1].set_title('Y Velocity vs Time')
        ax2[1, 0].set_title('Z Velocity vs Time')
        ax2[1, 1].set_title('Velocity Difference Magnitude vs Time')
        ax2[1, 1].legend(['Mothership', 'Daughtership1', 'Daughtership2'], prop={"size": 8})

        # Plot 3: Control
        # control vs time
        legend_list = []
        for i in [1, 2]:
            ax3[0, 0].plot(range(spacecraft[i].duration), spacecraft[i].delta_v_hist[0, :]*1e6, label='x (km/s)', color=colors[2*i])
            ax3[0, 1].plot(range(spacecraft[i].duration), spacecraft[i].delta_v_hist[1, :]*1e6, label='y (km/s)', color=colors[2*i])
            ax3[1, 0].plot(range(spacecraft[i].duration), spacecraft[i].delta_v_hist[2, :]*1e6, label='z (km/s)', color=colors[2*i])
            ax3[1, 1].plot(range(spacecraft[i].duration), np.linalg.norm(spacecraft[i].delta_v_hist[:, :], axis=0)*1e6, label='d (km/s)', color=colors[2*i])
            legend_list.append(spacecraft[i].designation + ' true trajectory')
            legend_list.append(spacecraft[i].designation + ' reference trajectory')

        ax3[0, 0].set_ylabel('Delta V (mm/s)')
        ax3[0, 1].set_ylabel('Delta V (mm/s)')
        ax3[1, 0].set_ylabel('Delta V (mm/s)')
        ax3[1, 1].set_ylabel('Magnitude (mm/s)')
        ax3[1, 0].set_xlabel('Epoch')
        ax3[1, 1].set_xlabel('Epoch')
        ax3[0, 0].legend(['Daughtership1', 'Daughtership2'], prop={"size": 8})
        ax3[1, 1].legend(['Daughtership1', 'Daughtership2'], prop={"size": 8})
        ax3[0, 0].set_title('Delta V_X vs Time')
        ax3[0, 1].set_title('Delta V_Y vs Time')
        ax3[1, 0].set_title('Delta V_Z vs Time')
        ax3[1, 1].set_title('Delta V Difference Magnitude vs Time')

        # Plot 4: Mass, Thrust, Distance, and Altitude
        # mass vs time

        legend_list = []
        for i in [1, 2]:
            delta_v_mag_hist = np.linalg.norm(spacecraft[i].delta_v_hist, axis=0)*1e6
            delta_v_error_mag_hist = np.linalg.norm(spacecraft[i].delta_v_error_hist, axis=0)*1e6
            ax4[0].step(range(spacecraft[i].duration), delta_v_mag_hist, label='delta V', color=colors[2*i])
            ax4[1].step(range(spacecraft[i].duration), delta_v_error_mag_hist, label='delta V error', color=colors[2*i])

        ax4[0].legend(['Daughtership1', 'Daughtership2'], prop={"size": 8})
        ax4[0].set_ylabel('Delta V (mm/s)')
        ax4[0].set_xlabel('Epoch')
        ax4[1].set_xlabel('Epoch')
        ax4[0].set_title('Daughtership Delta V Magnitude history')
        ax4[1].set_title('Daughtership Delta V Error Magnitude history')


        ax5.plot(range(len(self.cost_hist)-1), self.cost_hist[:-1], label='cost')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Cost')

        legend_list = []
        for i in range(num_spacecraft):
            ax6[0, 0].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[0, :], color=colors[2*i])
            ax6[0, 1].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[1, :], color=colors[2*i])
            ax6[0, 2].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[2, :], color=colors[2*i])
            ax6[1, 0].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[3, :]*1e6, color=colors[2*i])
            ax6[1, 1].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[4, :]*1e6, color=colors[2*i])
            ax6[1, 2].plot(range(spacecraft[i].duration), spacecraft[i].shat_post_BLS_hist[5, :]*1e6, color=colors[2*i])
            ax6[0, 0].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[0, :], color=colors[2*i+1])
            ax6[0, 1].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[1, :], color=colors[2*i+1])
            ax6[0, 2].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[2, :], color=colors[2*i+1])
            ax6[1, 0].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[3, :]*1e6, color=colors[2*i+1])
            ax6[1, 1].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[4, :]*1e6, color=colors[2*i+1])
            ax6[1, 2].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[5, :]*1e6, color=colors[2*i+1])
            legend_list.append(spacecraft[i].designation + ' estimated trajectory')
            legend_list.append(spacecraft[i].designation + ' true trajectory')
        ax6[0, 0].legend(legend_list, prop={"size": 8})
        ax6[1, 0].set_xlabel('Epoch')
        ax6[1, 1].set_xlabel('Epoch')
        ax6[1, 2].set_xlabel('Epoch')
        ax6[0, 0].set_ylabel('State and Estimated State History (km)')
        ax6[1, 0].set_ylabel('State and Estimated State History (mm/s)')
        ax6[0, 0].set_title('Position X')
        ax6[0, 1].set_title('Position Y')
        ax6[0, 2].set_title('Position Z')
        ax6[1, 0].set_title('Velocity X')
        ax6[1, 1].set_title('Velocity Y')
        ax6[1, 2].set_title('Velocity Z')

        legend_list = []

        for i in range(num_spacecraft):
            legend_list.append([spacecraft[i].designation + ' true - estimated diff'])
            ax7[0, 0].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[0, :] - spacecraft[i].shat_post_BLS_hist[0, :], color=colors[2*i])
            ax7[0, 1].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[1, :] - spacecraft[i].shat_post_BLS_hist[1, :], color=colors[2*i])
            ax7[0, 2].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[2, :] - spacecraft[i].shat_post_BLS_hist[2, :], color=colors[2*i])
            ax7[1, 0].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[3, :]*1e6 - spacecraft[i].shat_post_BLS_hist[3, :]*1e6, color=colors[2*i])
            ax7[1, 1].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[4, :]*1e6 - spacecraft[i].shat_post_BLS_hist[4, :]*1e6, color=colors[2*i])
            ax7[1, 2].plot(range(spacecraft[i].duration), spacecraft[i].s_hist[5, :]*1e6 - spacecraft[i].shat_post_BLS_hist[5, :]*1e6, color=colors[2*i])
        ax7[1, 0].set_xlabel('Epoch')
        ax7[1, 1].set_xlabel('Epoch')
        ax7[1, 2].set_xlabel('Epoch')
        ax7[0, 0].set_ylabel('True - Estimated State Difference (km)')
        ax7[1, 0].set_ylabel('True - Estimated State Difference (mm/s)')
        ax7[0, 0].set_title('Position X')
        ax7[0, 1].set_title('Position Y')
        ax7[0, 2].set_title('Position Z')
        ax7[1, 0].set_title('Velocity X')
        ax7[1, 1].set_title('Velocity Y')
        ax7[1, 2].set_title('Velocity Z')
        ax7[1, 2].legend(legend_list, prop={"size": 8})

        legend_list = []
        for i in range(num_spacecraft):
            legend_list.append(spacecraft[i].designation + ' true accel')
            legend_list.append(spacecraft[i].designation + ' predicted accel')
            ax8[0].plot(range(spacecraft[i].duration-1), spacecraft[i].true_accel_hist[0, 1:]*1e6, color=colors[2*i])
            ax8[1].plot(range(spacecraft[i].duration-1), spacecraft[i].true_accel_hist[1, 1:]*1e6, color=colors[2*i])
            ax8[2].plot(range(spacecraft[i].duration-1), spacecraft[i].true_accel_hist[2, 1:]*1e6, color=colors[2*i])
            ax8[0].plot(range(spacecraft[i].duration-1), spacecraft[i].pred_accel_hist[0, 1:]*1e6, color=colors[2*i+1])
            ax8[1].plot(range(spacecraft[i].duration-1), spacecraft[i].pred_accel_hist[1, 1:]*1e6, color=colors[2*i+1])
            ax8[2].plot(range(spacecraft[i].duration-1), spacecraft[i].pred_accel_hist[2, 1:]*1e6, color=colors[2*i+1])
        ax8[0].set_xlabel('Epoch')
        ax8[1].set_xlabel('Epoch')
        ax8[2].set_xlabel('Epoch')
        ax8[0].set_ylabel('Gravitational Acceleration (mm/s^2)')
        ax8[0].set_title('Gravitational Acceleration in X')
        ax8[1].set_title('Gravitational Acceleration in Y')
        ax8[2].set_title('Gravitational Acceleration in Z')
        ax8[0].legend(legend_list, prop={"size": 8})

        ax9.scatter(range(len(self.pos_cost_hist)), self.pos_cost_hist, label='pos_cost')
        ax9.scatter(range(len(self.dot_cost_hist)), self.dot_cost_hist, label='dot_cost')
        ax9.scatter(range(len(self.dv_cost_1_hist)), self.dv_cost_1_hist, label='dv_cost_1')
        ax9.scatter(range(len(self.dv_cost_2_hist)), self.dv_cost_2_hist, label='dv_cost_2')
        ax9.scatter(range(len(self.a_cost_hist)), self.a_cost_hist, label='a_cost')
        ax9.scatter(range(len(self.e_cost_hist)), self.e_cost_hist, label='e_cost')
        ax9.scatter(range(len(self.i_cost_hist)), self.i_cost_hist, label='i_cost')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Cost')
        ax9.set_title('Cost components')
        ax9.legend(['position', 'dot', 'deltaV_1', 'deltaV_2', 'semimajor axis', 'eccentricity', 'inclination'], loc='upper left', prop={"size": 8})

        ax10.plot(range(len(self.bistatic_angle_true)-1), self.bistatic_angle_true[:-1], label='true')
        ax10.plot(range(len(self.bistatic_angle_hat)-1), self.bistatic_angle_hat[:-1], label='true')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Angle (deg)')
        ax10.set_title('True and Estimated Bistatic Angle History')
        ax10.legend(['True', 'Estimated'], loc='upper left', prop={"size": 8})

        for i in [1, 2]:
            ax11[i - 1].step(range(spacecraft[i].duration), (spacecraft[i].delta_v_hist[0, :] + spacecraft[i].delta_v_error_hist[0, :] - spacecraft[i].delta_v_hat_hist[0, :])*1e6, label='delta V diff x')
            ax11[i - 1].step(range(spacecraft[i].duration), (spacecraft[i].delta_v_hist[1, :] + spacecraft[i].delta_v_error_hist[1, :] - spacecraft[i].delta_v_hat_hist[1, :])*1e6, label='delta V diff y')
            ax11[i - 1].step(range(spacecraft[i].duration), (spacecraft[i].delta_v_hist[2, :] + spacecraft[i].delta_v_error_hist[2, :] - spacecraft[i].delta_v_hat_hist[2, :])*1e6, label='delta V diff z')

        ax11[0].legend(['X', 'Y', 'Z'], prop={"size": 8})
        ax11[0].set_ylabel('True - Estimated Delta V (mm/s)')
        ax11[0].set_xlabel('Epoch')
        ax11[1].set_xlabel('Epoch')
        ax11[0].set_title('Daughtership 1 True - Estimated Delta V history')
        ax11[1].set_title('Daughtership 2 True - Estimated Delta V history')

        # save figures
        fig0.savefig('output/MPC and Ref Trajectories.png')
        fig1.savefig('output/MPC Position.png')
        fig2.savefig('output/MPC Velocity.png')
        fig3.savefig('output/MPC and Ref Control.png')
        fig4.savefig('output/MPC and Ref Mass and Thrust.png')
        fig5.savefig('output/MPC Cost.png')
        fig6.savefig('output/MPC True and Estimated States.png')
        fig7.savefig('output/MPC True and Estimated State Diff.png')
        fig8.savefig('output/MPC True and Predicted Gravitational Accelerations.png')
        fig9.savefig('output/MPC Cost Components.png')
        fig10.savefig('output/MPC True and Estimated Bistatic Angle.png')
        fig11.savefig('output/MPC True - Estimated Delta V.png')

    def update_trajectory(self, trajectory, spacecraft, i, ref=False):
        trajectory.set_data(spacecraft.s_hist[0, :i + 1], spacecraft.s_hist[1, :i + 1])
        trajectory.set_3d_properties(spacecraft.s_hist[2, :i + 1])
        return trajectory

    #def animate(self, i, ref_traj1, MPC_traj1, features_plot, ref_traj2, MPC_traj2, ref_traj3, MPC_traj3, spacecraft):
    def animate(self, i, MPC_traj1, features_plot, MPC_traj2, MPC_traj3, spacecraft):
        MPC_traj1 = self.update_trajectory(MPC_traj1, spacecraft[0], i, ref=False)
        MPC_traj2 = self.update_trajectory(MPC_traj2, spacecraft[1], i, ref=False)
        MPC_traj3 = self.update_trajectory(MPC_traj3, spacecraft[2], i, ref=False)

        feature_array = spacecraft[0].feature_array
        if i < len(feature_array):
            if feature_array[i].shape[0] > 0:
                features_seen = feature_array[i]
                features_plot.set_data(features_seen[:, 0], features_seen[:, 1])
                features_plot.set_3d_properties(features_seen[:, 2])

        return MPC_traj1, features_plot, MPC_traj2, MPC_traj3

    def plot_animation(self, asteroid, spacecraft):
        """Method to plot and show the MPC and reference trajectory animation with features seen"""

        fig9, ax9 = asteroid.plotpoly()

        # set up mothership
        MPC_traj1, = ax9.plot3D([], [], [], linewidth=4, zorder=6, color='purple')
        features_plot, = ax9.plot3D([], [], [], linestyle='None', marker='o', markersize=3, zorder=5, color='orange')

        # set up daughterships
        MPC_traj2, = ax9.plot3D([], [], [], linewidth=4, zorder=6, color='orange')
        MPC_traj3, = ax9.plot3D([], [], [], linewidth=4, zorder=6, color='green')

        ani = FuncAnimation(fig9, self.animate, frames=range(spacecraft[0].duration), fargs=[MPC_traj1, features_plot, MPC_traj2, MPC_traj3, spacecraft], blit=True, interval=100)

        ax9.legend([asteroid.title, 'Mothership true', 'Features Seen', 'Daughtership 1 true', 'Daughtership 2 true'], loc='upper left', prop={"size": 8})

        ax9.axis('equal')
        ax9.view_init(elev=25, azim=40)
        ax9.set_xlim([-4, 4])
        ax9.set_ylim([-4, 4])
        ax9.set_zlim([-4, 4])

        ani.save('output/MPC_and_Ref_Trajectory_Animation.gif')
        plt.show()


class ConstArgs(NamedTuple):
    s0: jnp.ndarray
    P0: jnp.ndarray
    dt_MPC_multiplier: jnp.ndarray
    mu: jnp.ndarray
    dt: jnp.ndarray
    exhaust_velocity: jnp.ndarray
    Q: jnp.ndarray
    max_segment_length: jnp.ndarray
    gates_sigma_1: jnp.ndarray
    gates_sigma_1_factor: jnp.ndarray
    r_0: jnp.ndarray
    C_R: jnp.ndarray
    surface_area: jnp.ndarray
    p1_1: jnp.ndarray
    p2_1: jnp.ndarray
    p1_2: jnp.ndarray
    p2_2: jnp.ndarray
    time_elapsed: jnp.ndarray
    times_grid: jnp.ndarray
    r_sun_grid: jnp.ndarray
    r_EMB_grid: jnp.ndarray
    r_jup_grid: jnp.ndarray
    offset: jnp.ndarray


def make_const_args(s, P, dt_MPC_multiplier, mu, dt, exhaust_velocity, Q, max_segment_length, gates_sigma_1, gates_sigma_1_factor, r_0, C_R, surface_area, p1_1, p2_1, p1_2, p2_2, time_elapsed, times_grid, r_sun_grid, r_EMB_grid, r_jup_grid, offset):
    return ConstArgs(
        s0=jnp.asarray(s, dtype=jnp.float32),
        P0=jnp.asarray(P, dtype=jnp.float32),
        dt_MPC_multiplier=jnp.asarray(dt_MPC_multiplier, dtype=jnp.float32),
        mu=jnp.asarray(mu, dtype=jnp.float32),
        dt=jnp.asarray(dt, dtype=jnp.float32),
        exhaust_velocity=jnp.asarray(exhaust_velocity, dtype=jnp.float32),
        Q=jnp.asarray(Q, dtype=jnp.float32),
        max_segment_length=jnp.asarray(max_segment_length, dtype=jnp.float32),
        gates_sigma_1=jnp.asarray(gates_sigma_1, dtype=jnp.float32),
        gates_sigma_1_factor=jnp.asarray(gates_sigma_1_factor, dtype=jnp.float32),
        r_0=jnp.asarray(r_0, dtype=jnp.float32),
        C_R=jnp.asarray(C_R, dtype=jnp.float32),
        surface_area=jnp.asarray(surface_area, dtype=jnp.float32),
        p1_1=jnp.asarray(p1_1, dtype=jnp.float32),
        p2_1=jnp.asarray(p2_1, dtype=jnp.float32),
        p1_2=jnp.asarray(p1_2, dtype=jnp.float32),
        p2_2=jnp.asarray(p2_2, dtype=jnp.float32),
        time_elapsed=jnp.asarray(time_elapsed, dtype=jnp.float32),
        times_grid=jnp.asarray(times_grid, dtype=jnp.float32),
        r_sun_grid=jnp.asarray(r_sun_grid, dtype=jnp.float32),
        r_EMB_grid=jnp.asarray(r_EMB_grid, dtype=jnp.float32),
        r_jup_grid=jnp.asarray(r_jup_grid, dtype=jnp.float32),
        offset=jnp.asarray(offset, dtype=jnp.float32),
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def cost_and_grad_jitted(dv, const_args):
    # dv: jnp.array with shape (H, 2, 3)
    tot, grad = jax.value_and_grad(cost_total_only_static, argnums=0)(dv, const_args)
    return tot, grad


@jax.jit
def cost_breakdown_jitted(dv, const_args):
    return cost_daughterships_jax_static(dv, const_args)

def safe_norm_jax_static(x, eps=1e-20):
    return jnp.sqrt(jnp.sum(x ** 2) + eps)


def get_rotation_jax_static(n_sun):
    # z-axis
    z = n_sun

    # reference vector not parallel to z
    r = jnp.where(jnp.abs(z[0]) < 0.9,
                  jnp.array([1.0, 0.0, 0.0]),
                  jnp.array([0.0, 1.0, 0.0]))

    # x and y-axes
    x = jnp.cross(r, z)
    x /= safe_norm_jax_static(x, 1e-20)
    y = jnp.cross(z, x)

    # rotation matrix
    R = jnp.vstack((x, y, z))
    return R


def rk4_integrate_static(f, s0, times_grid, r_sun_grid, r_EMB_grid, r_jup_grid, *args):
    N = times_grid.shape[0]
    dt = times_grid[1] - times_grid[0]

    def body_fun(i, s):
        t = times_grid[i]
        r_sun = r_sun_grid[:, i]
        r_EMB = r_EMB_grid[:, i]
        r_jup = r_jup_grid[:, i]

        k1 = f(s, t, r_sun, r_EMB, r_jup, *args)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt, r_sun, r_EMB, r_jup, *args)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt, r_sun, r_EMB, r_jup, *args)
        k4 = f(s + dt * k3, t + dt, r_sun, r_EMB, r_jup, *args)

        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.lax.fori_loop(0, N, body_fun, s0)


def ode_hor_jax_static(s, t, r_sun_to_apophis, r_sun_to_EMB, r_sun_to_jup, A, mu, C_R, surface_area):
    # Gravitational term
    radius = jnp.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) + 1e-20
    grav = -mu / radius ** 3
    A_mod = A + jnp.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [grav, 0, 0, 0, 0, 0, 0],
        [0, grav, 0, 0, 0, 0, 0],
        [0, 0, grav, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    # SRP and third-body accelerations
    C_SRP = solar_rad_pressure_jax(s, r_sun_to_apophis, C_R, surface_area)
    C_3rd_sun = third_body_effect_jax(s, mu=1.327e11, r_third_body=-r_sun_to_apophis)
    C_3rd_EMB = third_body_effect_jax(s, mu=3.986e5 + 4.902e3, r_third_body=-r_sun_to_apophis + r_sun_to_EMB)
    C_3rd_jup = third_body_effect_jax(s, mu=1.26686534e8, r_third_body=-r_sun_to_apophis + r_sun_to_jup)

    return jnp.matmul(A_mod, s) + C_SRP + C_3rd_sun + C_3rd_EMB + C_3rd_jup


def get_RTN_to_inertial_jax(r, v):
    R = r / jnp.linalg.norm(r)
    h = jnp.cross(r, v)
    N = h / jnp.linalg.norm(h)
    T = jnp.cross(N, R)
    return jnp.stack([R, T, N], axis=0)   # 3×3


def solar_rad_pressure_jax(state, r_sun_to_body, C_R, surface_area):
    """Method computes acceleration from SRP for JAX

    Args:
        state [7]: vector containing the state from the central body to the spacecraft
        r_sun_to_body [3]: vector from the sun to the central body

    Returns:
        [7] vector of the SRP acceleration
    """
    r_sun_sc = (state[0:3] + r_sun_to_body)*1000.0  # sun to s/c, meters
    r_sun_sc_mag = safe_norm_jax_static(r_sun_sc, 1e-20)
    r_sun_sc_norm = r_sun_sc / r_sun_sc_mag
    r_sun_sc_AU = r_sun_sc_mag / 1.495978707e11
    a_SRP = 4.56*10**-6 * C_R * surface_area / jnp.exp(state[6]) * (1 / r_sun_sc_AU)**2 * r_sun_sc_norm
    return jnp.concatenate([jnp.zeros(3), a_SRP, jnp.array([0.0])]) / 1000.0


def third_body_effect_jax(state, mu, r_third_body):
    """Method computes acceleration from third body effects for JAX

    Args:
        state: vector containing the state from the central body to the spacecraft
        mu: gravitational parameter of the third body
        r_third_body: position vector from the central body to the third body

    Returns:
        [7] vector of the 3rd body acceleration
    """
    r_third_minus_sc = r_third_body - state[0:3]
    r_third_minus_sc_norm = safe_norm_jax_static(r_third_minus_sc, 1e-20)
    r_third_norm = safe_norm_jax_static(r_third_body, 1e-20)
    a_3rd = mu * (r_third_minus_sc / r_third_minus_sc_norm ** 3 - r_third_body / r_third_norm ** 3)
    return jnp.concatenate([jnp.zeros(3), a_3rd, jnp.array([0.0])])

def cost_daughterships_jax_static(dv, const_args):
    """Method contains fully JAX-ified cost function for use in the MPC algorithm

    Args:
        dv: [H,2,3] tensor of the deltaV decision vector. Dimension 0 is step in the prediction horizon, dimension 1 is which daughtership, dimension 2 is x,y,z axis.
        const_args: instance of the ConstArgs class

    Returns:
        dictionary of cost values:
        "total": tot,
        "pos_cost": final["pos_cost"],
        "dot_cost": final["dot_cost"],
        "dv_cost_1": final["dv_cost_1"],
        "dv_cost_2": final["dv_cost_2"],
        "a_cost": final["a_cost"],
        "e_cost": final["e_cost"],
        "i_cost": final["i_cost"]
    """
    s0 = const_args.s0
    P0 = const_args.P0
    dt_MPC_multiplier = const_args.dt_MPC_multiplier
    mu = const_args.mu
    dt = const_args.dt
    exhaust_velocity = const_args.exhaust_velocity
    Q = const_args.Q
    max_segment_length = const_args.max_segment_length
    gates_sigma_1 = const_args.gates_sigma_1
    r_0 = const_args.r_0
    C_R = const_args.C_R
    surface_area = const_args.surface_area
    p1_1 = const_args.p1_1
    p2_1 = const_args.p2_1
    p1_2 = const_args.p1_2
    p2_2 = const_args.p2_2
    time_elapsed = const_args.time_elapsed
    times_grid = const_args.times_grid
    r_sun_grid = const_args.r_sun_grid
    r_EMB_grid = const_args.r_EMB_grid
    r_jup_grid = const_args.r_jup_grid
    gates_sigma_1_factor = const_args.gates_sigma_1_factor
    num_states = 7
    H = dv.shape[0]
    offset = const_args.offset
    dt_MPC_multiplier_int = 8

    eps = 1e-20

    p1_list = jnp.stack([p1_1, p1_2])
    p2_list = jnp.stack([p2_1, p2_2])

    A = jnp.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    # set deltaV minimum threshold purely for scaling purposes, not a constraint!
    dv_min = gates_sigma_1_factor * gates_sigma_1

    def per_ship_dv(dv_ship, exvel):
        """Internal method that extracts the individual prediction horizon epoch's deltaV vector and its magnitude

        Args:
            dv_ship: [2,3] array containing both daughtership's current epoch deltaV
            exvel: [2] vector containing daughtership exhaust velocities

        Returns:
            dv_vec: [2,7] vector containing the change in velocity and mass from the deltaV
            dv_norm: [2] norms of the deltaV's for each daughtership
        """
        dv_norm = safe_norm_jax_static(dv_ship, eps)
        dv_vec = jnp.array([0., 0., 0., dv_ship[0], dv_ship[1], dv_ship[2], -dv_norm / exvel])
        return dv_vec, dv_norm

    def propagate_cov(s_j, P_j, Q_j):
        """Internal method that propagates daughtership state covariance

        Args:
            s_j: [2,7] array containing both daughtership's current epoch states
            P_j: [2,7,7] tensor containing daughtership current covariance [7,7] arrays
            Q_j: [2,7,7] tensor containing the daughtersihp current process noise [7,7] arrays

        Returns:
            P_new: [2,7,7] updated covariance arrays
        """
        r = s_j[:3]
        rmag = safe_norm_jax_static(r, eps)
        gg = mu / rmag ** 3 * (3 * jnp.outer(r, r) / rmag ** 2 - jnp.eye(3))
        dCdx = jnp.pad(gg, ((3, 1), (0, 4)))
        F = jnp.eye(num_states) + (A + dCdx) * dt * dt_MPC_multiplier
        R_RTN_to_I = get_RTN_to_inertial_jax(r, s_j[3:6])
        Q_tilde = R_RTN_to_I @ Q_j @ R_RTN_to_I.T
        S_i = jnp.block([[dt ** 3 / 3 * Q_tilde, dt ** 2 / 2 * Q_tilde, jnp.zeros([3, 1])],
                         [dt ** 2 / 2 * Q_tilde, dt * Q_tilde, jnp.zeros([3, 1])],
                         [jnp.zeros([1, 3]), jnp.zeros([1, 3]), 1e-20]])
        return F @ P_j @ F.T + S_i

    def orbital_elements_per_ship(s_j, R, offset):
        """Internal method that computes the orbital elements for each daughtership

        Args:
            s_j: [2,7] array containing both daughtership's current epoch states
            R: [3,3] rotation matrix to go from inertial to local terminator frame
            offset: [3] inertial frame vector for how far off terminator plane the halo orbit should be

        Returns:
            a: [1] semimajor axis (km)
            e: [1] eccentricity
            i: [1] inclination (rad)
            r_vec: [3] terminator frame position vector for the daughtership
        """
        r_vec = R @ s_j[0:3]
        v_vec = R @ s_j[3:6]

        offset_term = R @ offset

        r_rel = r_vec - offset_term

        rmag = safe_norm_jax_static(r_rel, eps)
        vmag = safe_norm_jax_static(v_vec, eps)

        h = jnp.cross(r_rel, v_vec)
        den = 2 / rmag - vmag ** 2 / mu
        den = jnp.where(jnp.abs(den) < eps, eps, den)

        a = 1.0 / den
        e = safe_norm_jax_static(jnp.cross(v_vec, h) / mu - r_rel / rmag, eps)

        hx, hy, hz = h
        sin_i = safe_norm_jax_static(jnp.array([hx, hy]), eps)
        i = jnp.arctan2(sin_i, hz)

        return a, e, i, r_vec

    def pos_cost_per_ship(r_vec, p1, p2, max_len, R):
        """Internal method that computes the position cost for the MPC

        Args:
            r_vec: [2,7] array containing both daughtership's current epoch states with offset removed
            p1: array of reference trajectory starting vertices
            p2: array of reference trajectory ending vertices
            max_len: [2] scalar max segment lengths for each daughtership's reference trajectory
            R: [3,3] rotation matrix from inertial to local terminator frame

        Returns:
            pos_cost: [2] position cost for each daughtership
            ref_idx: [2] what index of the daughtership reference trajectory they are nearest to
        """
        _, dmin, _, _, ref_idx = MinDisSegmentPoint_Vectorized_jax(p1, p2, R @ r_vec)
        return dmin / max_len, ref_idx

    carry0 = {
        "s": s0,
        "P": P0,
        "pos_cost": 0.,
        "dot_cost": 0.,
        "dv_cost_1": 0.,
        "dv_cost_2": 0.,
        "a_cost": 0.,
        "e_cost": 0.,
        "i_cost": 0.,
    }

    def step_fn(carry, i):
        """Internal method computes states, covariances, and accrued costs over the prediction horizon in the MPC

        Args:
            carry: dictionary containing current states, covariances, and currently accrued MPC costs
            i: epoch in the prediction horizon

        Returns:
            new_carry: updated carry dictionary
        """
        s = carry["s"]
        P = carry["P"]
        dv_i = dv[i]

        N_step = 5 * dt_MPC_multiplier_int
        start = i * N_step

        times_step = lax.dynamic_slice(times_grid, (start,), (N_step,))
        r_sun_step = lax.dynamic_slice(r_sun_grid, (0, start), (3, N_step))
        r_EMB_step = lax.dynamic_slice(r_EMB_grid, (0, start), (3, N_step))
        r_jup_step = lax.dynamic_slice(r_jup_grid, (0, start), (3, N_step))

        deltav, deltav_mag = vmap(per_ship_dv)(dv_i, exhaust_velocity)
        P_new = vmap(propagate_cov)(s.T, P.transpose(2, 0, 1), Q).transpose(1, 2, 0)

        def propagate_state(s_j, dv_j, CR_j, SA_j):
            """Even more internal method that propagates the current state to the next state based on the deltaV

            Args:
                r_j: [2,7] array containing both daughtership's current epoch states with offset removed
                dv_j: [2,3] current epoch's deltaV commands
                CR_j: [2] daughtership coefficient of reflection
                SA_j: [2] daughtership surface areas
            """
            return rk4_integrate_static(
                ode_hor_jax_static,
                s_j + dv_j,
                times_step,
                r_sun_step,
                r_EMB_step,
                r_jup_step,
                A, mu, CR_j, SA_j)

        s_new = vmap(propagate_state)(s.T, deltav, C_R, surface_area).T

        dv_cost_1, dv_cost_2 = (deltav_mag / dv_min) ** 2 * (((deltav_mag / dv_min) - 1) ** 2 + 1)

        time_passed = time_elapsed * dt + i * dt * dt_MPC_multiplier
        idx = jnp.argmin(jnp.abs(times_grid - time_passed))
        n_sun = r_sun_grid[:, idx]
        n_sun = n_sun / safe_norm_jax_static(n_sun, eps)
        R = get_rotation_jax_static(n_sun)

        pos_costs, ref_idx = vmap(pos_cost_per_ship, in_axes=(1, 0, 0, 0, None))(s_new[0:3], p1_list, p2_list, max_segment_length, R)
        pos_cost = jnp.sum(pos_costs) / dt_MPC_multiplier

        a_vals, e_vals, i_vals, r_vecs = vmap(orbital_elements_per_ship, in_axes=(1, None, 1))(s_new, R, offset)

        a_cost = ((a_vals[0] - a_vals[1]) / (0.1 * r_0)) ** 2
        e_cost = 10 * e_vals[0] ** 2 + 10 * e_vals[1] ** 2
        i_cost = 0.5*(i_vals[0] + i_vals[1])*180/jnp.pi/10

        avg_offset = R @ (0.5 * (offset[:, 0] + offset[:, 1]))
        d1 = r_vecs[0] - avg_offset
        d2 = r_vecs[1] - avg_offset

        u1 = d1 / safe_norm_jax_static(d1, eps)
        u2 = d2 / safe_norm_jax_static(d2, eps)

        cross_norm = safe_norm_jax_static(jnp.cross(u1, u2), eps)
        dot_val = jnp.dot(u1, u2)
        angle = jnp.arctan2(cross_norm, dot_val)

        dot_cost = (jnp.pi - angle) / (5.0 * jnp.pi / 180)

        new_carry = {
            "s": s_new,
            "P": P_new,
            "pos_cost": carry["pos_cost"] + pos_cost/H,
            "dot_cost": carry["dot_cost"] + dot_cost/H,
            "dv_cost_1": carry["dv_cost_1"] + dv_cost_1/H,
            "dv_cost_2": carry["dv_cost_2"] + dv_cost_2/H,
            "a_cost": carry["a_cost"] + a_cost/H,
            "e_cost": carry["e_cost"] + e_cost/H,
            "i_cost": carry["i_cost"] + i_cost/H,
        }

        return new_carry, None

    final, _ = lax.scan(step_fn, carry0, jnp.arange(H))

    tot = (
        final["pos_cost"]
        + final["dot_cost"]
        + final["dv_cost_1"]
        + final["dv_cost_2"]
        + final["a_cost"]
        + final["e_cost"]
        + final["i_cost"]
    )

    return {
        "total": tot,
        "pos_cost": final["pos_cost"],
        "dot_cost": final["dot_cost"],
        "dv_cost_1": final["dv_cost_1"],
        "dv_cost_2": final["dv_cost_2"],
        "a_cost": final["a_cost"],
        "e_cost": final["e_cost"],
        "i_cost": final["i_cost"],
    }


def cost_total_only_static(dv_scaled, const_args: ConstArgs):
    out = cost_daughterships_jax_static(dv_scaled, const_args)
    return out["total"]


def main():
    # sets up the problem
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    num_spacecraft = 3
    spacecraft = [Spacecraft(asteroid, config_file, None, 0)]
    if num_spacecraft > 1:
        for i in range(num_spacecraft-1):
            spacecraft.append(Spacecraft(asteroid, config_file, None, i+1))

    simulation = Simulation(asteroid, spacecraft, config_file, None)
    starttime = datetime.now()

    mpc = MPC(simulation, asteroid, spacecraft)
    print("Run time: " + str(datetime.now() - starttime))
    mpc.plot_MPC_and_SCvx(simulation, asteroid, spacecraft)
    mpc.plot_animation(asteroid, spacecraft)


if __name__ == '__main__':
    main()

