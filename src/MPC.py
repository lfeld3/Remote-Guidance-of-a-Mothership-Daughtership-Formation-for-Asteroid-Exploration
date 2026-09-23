# Polyhedral Gravity MPC Algorithm

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
            dv_flat: [6] vector of initial deltaV guess

        Returns:
            tot: total scalar cost from cost function
            grad_scaled: [6] vector of the jacobians of the scalar cost with respect to the deltaV components
        """
        scale = 1e-6
        scale_tile = jnp.float32([1, 1, scale])  # only magnitude scaled
        full_scaling = jnp.tile(scale_tile, 2)

        dv_phys_single = jnp.multiply(dv_flat, full_scaling).reshape(2, 3)
        tot, grad_phys = cost_and_grad_jitted(dv_phys_single, self.const_args)

        grad_scaled = jnp.multiply(grad_phys, full_scaling.reshape(2, 3))
        return float(tot), grad_scaled.ravel()

    def run_MPC(self, simulation, asteroid, spacecraft):
        """Method runs the model predictive control (MPC) algorithm!

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
        allow_D1 = False
        allow_D2 = False

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
        self.offset_hist = np.zeros([spacecraft[0].duration, (num_spacecraft-1)])

        self.horizon_dv_hist = np.zeros([longest_control_dim*(num_spacecraft-1), spacecraft[0].duration])

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
        times_grid, r_sun_grid, r_EMB_grid, r_jup_grid = self.make_grids(spacecraft[0], time_passed, time_passed + spacecraft[0].dt * simulation.dt_MPC_multiplier * spacecraft[0].horizon_steps, 5*int(simulation.dt_MPC_multiplier)*spacecraft[0].horizon_steps)

        self.const_args = make_const_args(s, P, dt_MPC_multiplier, mu, dt, exhaust_velocity, Q, max_segment_length, gates_sigma_1, gates_sigma_1_factor, r_0, C_R, surface_area, p1_1, p2_1, p1_2, p2_2, time_elapsed, times_grid, r_sun_grid, r_EMB_grid, r_jup_grid)

        control_guess_length = 0
        for i in [1, 2]:
            control_guess_length += (spacecraft[i].num_controls - 1)

        delta_v_ig = jnp.zeros(control_guess_length)

        self.jacobian_hist = np.zeros([spacecraft[0].duration, control_guess_length])

        scale = 1e-6
        scale_tile = jnp.float32([1, 1, scale])
        full_scaling = jnp.tile(scale_tile, 2)

        dv_min = spacecraft[0].gates_sigma_1_factor * spacecraft[0].gates_sigma_1 / scale
        dv_max = None

        bounds_temp = [(0.0, 2 * math.pi - 1e-4), (0.0, math.pi), (dv_min, dv_max)]  # azimuth, zenith, magnitude
        bounds_both = bounds_temp * 2
        bounds_D1 = [(0.0, 2 * math.pi - 1e-4), (0.0, math.pi), (dv_min, dv_max), (0.0, 2 * math.pi - 1e-4), (0.0, math.pi), (0.0, 0.0)]
        bounds_D2 = [(0.0, 2 * math.pi - 1e-4), (0.0, math.pi), (0.0, 0.0), (0.0, 2 * math.pi - 1e-4), (0.0, math.pi), (dv_min, dv_max)]
        print(bounds_both)
        print(bounds_D1)
        print(bounds_D2)

        print("============================================================================")

        while duration > time_elapsed + 1:

            # generate initial guess
            control_guess_length = 0
            delta_v_ig = []
            for i in [1, 2]:
                control_guess_length += (spacecraft[i].num_controls - 1)
                current_velocity = np.copy(spacecraft[i].shat[3:6])
                azimuth = np.arctan2(current_velocity[1], current_velocity[0])
                zenith = np.arccos(current_velocity[2]/np.linalg.norm(current_velocity))
                current_velocity_dir = np.array([azimuth, zenith, 0.0])
                delta_v_ig = np.hstack((delta_v_ig, current_velocity_dir))

            x0 = jnp.copy(jnp.float32(delta_v_ig))

            # update states, covariances for current MPC time step/iteration
            s = np.array([np.copy(spacecraft[1].shat), np.copy(spacecraft[2].shat)]).T
            P = np.zeros([7, 7, 2])
            P[:, :, 0] = np.copy(spacecraft[1].P)
            P[:, :, 1] = np.copy(spacecraft[2].P)

            # update full JAX list of params/values
            time_passed = time_elapsed * spacecraft[0].dt
            times_grid, r_sun_grid, r_EMB_grid, r_jup_grid = self.make_grids(spacecraft[0], time_passed, time_passed + spacecraft[0].dt * simulation.dt_MPC_multiplier * spacecraft[0].horizon_steps, 5 * int(simulation.dt_MPC_multiplier) * spacecraft[0].horizon_steps)
            self.const_args = self.const_args._replace(s0=jnp.array(s), P0=jnp.array(P), time_elapsed=jnp.array(time_elapsed), times_grid=times_grid, r_sun_grid=r_sun_grid, r_EMB_grid=r_EMB_grid, r_jup_grid=r_jup_grid)

            # ============================================================
            # case 1: zero burn case
            # ============================================================
            dv_zero = jnp.zeros((2, 3), dtype=jnp.float32)
            J0_costs = cost_breakdown_jitted(dv_zero, self.const_args)
            J0 = J0_costs["total"]

            dv_zero_flat = dv_zero.reshape(-1)
            _, J0_jac = self.scipy_fun(dv_zero_flat)

            scenario_results = {
                "zero": {
                    "cost": J0,
                    "costs": J0_costs,
                    "dv": dv_zero,
                    "sol": None,
                    "jac": J0_jac,
                }
            }

            print("==================")
            print("NO DELTAV COST")
            print(J0_costs)

            # ============================================================
            # cases 2-4, burn cases
            # ============================================================
            if allow_D1 or allow_D2:
                # ============================================================
                # case 4: both daughterships burn
                # ============================================================
                if allow_D1 and allow_D2:
                    sol_both = fmin_l_bfgs_b(func=lambda x: self.scipy_fun(x)[0],
                                             x0=x0, fprime=lambda x: self.scipy_fun(x)[1],
                                             maxiter=10000, factr=0.01, pgtol=1e-8,
                                             bounds=bounds_both, m=30, maxls=50)

                    dv_both = jnp.multiply(jnp.float32(sol_both[0]), full_scaling).reshape(2, 3)
                    Jboth_costs = cost_breakdown_jitted(dv_both, self.const_args)
                    Jboth = Jboth_costs["total"]

                    print("==================")
                    print("BOTH DELTAV COST")
                    print(Jboth_costs)
                    print("==================")

                    scenario_results["both"] = {
                        "cost": Jboth,
                        "costs": Jboth_costs,
                        "dv": dv_both,
                        "sol": sol_both,
                        "jac": sol_both[2]["grad"],
                    }
                else:
                    scenario_results["both"] = {"cost": 1e10}

                # ============================================================
                # case 2: only daughtership 1 burns
                # ============================================================
                if allow_D1:
                    sol_D1 = fmin_l_bfgs_b(func=lambda x: self.scipy_fun(x)[0],
                                           x0=x0, fprime=lambda x: self.scipy_fun(x)[1],
                                           maxiter=10000, factr=0.01, pgtol=1e-8,
                                           bounds=bounds_D1, m=30, maxls=50)

                    dv_D1 = jnp.multiply(jnp.float32(sol_D1[0]), full_scaling).reshape(2, 3)
                    J1_costs = cost_breakdown_jitted(dv_D1, self.const_args)
                    J1 = J1_costs["total"]

                    print("==================")
                    print("D1 DELTAV COST")
                    print(J1_costs)
                    print("==================")

                    scenario_results["D1"] = {
                        "cost": J1,
                        "costs": J1_costs,
                        "dv": dv_D1,
                        "sol": sol_D1,
                        "jac": sol_D1[2]["grad"],
                    }
                else:
                    scenario_results["D1"] = {"cost": 1e10}

                # ============================================================
                # case 3: only daughtership 2 burns
                # ============================================================
                if allow_D2:
                    sol_D2 = fmin_l_bfgs_b(func=lambda x: self.scipy_fun(x)[0],
                                           x0=x0, fprime=lambda x: self.scipy_fun(x)[1],
                                           maxiter=10000, factr=0.01, pgtol=1e-8,
                                           bounds=bounds_D2, m=30, maxls=50)

                    dv_D2 = jnp.multiply(jnp.float32(sol_D2[0]), full_scaling).reshape(2, 3)
                    J2_costs = cost_breakdown_jitted(dv_D2, self.const_args)
                    J2 = J2_costs["total"]

                    print("==================")
                    print("D2 DELTAV COST")
                    print(J2_costs)
                    print("==================")

                    scenario_results["D2"] = {
                        "cost": J2,
                        "costs": J2_costs,
                        "dv": dv_D2,
                        "sol": sol_D2,
                        "jac": sol_D2[2]["grad"],
                    }
                else:
                    scenario_results["D2"] = {"cost": 1e10}

            else:
                # if a burn already happened, all burn scenarios are invalid
                scenario_results["both"] = {"cost": 1e10}
                scenario_results["D1"] = {"cost": 1e10}
                scenario_results["D2"] = {"cost": 1e10}

            # pick the lowest cost case/scenario
            best_key = min(scenario_results.keys(), key=lambda k: scenario_results[k]["cost"])
            best = scenario_results[best_key]

            dv_applied_angles = best["dv"]
            costs = best["costs"]
            total_cost = best["cost"]
            sol = best["sol"]
            jac = best["jac"]

            # update allow_D1 / allow_D2 based on which daughtership have thrusted this BLS window
            if best_key == "D1":
                allow_D1 = False
            elif best_key == "D2":
                allow_D2 = False
            elif best_key == "both":
                allow_D1 = False
                allow_D2 = False

            print("Chosen scenario:", best_key)
            print("sol:", sol)

            self.horizon_dv_hist[:, time_elapsed] = dv_applied_angles.ravel()  # commanded deltaV from prediction horizon

            print('Computed minimum cost function value = ' + str(total_cost))
            self.cost_hist[time_elapsed] = total_cost

            # get current sun direction
            n_sun_to_apophis = np.array([spacecraft[0].spline_sun_x(time_elapsed * spacecraft[0].dt),
                                         spacecraft[0].spline_sun_y(time_elapsed * spacecraft[0].dt),
                                         spacecraft[0].spline_sun_z(time_elapsed * spacecraft[0].dt)])
            n_sun_to_apophis /= np.linalg.norm(n_sun_to_apophis)
            print(time_elapsed * spacecraft[0].dt)
            print(n_sun_to_apophis)
            R = self.get_rotation(n_sun_to_apophis)
            dot_vec_d1 = np.matmul(R, np.copy(spacecraft[1].s[0:3]))
            dot_vec_d2 = np.matmul(R, np.copy(spacecraft[2].s[0:3]))
            dot_vec_d1_hat = np.matmul(R, np.copy(spacecraft[1].shat[0:3]))
            dot_vec_d2_hat = np.matmul(R, np.copy(spacecraft[2].shat[0:3]))

            # print out cost by term, store cost into cost histories, compute antipodal true and estimated angles
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
                print('offsets')
                print(costs["offsets"])
                print('jacobians')
                print(jac)
                print('-----------------------------')
                self.pos_cost_hist[time_elapsed] = costs["pos_cost"]
                self.dot_cost_hist[time_elapsed] = costs["dot_cost"]
                self.dv_cost_1_hist[time_elapsed] = costs["dv_cost_1"]
                self.dv_cost_2_hist[time_elapsed] = costs["dv_cost_2"]
                self.a_cost_hist[time_elapsed] = costs["a_cost"]
                self.e_cost_hist[time_elapsed] = costs["e_cost"]
                self.i_cost_hist[time_elapsed] = costs["i_cost"]
                self.jacobian_hist[time_elapsed, :] = jac.flatten()  # C order
                self.offset_hist[time_elapsed, :] = costs["offsets"][:, 0]
                offset_d1 = np.array([0, 0, np.copy(costs["offsets"][0, 0])])
                offset_d2 = np.array([0, 0, np.copy(costs["offsets"][1, 0])])
                print((dot_vec_d1 - offset_d1) / np.linalg.norm(dot_vec_d1 - offset_d1))
                print((dot_vec_d2 - offset_d2) / np.linalg.norm(dot_vec_d2 - offset_d2))
                print('true bistatic angle (deg)')
                bistatic_true = 180 / math.pi * np.arccos(
                    np.dot((dot_vec_d1 - offset_d1) / np.linalg.norm(dot_vec_d1 - offset_d1),
                           (dot_vec_d2 - offset_d2) / np.linalg.norm(dot_vec_d2 - offset_d2)))
                print(bistatic_true)
                self.bistatic_angle_true[time_elapsed] = bistatic_true
                print('estimated bistatic angle (deg)')
                bistatic_hat = 180 / math.pi * np.arccos(
                    np.dot((dot_vec_d1_hat - offset_d1) / np.linalg.norm(dot_vec_d1_hat - offset_d1),
                           (dot_vec_d2_hat - offset_d2) / np.linalg.norm(dot_vec_d2_hat - offset_d2)))
                print(bistatic_hat)
                self.bistatic_angle_hat[time_elapsed] = bistatic_hat
            else:
                self.pos_cost_hist[time_elapsed] = 0
                self.dot_cost_hist[time_elapsed] = 0
                self.dv_cost_1_hist[time_elapsed] = 0
                self.dv_cost_2_hist[time_elapsed] = 0
                self.a_cost_hist[time_elapsed] = 0
                self.e_cost_hist[time_elapsed] = 0
                self.i_cost_hist[time_elapsed] = 0
                self.jacobian_hist[time_elapsed, :] = np.zeros(control_guess_length)
                offset_d1 = np.matmul(R, np.copy(offset[:, 0]))
                offset_d2 = np.matmul(R, np.copy(offset[:, 1]))
                offset_avg = 0.5 * (offset_d1 + offset_d2)
                print(offset_avg)
                print((dot_vec_d1 - offset_avg) / np.linalg.norm(dot_vec_d1 - offset_avg))
                print((dot_vec_d2 - offset_avg) / np.linalg.norm(dot_vec_d2 - offset_avg))
                print('true bistatic angle (deg)')
                bistatic_true = 180 / math.pi * np.arccos(
                    np.dot((dot_vec_d1 - offset_avg) / np.linalg.norm(dot_vec_d1 - offset_avg),
                           (dot_vec_d2 - offset_avg) / np.linalg.norm(dot_vec_d2 - offset_avg)))
                print(bistatic_true)
                self.bistatic_angle_true[time_elapsed] = bistatic_true
                print('estimated bistatic angle (deg)')
                bistatic_hat = 180 / math.pi * np.arccos(
                    np.dot((dot_vec_d1_hat - offset_avg) / np.linalg.norm(dot_vec_d1_hat - offset_avg),
                           (dot_vec_d2_hat - offset_avg) / np.linalg.norm(dot_vec_d2_hat - offset_avg)))
                print(bistatic_hat)
                self.bistatic_angle_hat[time_elapsed] = bistatic_hat

            print('DeltaV (mm/s)')
            print(dv_applied_angles[0, 2]/scale)
            print(dv_applied_angles[1, 2]/scale)

            obs_count = time_elapsed + 2
            if obs_count <= spacecraft[0].N_batch:
                iter_in_batch = obs_count - 1
            else:
                iter_in_batch = (obs_count - spacecraft[0].N_batch) % (spacecraft[0].N_batch - 1)

            print('iter in batch')
            print(iter_in_batch)
            print('----')
            for i in [1, 2]:
                dv_angles = np.array(dv_applied_angles[i - 1])  # azimuth, zenith, magnitude

                if dv_angles[2] <= 1e-12:  # if there's no deltaV magnitude
                    # no burn case
                    delta_v_opt = np.zeros(spacecraft[i].num_controls - 1)
                else:   # rotate deltaVs into x, y, z inertial frame
                    dv_x = dv_angles[2] * math.sin(dv_angles[1]) * math.cos(dv_angles[0])
                    dv_y = dv_angles[2] * math.sin(dv_angles[1]) * math.sin(dv_angles[0])
                    dv_z = dv_angles[2] * math.cos(dv_angles[1])
                    delta_v_opt = [dv_x, dv_y, dv_z]
                    print('HERE')
                    print(delta_v_opt)

                # cap thrusters if they go above thrust max or dont burn if out of fuel
                delta_v_opt = self.max_vec(np.copy(delta_v_opt), spacecraft[i].thrust_max / np.exp(spacecraft[i].s[6]) * 0.0005)
                delta_v_opt = self.zero_vec(np.copy(delta_v_opt), np.exp(spacecraft[i].s[6]), spacecraft[i].dry_mass)
                print('dv mag')
                print(np.linalg.norm(delta_v_opt)/scale)
                print('MPC Optimal Delta V for ' + spacecraft[i].designation + ' the timestep (Delta V_X, V_Y, V_Z, mm/s) = ' + str(delta_v_opt*10**6))
                spacecraft[i].delta_v = np.copy(delta_v_opt)
                spacecraft[i].delta_v_hist[:, time_elapsed+1] = spacecraft[i].delta_v
                if np.linalg.norm(delta_v_opt) > 0:
                    spacecraft[i].delta_v_num_in_batch[time_elapsed+1] = iter_in_batch

            offset_m = spacecraft[0].get_local_offset(asteroid, time_elapsed * dt, spacecraft[0].s[0:3])
            offset_list = (np.array([0, 0, offset_m]), offset_d1, offset_d2)
            for i in range(len(spacecraft)):
                # propagate states with deltaV commands and dynamics
                true_accel = spacecraft[i].state_prop(asteroid, time_elapsed)
                pred_accel = -asteroid.mu_grav_param / np.linalg.norm(spacecraft[i].s[0:3]) ** 3 * spacecraft[i].s[0:3]

                # update mothership state with EKF
                EKF = ekf.EKF(time_elapsed)
                seen = EKF.run_EKF(simulation, asteroid, spacecraft[i])
                print('observations total:')
                print(time_elapsed+2)
                print('Estimated State (pre BLS):')
                print(spacecraft[i].shat)

                # if enough new obs for BLS run, run BLS
                if obs_count >= spacecraft[i].N_batch:
                    if (obs_count - spacecraft[i].N_batch) % (spacecraft[i].N_batch - 1) == 0:
                        if spacecraft[i].designation != 'mothership':
                            leftover = 0
                            bls = BLS(simulation, asteroid, spacecraft[i], mothership, time_elapsed, leftover)
                            has_done_a_batch = True  # comment this out to only propagate dynamics, no thrust
                            allow_D1 = True
                            allow_D2 = True

                    # if performing a BLS run will leave less than N_BLS number of remaining obs before the end of the sim, wait to perform BLS until end of sim then do a large BLS
                    elif obs_count == duration:
                        if spacecraft[i].designation != 'mothership':
                            leftover = (obs_count - spacecraft[i].N_batch) % (spacecraft[i].N_batch - 1)
                            bls = BLS(simulation, asteroid, spacecraft[i], mothership, time_elapsed, leftover)

                if spacecraft[i].designation == 'mothership':
                    spacecraft[i].feature_array.append(seen)
                    print('Features in view: ' + str(seen.shape[0]))

                # store new state estimates and true states, get min dist and vel to nearest ref traj segments
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
                r_shifted = R @ spacecraft[i].s[0:3] - offset_list[i]
                tout, dmin, p1s, p2s, sindex = MinDisSegmentPoint_Vectorized(spacecraft[i].p1, spacecraft[i].p2, r_shifted)
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
        savemat('output/MATLAB_mats/offset_hist.mat', {'myArray': self.offset_hist})
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
        # should make a new plot rotating everything into the local terminator plane and plotting there
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
    dt_MPC_multiplier: jnp.int64
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


def make_const_args(s, P, dt_MPC_multiplier, mu, dt, exhaust_velocity, Q, max_segment_length, gates_sigma_1, gates_sigma_1_factor, r_0, C_R, surface_area, p1_1, p2_1, p1_2, p2_2, time_elapsed, times_grid, r_sun_grid, r_EMB_grid, r_jup_grid):
    return ConstArgs(
        s0=jnp.asarray(s, dtype=jnp.float32),
        P0=jnp.asarray(P, dtype=jnp.float32),
        dt_MPC_multiplier=jnp.asarray(dt_MPC_multiplier, dtype=jnp.int32), # dont forget to change this to 32/64 also!
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
    )


@functools.partial(jax.jit, donate_argnums=(0,))
def cost_and_grad_jitted(dv, const_args):
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


def local_offset_jax(s_j, n_sun, t_i, r_sun_i, r_EMB_i, r_jup_i, A, mu, CR_j, SA_j, eps=1e-16):
    s_j_copy = jnp.copy(s_j)
    r0 = s_j_copy[0:3]

    def accel_at_pos(pos):
        # temporary state with updated position
        s_temp = jnp.array([pos[0], pos[1], pos[2], s_j_copy[3], s_j_copy[4], s_j_copy[5], s_j_copy[6]])
        deriv = ode_hor_jax_static(s_temp, t_i, r_sun_i, r_EMB_i, r_jup_i, A, mu, CR_j, SA_j)
        return deriv[3:6]  # acceleration components

    def f(s):
        pos = r0 + s * n_sun
        a = accel_at_pos(pos)
        return jnp.dot(a, n_sun)/jnp.linalg.norm(a)

    # same bracket as NumPy
    s_lo = -0.5
    s_hi = 0.5

    f_lo = f(s_lo)
    f_hi = f(s_hi)

    def body_fun(k, vals):
        s_lo, s_hi, f_lo, f_hi = vals

        s_mid = 0.5 * (s_lo + s_hi)
        f_mid = f(s_mid)

        # if f_lo * f_mid <= 0, root in [s_lo, s_mid]
        cond = f_lo * f_mid <= 0.0

        s_hi_new = jnp.where(cond, s_mid, s_hi)
        f_hi_new = jnp.where(cond, f_mid, f_hi)

        s_lo_new = jnp.where(cond, s_lo, s_mid)
        f_lo_new = jnp.where(cond, f_lo, f_mid)

        return (s_lo_new, s_hi_new, f_lo_new, f_hi_new)

    s_lo, s_hi, f_lo, f_hi = lax.fori_loop(0, 100, body_fun, (s_lo, s_hi, f_lo, f_hi))
    s_mid = 0.5 * (s_lo + s_hi)

    R = get_rotation_jax_static(n_sun)
    z_current = (R @ r0)[2]
    z_equilibrium = z_current + s_mid

    return z_equilibrium

def cost_daughterships_jax_static(dv, const_args):
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
    dt_MPC_multiplier_int = 8
    H = int(times_grid.shape[0]/5/dt_MPC_multiplier_int)

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

    dv_min = gates_sigma_1_factor * gates_sigma_1

    def per_ship_dv(dv_ship, exvel):
        # element 0 = azimuth, angle from x-axis in x-y plane. pi/2 = y-axis. theta=[0, 2*pi)
        # element 1 = zenith, angle from z-axis down. phi = [0, pi]
        # element 2 = radius/magnitude. dv >= 0
        dv_x = dv_ship[2] * jnp.sin(dv_ship[1]) * jnp.cos(dv_ship[0])
        dv_y = dv_ship[2] * jnp.sin(dv_ship[1]) * jnp.sin(dv_ship[0])
        dv_z = dv_ship[2] * jnp.cos(dv_ship[1])
        dv_norm = safe_norm_jax_static(dv_ship[2], eps)
        dv_vec = jnp.array([0., 0., 0., dv_x, dv_y, dv_z, -dv_norm / exvel])
        return dv_vec, dv_norm

    def propagate_cov(s_j, P_j, Q_j):
        r = s_j[:3]
        rmag = safe_norm_jax_static(r, eps)
        gg = mu / rmag ** 3 * (3 * jnp.outer(r, r) / rmag ** 2 - jnp.eye(3))
        dCdx = jnp.pad(gg, ((3, 1), (0, 4)))
        F = jnp.eye(num_states) + (A + dCdx) * dt * dt_MPC_multiplier
        R_RTN_to_I = get_RTN_to_inertial_jax(r, s_j[3:6])
        Q_tilde = R_RTN_to_I @ Q_j @ R_RTN_to_I.T
        S_i = jnp.block([[dt**3/3*Q_tilde, dt**2/2*Q_tilde, jnp.zeros([3,1])],
                         [dt**2/2*Q_tilde, dt*Q_tilde, jnp.zeros([3,1])],
                         [jnp.zeros([1, 3]), jnp.zeros([1, 3]), 1e-20]])
        return F @ P_j @ F.T + S_i

    def orbital_elements_per_ship(s_j, R, offset):
        r_vec = R @ s_j[0:3]
        v_vec = R @ s_j[3:6]

        r_rel = r_vec - jnp.array([0, 0, offset])

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

    def cov_cost_per_ship(P_j, S_j):
        return 0 * jnp.trace(S_j @ P_j)

    def pos_cost_per_ship(r_vec, p1, p2, max_len, R, offsets):
        pos = R @ r_vec - jnp.array([0, 0, offsets])
        _, dmin, _, _, ref_idx = MinDisSegmentPoint_Vectorized_jax(p1, p2, pos)
        return dmin / max_len, ref_idx

    # apply dv_single once at the start, then scan over H with no further control
    deltav, deltav_mag = vmap(per_ship_dv)(dv, exhaust_velocity)
    #jax.debug.print("deltaV = {x}", x=deltav[:, 3:6])

    w_dv1 = 0.5
    w_dv2 = 0.5
    only_D1 = (deltav_mag[0] >= dv_min/10) & (deltav_mag[1] < dv_min/10)
    only_D2 = (deltav_mag[1] >= dv_min/10) & (deltav_mag[0] < dv_min/10)

    w_dv1 = jnp.where(only_D1, 2.0 * w_dv1, w_dv1)
    w_dv2 = jnp.where(only_D2, 2.0 * w_dv2, w_dv2)

    dv_cost_1, dv_cost_2 = w_dv1*(deltav_mag[0]/dv_min)**2, w_dv2*(deltav_mag[1]/dv_min)**2  # (deltav_mag / dv_min) ** 2 * (((deltav_mag / dv_min) - 1) ** 2 + 1)  # 0.5*((deltav_mag - dv_min)/dv_min)**2  #

    carry0 = {
        "s": s0 + deltav.T,
        "P": P0,
        "pos_cost": 0.,
        "dot_cost": 0.,
        "dv_cost_1": dv_cost_1,
        "dv_cost_2": dv_cost_2,
        "a_cost": 0.,
        "e_cost": 0.,
        "i_cost": 0.,
        "offsets": jnp.zeros((2, H))
    }

    def step_fn(carry, i):
        s = carry["s"]
        P = carry["P"]

        P_new = vmap(propagate_cov)(s.T, P.transpose(2, 0, 1), Q).transpose(1, 2, 0)

        N_step = 5 * dt_MPC_multiplier_int
        start = i * N_step

        times_step = lax.dynamic_slice(times_grid, (start,), (N_step,))
        r_sun_step = lax.dynamic_slice(r_sun_grid, (0, start), (3, N_step))
        r_EMB_step = lax.dynamic_slice(r_EMB_grid, (0, start), (3, N_step))
        r_jup_step = lax.dynamic_slice(r_jup_grid, (0, start), (3, N_step))

        def propagate_state(s_j, CR_j, SA_j):
            return rk4_integrate_static(
                ode_hor_jax_static,
                s_j,
                times_step,
                r_sun_step,
                r_EMB_step,
                r_jup_step,
                A, mu, CR_j, SA_j)

        s_new = vmap(propagate_state)(s.T, C_R, surface_area).T

        time_passed = time_elapsed * dt + i * dt * dt_MPC_multiplier
        idx = jnp.argmin(jnp.abs(times_grid - time_passed))
        n_sun = r_sun_grid[:, idx]
        n_sun = n_sun / safe_norm_jax_static(n_sun, eps)
        R = get_rotation_jax_static(n_sun)

        #offsets = vmap(lambda s_j, CR_j, SA_j: local_offset_jax(s_j, n_sun, times_grid[idx], r_sun_grid[:, idx], r_EMB_grid[:, idx], r_jup_grid[:, idx], A, mu, CR_j, SA_j))(s_new.T, C_R, surface_area)
        offsets = vmap(local_offset_jax, in_axes=(1, None, None, None, None, None, None, None, 0, 0))(s_new, n_sun, times_grid[idx], r_sun_grid[:, idx], r_EMB_grid[:, idx], r_jup_grid[:, idx], A, mu, C_R, surface_area)

        pos_costs, ref_idx = vmap(pos_cost_per_ship, in_axes=(1, 0, 0, 0, None, 0))(s_new[0:3], p1_list, p2_list, max_segment_length, R, offsets)
        pos_cost = jnp.sum(pos_costs)

        a_vals, e_vals, i_vals, r_vecs = vmap(orbital_elements_per_ship, in_axes=(1, None, 0))(s_new, R, offsets)

        a_cost = 0*((a_vals[0] - a_vals[1]) / (0.1 * r_0)) ** 2
        e_cost = 0*(50 * e_vals[0] ** 2 + 50 * e_vals[1] ** 2)
        i_cost = 0.5*(i_vals[0] + i_vals[1])*180/jnp.pi/5

        #avg_offset = R @ (0.5 * (offset[:, 0] + offset[:, 1]))
        d1 = r_vecs[0] - jnp.array([0, 0, offsets[0]])
        d2 = r_vecs[1] - jnp.array([0, 0, offsets[1]])

        u1 = d1 / safe_norm_jax_static(d1, eps)
        u2 = d2 / safe_norm_jax_static(d2, eps)

        cross_norm = safe_norm_jax_static(jnp.cross(u1, u2), eps)
        dot_val = jnp.dot(u1, u2)
        angle = jnp.arctan2(cross_norm, dot_val)

        dot_cost = ((jnp.pi - angle) / (5.0 * jnp.pi / 180))

        # cov_terms = vmap(cov_cost_per_ship, in_axes=(2, 0))(P_new, S)
        # cov_cost = jnp.sum(cov_terms) / (P_new.shape[2] * num_states)

        new_carry = {
            "s": s_new,
            "P": P_new,
            "pos_cost": carry["pos_cost"] + pos_cost/H,
            "dot_cost": carry["dot_cost"] + dot_cost/H,
            "dv_cost_1": carry["dv_cost_1"],
            "dv_cost_2": carry["dv_cost_2"],
            "a_cost": carry["a_cost"] + a_cost/H,
            "e_cost": carry["e_cost"] + e_cost/H,
            "i_cost": carry["i_cost"] + i_cost/H,
            "offsets": carry["offsets"].at[:, i].set(offsets)
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
        "offsets": final["offsets"],
    }


def cost_total_only_static(dv_scaled, const_args: ConstArgs):
    out = cost_daughterships_jax_static(dv_scaled, const_args)
    return out["total"]


def main():
    # sets up the problem
    config_file = 'config.ini'
    asteroid = Asteroid(config_file)
    print('density kg/km^3')
    print(asteroid.rho)
    print('grav param km^3/s^2')
    print(asteroid.mu_grav_param)
    num_spacecraft = 3
    spacecraft = [Spacecraft(asteroid, config_file, None, 0)]
    if num_spacecraft > 1:
        for i in range(num_spacecraft-1):
            spacecraft.append(Spacecraft(asteroid, config_file, None, i+1))

    # time array (seconds)
    epochs = np.arange(len(spacecraft[1].a_hist)) * spacecraft[1].dt

    # fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    #
    # # semimajor axis
    # axs[0].plot(epochs, spacecraft[1].a_hist, lw=2)
    # axs[0].plot(epochs, spacecraft[2].a_hist, lw=2)
    # axs[0].set_ylabel("a (km)")
    # axs[0].set_title("Orbital Elements in Terminator Frame")
    # axs[0].grid(True)
    #
    # # eccentricity
    # axs[1].plot(epochs, spacecraft[1].e_hist, lw=2)
    # axs[1].plot(epochs, spacecraft[2].e_hist, lw=2)
    # axs[1].set_ylabel("e")
    # axs[1].grid(True)
    #
    # # inclination
    # axs[2].plot(epochs, 180/math.pi*spacecraft[1].i_hist, lw=2)
    # axs[2].plot(epochs, 180/math.pi*spacecraft[2].i_hist, lw=2)
    # axs[2].set_ylabel("i (deg)")
    # axs[2].grid(True)
    #
    # # RAAN
    # axs[3].plot(epochs, 180/math.pi*spacecraft[1].raan_hist, lw=2)
    # axs[3].plot(epochs, 180/math.pi*spacecraft[2].raan_hist, lw=2)
    # axs[3].set_ylabel("RAAN (deg)")
    # axs[3].grid(True)
    #
    # # argument of periapsis
    # axs[4].plot(epochs, 180/math.pi*spacecraft[1].aop_hist, lw=2)
    # axs[4].plot(epochs, 180/math.pi*spacecraft[2].aop_hist, lw=2)
    # axs[4].set_ylabel("AoP (deg)")
    # axs[4].set_xlabel("Epoch (s)")
    # axs[4].grid(True)
    # axs[4].legend(['D1', 'D2'])
    #
    # plt.tight_layout()
    # plt.show()

    simulation = Simulation(asteroid, spacecraft, config_file, None)
    starttime = datetime.now()

    mpc = MPC(simulation, asteroid, spacecraft)
    print("Run time: " + str(datetime.now() - starttime))
    mpc.plot_MPC_and_SCvx(simulation, asteroid, spacecraft)
    mpc.plot_animation(asteroid, spacecraft)


if __name__ == '__main__':
    #cProfile.run('main()', sort='tottime')
    main()

