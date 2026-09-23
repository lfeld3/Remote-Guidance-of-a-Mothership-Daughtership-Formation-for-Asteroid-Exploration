import math
import sys

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')

from src.Spacecraft import Spacecraft
from src.asteroid import Asteroid
from src.simulation import Simulation
from scipy.integrate import odeint
from scipy.sparse.linalg import cg
import copy

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True, linewidth=100000)



class BLS:
    """Class contains all the methods for batch least squares (BLS) for daughtership state estimation"""

    def __init__(self, simulation, asteroid, spacecraft, mothership, time_elapsed, leftover):
        obs_end = time_elapsed+2
        obs_start = obs_end - spacecraft.N_batch - leftover
        self.range_to_extract = np.arange(obs_start, obs_end)
        self.range_to_extract_dvs = np.arange(obs_start+1, obs_end)
        print(self.range_to_extract)
        print(self.range_to_extract_dvs)
        self.observations(simulation, spacecraft, mothership, simulation.model)
        self.batch_estimation(simulation, asteroid, spacecraft, mothership)

    def observations(self, simulation, spacecraft, mothership, model):
        """Method obtains observations (noised true states) for use in the BLS algorithm

        Args:
            simulation: an instance of the Simulation class
            spacecraft: an instance of the Spacecraft class for a daughtership
            mothership: the instance of the Spacecraft class that represents the mothership
            model: string for what measurement model to use. "image_range" is what the paper uses
        """
        if model == 'image_range':
            self.s_observed_mothership = mothership.s_observed[:, self.range_to_extract]  # copy.deepcopy(
            self.s_observed = spacecraft.s_observed[0:6, self.range_to_extract]  # copy.deepcopy(
            self.r_diff = self.s_observed[0:3,:] - self.s_observed_mothership[0:3,:]  # copy.deepcopy(  copy.deepcopy(
            self.rho_observed = np.linalg.norm(self.r_diff, axis=0) + spacecraft.v_cov[2] ** 0.5 * np.random.normal(loc=0, scale=1, size=len(self.range_to_extract))

            # get rotation matrix
            R_hat = (self.s_observed[0:3, np.floor(len(self.range_to_extract) / 2).astype(int)] - self.s_observed_mothership[0:3, np.floor(len(self.range_to_extract) / 2).astype(int)])  # radial
            R_hat /= np.linalg.norm(R_hat)
            C_hat = np.cross(R_hat, self.s_observed[3:6, np.floor(len(self.range_to_extract) / 2).astype(int)] - self.s_observed_mothership[3:6, np.floor(len(self.range_to_extract) / 2).astype(int)])  # cross track
            C_hat /= np.linalg.norm(C_hat)
            I_hat = np.cross(C_hat, R_hat)  # in track
            I_hat /= np.linalg.norm(I_hat)
            self.rotation_camera_to_inertial = np.column_stack([I_hat, C_hat, R_hat])

            # camera calibration matrix
            f = simulation.focal_length
            f_x = f / simulation.pixel_pitch
            f_y = f / simulation.pixel_pitch
            w = simulation.camera_width
            h = simulation.camera_height
            self.K_camera = np.array([[f_x, 0, w / 2], [0, f_y, h / 2]])

            # get observed pixel coords
            meas = self.s_observed[0:3, :] - self.s_observed_mothership[0:3, :]
            camera_observed = np.matmul(self.rotation_camera_to_inertial.T, meas)
            dividend = np.tile(camera_observed[2, :].reshape([1, -1]), (3, 1))
            camera_focal_plane_observed = np.divide(camera_observed, dividend)  # this should be [3 x n]
            noises = (np.diag((mothership.v_cov) ** 0.5) @ np.random.normal(loc=0, scale=1, size=(len(mothership.v_cov), len(self.range_to_extract))))
            self.pixel_line_observed = np.matmul(self.K_camera, camera_focal_plane_observed) + noises   # .reshape([len(mothership.v_cov), spacecraft.N_batch], order='F')

    def batch_estimation(self, simulation, asteroid, spacecraft, mothership):
        """Method performs the Batch Least Squares (BLS) algorithm to update state estimates for the daughterships

        Args:
            simulation: instance of the Simulation class
            asteroid: instance of the Asteroid class
            spacecraft: instance of the Spacecraft class for the daughtership whose state estimate we want to update
            mothership: instance of the Spacecraft class for the mothership
        """
        do_printout = False
        num_iters = 8
        dt = spacecraft.dt
        t = np.array([0, dt])
        s_hat_prev = 1000 * np.ones(spacecraft.s_star.shape)

        N_batch = len(self.range_to_extract)
        num_states = spacecraft.num_states
        num_ctrl = spacecraft.num_controls - 1

        # delta-V bookkeeping for this batch
        dvs_batch_full = spacecraft.delta_v_hist[:, self.range_to_extract_dvs]
        dv_flags_full = spacecraft.delta_v_num_in_batch[self.range_to_extract_dvs]

        # keep only nonzero burns
        mask_burns = dv_flags_full > -1
        dvs_batch = dvs_batch_full[:, mask_burns]
        burn_epochs_global = self.range_to_extract_dvs[mask_burns]
        delta_v_count = dvs_batch.shape[1]

        # local burn indices (0..N_batch-1)
        obs_start = self.range_to_extract[0]
        burn_epochs_local = burn_epochs_global - obs_start - 1  # burns between obs j->j+1

        # augmented state: [x0; dv_0; dv_1; ...]
        s_aug_0 = spacecraft.s_star.copy()
        for k in range(delta_v_count):
            s_aug_0 = np.append(s_aug_0, dvs_batch[:, k])

        # prior covariance inverse
        P0_inv = np.linalg.inv(spacecraft.P)
        P_bar_inv = np.zeros((num_states + num_ctrl * delta_v_count, num_states + num_ctrl * delta_v_count))
        P_bar_inv[:num_states, :num_states] = P0_inv
        P_bar_inv[num_states:, num_states:] = np.diag(np.repeat(1 / (spacecraft.gates_sigma_1) ** 2, num_ctrl * delta_v_count))

        s_bar_0 = np.zeros((num_states + num_ctrl * delta_v_count, 1))

        # system matrix for dynamics
        A = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

        # batch iterations
        iter_num = 1
        converged = False
        while iter_num < num_iters and not converged:
            s_aug = s_aug_0.copy()
            Lambda = P_bar_inv.copy()
            N = P_bar_inv @ s_bar_0

            x_nom = s_aug[:num_states].copy()
            phi0 = np.eye(num_states)
            phi_dv = [np.eye(num_states) for _ in range(delta_v_count)]

            # store STM from epoch 0 to each measurement epoch
            Phi_hist = [np.eye(num_states)]  # Phi at j=0 is identity

            for j in range(N_batch):

                # measurement model
                H_tilde_s, extras = self.compute_H_tilde(j, x_nom, simulation.model, do_printout, simulation, mothership)
                H_blocks = [phi0]

                for k in range(delta_v_count):
                    dv_hat_k = s_aug[num_states + k * num_ctrl: num_states + (k + 1) * num_ctrl]
                    mass_loss_k = -dv_hat_k / spacecraft.exhaust_velocity

                    delta_v_mat_with_mass = np.array([
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [mass_loss_k[0], mass_loss_k[1], mass_loss_k[2]]])

                    burn_j = burn_epochs_local[k]

                    if j < burn_j:
                        H_block_k = np.zeros((num_states, num_ctrl))
                    elif j == burn_j:
                        H_block_k = delta_v_mat_with_mass
                    else:
                        H_block_k = phi_dv[k] @ delta_v_mat_with_mass

                    H_blocks.append(H_block_k)

                H_big = np.hstack(H_blocks)
                H_i = H_tilde_s @ H_big

                # measurement residual
                y_i = self.compute_y_i(j, x_nom, simulation.model, len(spacecraft.v_cov), extras, mothership)

                # weights
                if simulation.model == 'image_range':
                    W = np.linalg.inv(np.diag(spacecraft.v_cov))

                # accumulate normal equations
                Lambda += H_i.T @ W @ H_i
                N += H_i.T @ W @ y_i

                # apply burn after meas j
                if j in burn_epochs_local:
                    k = np.where(burn_epochs_local == j)[0][0]
                    dv_hat = s_aug[num_states + k * num_ctrl: num_states + (k + 1) * num_ctrl]
                    dv_vec = np.array([0, 0, 0, dv_hat[0], dv_hat[1], dv_hat[2], -np.linalg.norm(dv_hat) / spacecraft.exhaust_velocity])

                    x_nom += dv_vec
                    phi_dv[k] = np.eye(num_states)

                # propagate to next meas
                if j < N_batch - 1:
                    epoch = self.range_to_extract[j]

                    # state + STM propagation
                    state_and_phi = np.append(x_nom, phi0.flatten(order='F'))
                    sol = odeint(self.prop_batch, state_and_phi, t, args=(A, asteroid, spacecraft, epoch))

                    x_nom = sol[-1, :num_states]
                    phi0 = sol[-1, num_states:].reshape((num_states, num_states), order='F')

                    # store STM from epoch 0 to epoch j+1
                    Phi_hist.append(phi0.copy())

                    # STM for each dv block
                    for k in range(delta_v_count):
                        if j >= burn_epochs_local[k]:
                            phi_flat = phi_dv[k].flatten(order='F')
                            sol_phi = odeint(self.ode_stm, phi_flat, t, args=(A, asteroid, spacecraft, x_nom, epoch))
                            phi_dv[k] = sol_phi[-1].reshape((num_states, num_states), order='F')

            # solve normal equations
            s_hat = np.linalg.inv(Lambda) @ N
            s_aug_0 += s_hat.flatten()
            s_bar_0 -= s_hat

            if np.linalg.norm(s_hat - s_hat_prev) < 1e-5:
                converged = True
                print('converged! iter num ' + str(iter_num))
            else:
                s_hat_prev = s_hat
                iter_num += 1

        # full augmented covariance at epoch 0
        P_aug = np.linalg.inv(np.copy(Lambda))
        P0 = P_aug[:num_states, :num_states]

        # final propagation of best estimate over this batch
        s_star_0 = s_aug_0.copy()
        shat = s_star_0[:num_states].copy()

        # record initial post-BLS state and covariance for this batch (first obs in window)
        spacecraft.shat_post_BLS_hist[:, self.range_to_extract[0]] = shat
        spacecraft.cov_hist[:, :, self.range_to_extract[0]] = P0.copy()

        # propagate state and covariance along the window
        P_k = P0.copy()
        for j in range(N_batch - 1):
            epoch = self.range_to_extract[j]

            # apply estimated deltaV at this local epoch j (burn between obs j and j+1)
            if j in burn_epochs_local:
                k = np.where(burn_epochs_local == j)[0][0]
                dv_hat = s_star_0[num_states + k * num_ctrl: num_states + (k + 1) * num_ctrl]
                dv_vec = np.array([
                    0, 0, 0,
                    dv_hat[0], dv_hat[1], dv_hat[2],
                    -np.linalg.norm(dv_hat) / spacecraft.exhaust_velocity
                ])
                spacecraft.delta_v_hat_hist[:, self.range_to_extract_dvs[j]] = dv_hat
            else:
                dv_vec = np.zeros(num_states)
                spacecraft.delta_v_hat_hist[:, self.range_to_extract_dvs[j]] = np.zeros(num_ctrl)

            # propagate state from obs j to obs j+1
            sol_shat = odeint(
                spacecraft.ode_dynamics, shat + dv_vec, t,
                args=(A, asteroid, epoch, 'poly')
            )
            shat = sol_shat[-1, :].copy()
            spacecraft.shat_post_BLS_hist[:, self.range_to_extract[j + 1]] = shat

            # propagate covariance using STM from epoch 0 to epoch j+1
            Phi_j1 = Phi_hist[j + 1]  # STM from epoch 0 to epoch j+1
            P_j1 = Phi_j1 @ P0 @ Phi_j1.T
            spacecraft.cov_hist[:, :, self.range_to_extract[j + 1]] = P_j1.copy()

        # update current state and prior at final epoch
        spacecraft.shat = shat.copy()
        spacecraft.s_star = shat.copy()

        # final covariance at last epoch in window
        Phi_final = Phi_hist[-1]
        P_final = Phi_final @ P0 @ Phi_final.T

        # keep mass prior covariance
        P_final[6, 6] = spacecraft.P[6, 6]

        spacecraft.P = P_final.copy()

        # store deltaV burn covariances
        # extract dv block
        P_dv_block = P_aug[num_states:, num_states:]  # shape (3*n, 3*n)

        # store each burn's 3×3 covariance at its global epoch
        for k, epoch in enumerate(burn_epochs_global):
            P_dv_k = P_dv_block[3 * k: 3 * (k + 1), 3 * k: 3 * (k + 1)]
            spacecraft.delta_v_cov_hist[:, :, epoch] = P_dv_k.copy()

        if do_printout:
            print(P_final)
            print('predicted deltaV (mm/s)')
            print(spacecraft.delta_v_hat_hist[:, self.range_to_extract_dvs] * 1e6)
            print('true deltaV (mm/s)')
            print((spacecraft.delta_v_hist[:, self.range_to_extract_dvs] + spacecraft.delta_v_error_hist[:, self.range_to_extract_dvs]) * 1e6)

        print("BATCH:")
        print("  range_to_extract     =", self.range_to_extract)
        print("  range_to_extract_dvs =", self.range_to_extract_dvs)
        print("  burn_epochs_global   =", burn_epochs_global)
        print("  burn_epochs_local    =", burn_epochs_local)

    def ode_stm(self, stm_flat, t, A_sum, asteroid, spacecraft, state, epoch_of_sim):
        """Method computes the time derivative of the full state transition matrix (flattened for use in odeint)

        Args:
            stm_flat [49,]: flattened 7x7 STM for the spacecraft
            t: additional time after the time that has passed since the last epoch (sec)
            A_sum: [7x7] instance of the A matrix
            asteroid: instance of the Asteroid class
            state [7]: current daughtership state in inertial frame
            epoch_of_sim: scalar for what integer number of time steps have passed in the total simulation (excluding this current step)

        Returns:
            phi_dot [49,]: flattened 7x7 time derivative of stm_flat with respect to polyhedral gravity, SRP, and third body effects from Sun, EMB, Jupiter
        """
        time_since_beginning_of_sim = epoch_of_sim * spacecraft.dt + t

        stm = stm_flat.reshape([len(state), len(state)], order='F')
        angle_to_rotate = -asteroid.asteroid_ang_vel * time_since_beginning_of_sim
        c = math.cos(angle_to_rotate)
        s = math.sin(angle_to_rotate)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # gravity STM partials
        state_body = R @ state[0:3].reshape([3, 1])
        gg = asteroid.gravgrad_vec(state_body[0:3])
        gg_inertial = R.T @ gg @ R
        dCds = np.pad(gg_inertial, ((3, 1), (0, 4)), mode='constant')

        r_sun_to_apophis = np.array(
            [spacecraft.spline_sun_x(time_since_beginning_of_sim), spacecraft.spline_sun_y(time_since_beginning_of_sim),
             spacecraft.spline_sun_z(time_since_beginning_of_sim)])
        r_sun_to_EMB = np.array(
            [spacecraft.spline_EMB_x(time_since_beginning_of_sim), spacecraft.spline_EMB_y(time_since_beginning_of_sim),
             spacecraft.spline_EMB_z(time_since_beginning_of_sim)])
        r_sun_to_jup = np.array(
            [spacecraft.spline_jup_x(time_since_beginning_of_sim), spacecraft.spline_jup_y(time_since_beginning_of_sim),
             spacecraft.spline_jup_z(time_since_beginning_of_sim)])

        # SRP STM partials
        P0 = 4.56e-6  # N/m^2 at 1 AU
        A = spacecraft.surface_area  # m^2
        C_R = spacecraft.C_R
        m = np.exp(state[6])  # kg
        AU = 1.495978707e8  # km in 1 AU

        C = P0 * A * C_R * AU ** 2 / m / 1000  # km/s^2

        r = state[0:3] + r_sun_to_apophis  # km, Sun->Apo + Apo->S/C = Sun->S/C
        R = np.linalg.norm(r)

        dadr = C * (np.eye(3) / R ** 3 - 3 * np.outer(r, r) / R ** 5)  # 3x3 wrt position

        # derivative wrt log(mass)
        dadlogm = -C * r / R ** 3  # 3-vector

        dSRPds = np.zeros((7, 7))
        dSRPds[3:6, 0:3] = dadr
        dSRPds[3:6, 6] = dadlogm

        # 3rd body Sun STM partials
        mu_sun = 1.327e11  # km^3/s^2

        d = -state[0:3] - r_sun_to_apophis  # S/C->Apo + Apo->Sun = SC->Sun
        D = np.linalg.norm(d)

        d3b_sun = -mu_sun * (np.eye(3) / D ** 3 - 3 * np.outer(d, d) / D ** 5)

        d3rdSunds = np.zeros((7, 7))
        d3rdSunds[3:6, 0:3] = d3b_sun

        # 3rd body Earth-Moon bary STM partials
        mu_EMB = 3.986e5 + 4.902e3  # km^3/s^2

        r_EMB = -r_sun_to_apophis + r_sun_to_EMB  # Apo->Sun + Sun->EMB = Apo->EMB
        d_emb = -state[0:3] + r_EMB   # S/C->Apo + Apo->EMB = S/C->EMB
        D_emb = np.linalg.norm(d_emb)

        d3b_emb = -mu_EMB * (np.eye(3) / D_emb ** 3 - 3 * np.outer(d_emb, d_emb) / D_emb ** 5)

        d3rdEMBds = np.zeros((7, 7))
        d3rdEMBds[3:6, 0:3] = d3b_emb

        # 3rd body Jupiter STM partials
        mu_J = 1.26686534e8
        r_J = -r_sun_to_apophis + r_sun_to_jup  # Apo->Sun + Sun->Jup = Apo->Jup
        d_jup = -state[0:3] + r_J  # S/C->Apo + Apo->Jup = S/C->Jup
        D_jup = np.linalg.norm(d_jup)

        mu_J = 1.26686534e8  # km^3/s^2

        d3b_jup = -mu_J * (np.eye(3) / D_jup ** 3 - 3 * np.outer(d_jup, d_jup) / D_jup ** 5)

        d3rdJds = np.zeros((7, 7))
        d3rdJds[3:6, 0:3] = d3b_jup

        Aphi = A_sum + dCds + dSRPds + d3rdSunds + d3rdEMBds + d3rdJds
        phi_dot = Aphi @ stm
        return phi_dot.flatten(order='F')

    def prop_batch(self, states, t, A, asteroid, spacecraft, epoch_of_sim):
        """Method computes the time derivative of the full state transition matrix and spacecraft state itself

        Args:
            states [56,]: state vector + flattened 7x7 STM for the spacecraft
            t: additional time after the time that has passed since the last epoch (sec)
            A: [7x7] instance of the A matrix
            asteroid: instance of the Asteroid class
            spacecraft: instance of the Spacecraft class for the current daughtership
            epoch_of_sim: scalar for what integer number of time steps have passed in the total simulation (excluding this current step)

        Returns:
            states_dot [56,]: state time derivative and flattened 7x7 time derivative of stm_flat with respect to polyhedral gravity, SRP, and third body effects from Sun, EMB, Jupiter
        """
        state = states[:7]
        sdot = spacecraft.ode_dynamics(state, t, A, asteroid, epoch_of_sim, 'poly')
        stm = states[7:]
        phi_dot = self.ode_stm(stm, t, A, asteroid, spacecraft, state, epoch_of_sim)

        return np.append(sdot, phi_dot)

    def compute_H_tilde(self, j, s_star, model, do_printout, simulation, mothership):
        """Method computes Htilde for use in the BLS algorithm

        Args:
            j: local index in the BLS window of what observations to use in Htilde computation
            s_star: [7] current spacecraft nominal state
            model: string for what measurement model we are using
            do_printout: boolean for if things get printed out or not (mostly for bugfixing stuff)
            simulation: instance of the Simulation class
            mothership: instance of the Spacecraft class corresponding to the mothership

        Returns:
            Htilde: array for the Htilde matrix. Dimensions depend on model used, but generally [m,n] shape, m being number of measurements per epoch, n being state dimension
            extras: list of additional parameters passed onto the residual computation, compute_y_i()
        """
        current_mothership_shat = mothership.shat_hist[0:6, self.range_to_extract[j]]  # copy.deepcopy(
        if do_printout:
            print('Measurement epoch ' + str(j))
            print(s_star[0:3] - current_mothership_shat[0:3])

        if model == "image_range":
            H_tilde_s = np.zeros([3, mothership.num_states])
            r_diff = s_star[0:3].reshape(3) - current_mothership_shat[0:3]

            f = simulation.focal_length
            f_x = f / simulation.pixel_pitch
            f_y = f / simulation.pixel_pitch

            meas = r_diff
            meas_RIC = np.matmul(self.rotation_camera_to_inertial.T, meas)

            H_tilde_s[0:2, 0:3] = np.array([[f_x / meas_RIC[2], 0, -f_x * meas_RIC[0] / meas_RIC[2] ** 2],
                                            [0, f_y / meas_RIC[2], -f_y * meas_RIC[1] / meas_RIC[2] ** 2]]) @ self.rotation_camera_to_inertial.T

            # range
            rho_k = np.linalg.norm(r_diff)
            H_tilde_s[2, 0:3] = np.array([r_diff[0] / rho_k, r_diff[1] / rho_k, r_diff[2] / rho_k])
            if do_printout:
                print('H_tilde_s')
                print(H_tilde_s)

            extras = [rho_k]

        return H_tilde_s, extras

    def compute_y_i(self, j, s_star, model, m, extras, mothership):
        """Method computes observed-computed "O-C" residuals for use in the BLS algorithm

        Args:
            j: local index in the BLS window of what observations to use in Htilde computation
            s_star: [7] current spacecraft nominal state
            model: string for what measurement model we are using
            m: number of measurements per epoch
            extras: list of extra parameters from compute_H_tilde() that would otherwise be recomputed
            mothership: instance of the Spacecraft class corresponding to the mothership

        Returns:
            y_i: [m] vector for the current epoch j's residuals
        """
        current_mothership_shat = mothership.shat_hist[0:6, self.range_to_extract[j]]  # copy.deepcopy(
        if model == 'image_range':
            y_i = np.zeros(m)
            rho_k = extras

            y_i[2] = self.rho_observed[j] - rho_k

            # get computed pixel coords
            meas_hat = s_star[0:3] - current_mothership_shat[0:3]
            camera_computed = np.matmul(self.rotation_camera_to_inertial.T, meas_hat)
            dividend = camera_computed[2] * np.ones(camera_computed.shape)
            camera_focal_plane_computed = np.divide(camera_computed, dividend).reshape([-1, 1])
            self.pixel_line_computed = np.matmul(self.K_camera, camera_focal_plane_computed).reshape([len(mothership.v_cov)], order='F')

            y_i[0:2] = self.pixel_line_observed[:, j] - self.pixel_line_computed

        return y_i.reshape([m, 1])
