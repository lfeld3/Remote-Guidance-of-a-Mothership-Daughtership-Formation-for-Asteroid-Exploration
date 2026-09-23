function [a_arr, e_arr, i_arr, RAAN_arr, argp_arr, r_rot_arr, v_rot_arr] = compute_elements_over_time(mu, s_d, spline_x, spline_y, spline_z, offset, offset_hist)

    N = size(s_d, 2);

    a_arr    = zeros(1, N);
    e_arr    = zeros(1, N);
    i_arr    = zeros(1, N);
    RAAN_arr = zeros(1, N);
    argp_arr = zeros(1, N);
    r_rot_arr = zeros(3, N);
    v_rot_arr = zeros(3, N);

    for k = 1:N

        % Sun direction
        n_sun = [ spline_x(k); spline_y(k); spline_z(k) ];
        n_sun = n_sun / norm(n_sun);

        % Rotation matrix
        R = get_rotation(n_sun);

        % Rotate state
        r = s_d(1:3, k);
        v = s_d(4:6, k);

        r_rot = R * r;
        v_rot = R * v;

        % Project into terminator plane
        % offset_rot = R * offset;
        % r_rot = r_rot - offset_rot;
        if offset_hist(k) == 0
            offset_k = R * offset;
        else
            offset_k = [0;0;offset_hist(k)];
        end

        r_rot = r_rot - offset_k;
        
        % Orbital elements
        [a, e, inc, RAAN, argp] = orbital_elements_from_rv(r_rot, v_rot, mu);

        a_arr(k)    = a;
        e_arr(k)    = e;
        i_arr(k)    = inc;
        RAAN_arr(k) = RAAN;
        argp_arr(k) = argp;
        r_rot_arr(:, k) = r_rot;
        v_rot_arr(:, k) = v_rot;
    end
end
