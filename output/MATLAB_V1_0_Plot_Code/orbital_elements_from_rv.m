function [a, e, inc, RAAN, argp] = orbital_elements_from_rv(r, v, mu)
% Computes classical orbital elements from r, v (assumed in terminator frame)

    r = r(:); 
    v = v(:);

    r_norm = norm(r);
    v_norm = norm(v);

    h = cross(r, v);
    h_norm = norm(h);

    % semimajor axis
    energy = v_norm^2/2 - mu/r_norm;
    a = -mu / (2 * energy);

    % eccentricity vector
    e_vec = cross(v, h)/mu - r/r_norm;
    e = norm(e_vec);

    % inclination
    hx = h(1);
    hy = h(2);
    hz = h(3);
    sin_i = norm([hx, hy]);
    inc = 180/pi*atan2(sin_i, hz);

    % node vector
    k = [0; 0; 1];
    n = cross(k, h);
    n_norm = norm(n);

    % RAAN
    if n_norm < 1e-12
        RAAN = 0;
    else
        RAAN = 180/pi*atan2(n(2), n(1));
    end

    % argument of periapsis
    if e < 1e-12 || n_norm < 1e-12
        argp = 0;
    else
        cx = cross(n, e_vec);
        argp = 180/pi*atan2( cx(3)/(n_norm*e), dot(n, e_vec)/(n_norm*e) );
    end
end
