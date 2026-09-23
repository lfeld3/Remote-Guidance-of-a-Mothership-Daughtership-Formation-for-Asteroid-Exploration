function R = get_rotation(n_sun)
% Computes rotation matrix R such that R * n_sun = [0; 0; 1]

    z = n_sun(:) / norm(n_sun);

    % reference vector not parallel to z
    if abs(z(1)) < 0.9
        r = [1; 0; 0];
    else
        r = [0; 1; 0];
    end

    % x- and y-axes
    x = cross(r, z);
    x = x / norm(x);

    y = cross(z, x);

    % rotation matrix
    R = [x.'; y.'; z.'];
end
