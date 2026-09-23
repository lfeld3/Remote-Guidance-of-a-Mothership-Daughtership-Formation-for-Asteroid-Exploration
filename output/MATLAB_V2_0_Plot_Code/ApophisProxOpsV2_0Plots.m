% Logan Feld
% Autonomous Landing and Proximity Operation Technology for
% Poorly-Characterized Small Bodies
% NSTGRO
% Mothership-Daughtership Formation Plotting for MPC V2.0

% Dependencies: ApophisProxOpsV2_0Plots.m, MATLAB_mats/...,
% read_ref_traj_data_file.m, apophis-reference-trajectories/...
% get_rotation.m, orbital_elements_from_rv.m
% compute_elements_over_time.m, Apophis_July22.obj

clc
clear all
close all
format longg
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
set(groot, 'DefaultFigureWindowState', 'normal');
set(groot, 'DefaultAxesFontSize', 20);
set(groot, 'DefaultTextFontSize', 22);
set(groot, 'DefaultLegendFontSize', 22);

% =========================================================================
% !!!!!!!!ATTENTION!!!!!!!!
% CHANGE THIS FILE PATH TO WHEREVER YOUR MATLAB_mats FOLDER IS
root = "MATLAB_mats/";
% =========================================================================


% =========================================================================
% Arrays
% =========================================================================

color1 = [0, 146, 146]/255;
color2 = [182, 109, 255]/255;
color3 = [219, 109, 0]/255;

mu_apophis = 3.00339*10^-9;  % km^3/s^2

s_hist_m = struct2array(load(strcat(root, "s_hist_m.mat")));
s_hist_1 = struct2array(load(strcat(root, "s_hist_1.mat")));
s_hist_2 = struct2array(load(strcat(root, "s_hist_2.mat")));
s_d_m = struct2array(load(strcat(root, "s_d_m.mat")));
s_d_1 = struct2array(load(strcat(root, "s_d_1.mat")));
s_d_2 = struct2array(load(strcat(root, "s_d_2.mat")));
dmin_hist_m = struct2array(load(strcat(root, "dmin_hist_m.mat")));
dmin_hist_1 = struct2array(load(strcat(root, "dmin_hist_1.mat")));
dmin_hist_2 = struct2array(load(strcat(root, "dmin_hist_2.mat")));
vmin_hist_m = struct2array(load(strcat(root, "vmin_hist_m.mat")));
vmin_hist_1 = struct2array(load(strcat(root, "vmin_hist_1.mat")));
vmin_hist_2 = struct2array(load(strcat(root, "vmin_hist_2.mat")));
delta_v_hist_m = struct2array(load(strcat(root, "delta_v_hist_m.mat")));
delta_v_hist_1 = struct2array(load(strcat(root, "delta_v_hist_1.mat")));
delta_v_hist_2 = struct2array(load(strcat(root, "delta_v_hist_2.mat")));
delta_v_error_hist_m = struct2array(load(strcat(root, "delta_v_error_hist_m.mat")));
delta_v_error_hist_1 = struct2array(load(strcat(root, "delta_v_error_hist_1.mat")));
delta_v_error_hist_2 = struct2array(load(strcat(root, "delta_v_error_hist_2.mat")));
cost_hist = struct2array(load(strcat(root, "cost_hist.mat")));
shat_post_BLS_hist_m = struct2array(load(strcat(root, "shat_post_BLS_hist_m.mat")));
shat_post_BLS_hist_1 = struct2array(load(strcat(root, "shat_post_BLS_hist_1.mat")));
shat_post_BLS_hist_2 = struct2array(load(strcat(root, "shat_post_BLS_hist_2.mat")));
shat_hist_m = struct2array(load(strcat(root, "shat_hist_m.mat")));
shat_hist_1 = struct2array(load(strcat(root, "shat_hist_1.mat")));
shat_hist_2 = struct2array(load(strcat(root, "shat_hist_2.mat")));
true_accel_hist_m = struct2array(load(strcat(root, "true_accel_hist_m.mat")));
true_accel_hist_1 = struct2array(load(strcat(root, "true_accel_hist_1.mat")));
true_accel_hist_2 = struct2array(load(strcat(root, "true_accel_hist_2.mat")));
pred_accel_hist_m = struct2array(load(strcat(root, "pred_accel_hist_m.mat")));
pred_accel_hist_1 = struct2array(load(strcat(root, "pred_accel_hist_1.mat")));
pred_accel_hist_2 = struct2array(load(strcat(root, "pred_accel_hist_2.mat")));
pos_cost_hist = struct2array(load(strcat(root, "pos_cost_hist.mat")));
dot_cost_hist = struct2array(load(strcat(root, "dot_cost_hist.mat")));
dv_cost_1_hist = struct2array(load(strcat(root, "dv_cost_1_hist.mat")));
dv_cost_2_hist = struct2array(load(strcat(root, "dv_cost_2_hist.mat")));
a_cost_hist = struct2array(load(strcat(root, "a_cost_hist.mat")));
e_cost_hist = struct2array(load(strcat(root, "e_cost_hist.mat")));
cov_cost_hist = struct2array(load(strcat(root, "cov_cost_hist.mat")));
i_cost_hist = struct2array(load(strcat(root, "i_cost_hist.mat")));
bistatic_angle_true_hist = struct2array(load(strcat(root, "bistatic_angle_true_hist.mat")));
bistatic_angle_hat_hist = struct2array(load(strcat(root, "bistatic_angle_hat_hist.mat")));
delta_v_hat_hist_m = struct2array(load(strcat(root, "delta_v_hat_hist_m.mat")));
delta_v_hat_hist_1 = struct2array(load(strcat(root, "delta_v_hat_hist_1.mat")));
delta_v_hat_hist_2 = struct2array(load(strcat(root, "delta_v_hat_hist_2.mat")));
feature_array = struct2array(load(strcat(root, "feature_array.mat")));
jacobian_hist = struct2array(load(strcat(root, "jacobian_hist.mat")));
duration = length(s_hist_m);
ref_duration_m = length(s_d_m);
ref_duration_1 = length(s_d_1);
ref_duration_2 = length(s_d_2);
a_hist_1 = struct2array(load(strcat(root, "a_hist_1.mat")));
a_hist_2 = struct2array(load(strcat(root, "a_hist_2.mat")));
e_hist_1 = struct2array(load(strcat(root, "e_hist_1.mat")));
e_hist_2 = struct2array(load(strcat(root, "e_hist_2.mat")));
i_hist_1 = struct2array(load(strcat(root, "i_hist_1.mat")));
i_hist_2 = struct2array(load(strcat(root, "i_hist_2.mat")));
raan_hist_1 = struct2array(load(strcat(root, "raan_hist_1.mat")));
raan_hist_2 = struct2array(load(strcat(root, "raan_hist_2.mat")));
aop_hist_1 = struct2array(load(strcat(root, "aop_hist_1.mat")));
aop_hist_2 = struct2array(load(strcat(root, "aop_hist_2.mat")));
offset_1 = struct2array(load(strcat(root, "offset_1.mat")))';
offset_2 = struct2array(load(strcat(root, "offset_2.mat")))';
horizon_dv_hist = struct2array(load(strcat(root, "horizon_dv_hist.mat")));
horizon_steps = size(horizon_dv_hist, 1)/6;
dt_multiplier = struct2array(load(strcat(root, "dt_multiplier.mat")))';
offset_hist = struct2array(load(strcat(root, "offset_hist.mat")))';
cov_hist_m = struct2array(load(strcat(root, "cov_hist_m.mat")));
cov_hist_1 = struct2array(load(strcat(root, "cov_hist_1.mat")));
cov_hist_2 = struct2array(load(strcat(root, "cov_hist_2.mat")));
delta_v_cov_hist_1 = struct2array(load(strcat(root, "delta_v_cov_hist_1.mat")));
delta_v_cov_hist_2 = struct2array(load(strcat(root, "delta_v_cov_hist_2.mat")));

% read ref traj data
s_sun_to_apophis = read_ref_traj_data_file('apophis-reference-trajectories/apophis.txt');

%duration = 4320;

epochs = 1:duration;

% =========================================================================

% Apophis 3D model

fid = fopen("Apophis_July22.obj");

verts = [];
facets = [];

while true
    line = fgetl(fid);
    if ~ischar(line), break, end

    if startsWith(line,'v ')
        vals = sscanf(line(3:end), '%f %f %f');
        verts(end+1,:) = vals';
    elseif startsWith(line,'f ')
        tokens = split(strtrim(line(3:end)));
        f = zeros(1,length(tokens));
        for k = 1:length(tokens)
            parts = split(tokens{k},'/');
            f(k) = str2double(parts{1});
        end
        facets(end+1,:) = f;
    end
end

fclose(fid);

facets = [facets facets(:,1)];

% compute all the unique edges

% expected number of edges using Euler's formula
numv = size(verts, 1);
numf = size(facets, 1);
E = numv + numf - 2;

% computed edges
edgeverts = [];
edgemap = [];
for i = 1:numf
    for j = 1:length(facets(1,:))-1
        % extract all vertex pairs from the facet matrix
        edgeverts = [edgeverts ; facets(i,j:j+1)];
        edgemap = [edgemap ; facets(i,j:j+1), i];
    end
end

% sort and remove duplicate/shared edges
edgeverts = sort(edgeverts,2);
edgeverts = unique(edgeverts,"rows");

% display if computed matches theoretical (it should! Euler is very smart)
if length(edgeverts(:,1))==E
    disp("Euler's and computed unique edge count match!")
else
    disp("Euler's and computed unique edge count do NOT match!")
end

% plot Apophis to check success of previous code
figure()
axis equal
patch("Faces",facets,"Vertices",verts,'FaceColor','#AAAAAA')
%title("Full 3D Trajectories Around Apophis")
xlabel("X (km)")
ylabel("Y (km)")
zlabel("Z (km)")
hold on
plot3(s_hist_m(1, epochs), s_hist_m(2, epochs), s_hist_m(3, epochs), "o-", 'MarkerSize', 3, 'Color', color1)
plot3(s_hist_1(1, epochs), s_hist_1(2, epochs), s_hist_1(3, epochs), "square-", 'MarkerSize', 3, 'Color', color2)
plot3(s_hist_2(1, epochs), s_hist_2(2, epochs), s_hist_2(3, epochs), "+-", 'MarkerSize', 3, 'Color', color3)
legend('Apophis', 'Mothership', 'D1', "D2")
view(30,20)

% SAM frame plots

r = s_sun_to_apophis(1:3, epochs);
v = s_sun_to_apophis(4:6, epochs);

% SAM axes
h = cross(r, v, 1);         % angular momentum (3 x K)
zhat = -h ./ vecnorm(h);    % anti-momentum direction

xhat = r ./ vecnorm(r);     % Sun -> Apophis direction

yhat = cross(zhat, xhat, 1);
yhat = yhat ./ vecnorm(yhat);   % normalize

% SAM rotation matrices
R_SAM = zeros(3,3,length(epochs));
for k = 1:length(epochs)
    R_SAM(:,:,k) = [xhat(:,k)'; yhat(:,k)'; zhat(:,k)'];
end

% rotate spacecraft trajectories
s_m_SAM = zeros(3,length(epochs));
s_1_SAM = zeros(3,length(epochs));
s_2_SAM = zeros(3,length(epochs));

for k = 1:length(epochs)
    s_m_SAM(:,k) = R_SAM(:,:,k) * s_hist_m(1:3,epochs(k));
    s_1_SAM(:,k) = R_SAM(:,:,k) * s_hist_1(1:3,epochs(k));
    s_2_SAM(:,k) = R_SAM(:,:,k) * s_hist_2(1:3,epochs(k));
end

% plot in SAM frame
figure()
axis equal
patch("Faces",facets,"Vertices",verts,'FaceColor','#AAAAAA')
xlabel("X_{SAM} (km)")
ylabel("Y_{SAM} (km)")
zlabel("Z_{SAM} (km)")
hold on

plot3(s_m_SAM(1,:), s_m_SAM(2,:), s_m_SAM(3,:), "o-", 'MarkerSize', 3, 'Color', color1)
plot3(s_1_SAM(1,:), s_1_SAM(2,:), s_1_SAM(3,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot3(s_2_SAM(1,:), s_2_SAM(2,:), s_2_SAM(3,:), "+-", 'MarkerSize', 3, 'Color', color3)

legend('Apophis', 'Mothership (SAM)', 'D1 (SAM)', 'D2 (SAM)')
view(30,20)

% plot in SAM frame XZ
figure()
axis equal
scatter(verts(:,1), verts(:,3), [], [0.3 0.3 0.3], 'filled')
%patch("Faces",facets,"Vertices",verts,'FaceColor','#AAAAAA')
xlabel("X_{SAM} (km)")
ylabel("Z_{SAM} (km)")
hold on

plot(s_m_SAM(1,:), s_m_SAM(3,:), "o-", 'MarkerSize', 3, 'Color', color1)
plot(s_1_SAM(1,:), s_1_SAM(3,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot(s_2_SAM(1,:), s_2_SAM(3,:), "+-", 'MarkerSize', 3, 'Color', color3)

legend('Apophis', 'Mothership (SAM)', 'D1 (SAM)', 'D2 (SAM)')
xlim([-3, 3])
ylim([-3, 3])

% plot in SAM frame YZ
figure()
axis equal
scatter(verts(:,2), verts(:,3), [], [0.3 0.3 0.3], 'filled')
%patch("Faces",facets,"Vertices",verts,'FaceColor','#AAAAAA')
xlabel("Y_{SAM} (km)")
ylabel("Z_{SAM} (km)")
hold on

plot(s_m_SAM(2,:), s_m_SAM(3,:), "o-", 'MarkerSize', 3, 'Color', color1)
plot(s_1_SAM(2,:), s_1_SAM(3,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot(s_2_SAM(2,:), s_2_SAM(3,:), "+-", 'MarkerSize', 3, 'Color', color3)

legend('Apophis', 'Mothership (SAM)', 'D1 (SAM)', 'D2 (SAM)')
xlim([-3, 3])
ylim([-3, 3])

% Plots
% =========================================================================

% Commanded deltaV Mag and Erroneous deltaV

max_dv_1 = 1.1*max(vecnorm(delta_v_hist_1 + delta_v_error_hist_1)*10^6);
max_dv_2 = 1.1*max(vecnorm(delta_v_hist_2 + delta_v_error_hist_2)*10^6);

max_dv = max([max_dv_1, max_dv_2]);

figure()
t = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile(t)
plot(epochs, vecnorm(delta_v_hist_1)*10^6, 'Color', color2)
hold on
plot(epochs, vecnorm(delta_v_hist_2)*10^6, 'Color', color3)
hold off
xlabel('Epochs (\Deltat)')
ylabel('||\DeltaV|| (mm/s)')
subtitle('Commanded \DeltaV')
xlim([0, duration])
ylim([0,max_dv])
nexttile(t)
plot(epochs, vecnorm(delta_v_hist_1 + delta_v_error_hist_1)*10^6, 'Color', color2)
hold on
plot(epochs, vecnorm(delta_v_hist_2 + delta_v_error_hist_2)*10^6, 'Color', color3)
hold off
xlabel('Epochs (\Deltat)')
subtitle('Actual \DeltaV (With Errors from Gates Model)')
xlim([0, duration])
ylim([0,max_dv])
nexttile(t)

% deltaV mask
mask_1 = vecnorm(delta_v_hist_1) > 0;
mask_2 = vecnorm(delta_v_hist_2) > 0;

delta_v_hist_1_unit = delta_v_hist_1(:, mask_1) ./ repmat(vecnorm(delta_v_hist_1(:, mask_1)), [3,1]);
delta_v_hist_2_unit = delta_v_hist_2(:, mask_2) ./ repmat(vecnorm(delta_v_hist_2(:, mask_2)), [3,1]);
delta_v_hist_1_actual_unit = (delta_v_hist_1(:, mask_1) + delta_v_error_hist_1(:, mask_1)) ./ vecnorm(delta_v_hist_1(:, mask_1) + delta_v_error_hist_1(:, mask_1));
delta_v_hist_2_actual_unit = (delta_v_hist_2(:, mask_2) + delta_v_error_hist_2(:, mask_2)) ./ vecnorm(delta_v_hist_2(:, mask_2) + delta_v_error_hist_2(:, mask_2));
dot_products_1 = acosd(sum(delta_v_hist_1_unit .* delta_v_hist_1_actual_unit, 1));
dot_products_2 = acosd(sum(delta_v_hist_2_unit .* delta_v_hist_2_actual_unit, 1));
scatter(epochs(mask_1), dot_products_1, 'MarkerEdgeColor', color2, 'MarkerFaceColor', color2)
hold on
scatter(epochs(mask_2), dot_products_2, 'MarkerEdgeColor', color3, 'MarkerFaceColor', color3)
hold off
xlabel('Epochs (\Deltat)')
ylabel('Angle between commanded and actual \DeltaV (deg)')
subtitle('Angle between commanded and actual \DeltaV')
legend("D1", "D2")
xlim([0, duration])


% true and predicted acceleration difference

[~, max_accel_m_x] = max(true_accel_hist_m(1,epochs));
[~, max_accel_m_y] = max(true_accel_hist_m(2,epochs));
[~, max_accel_m_z] = max(true_accel_hist_m(3,epochs));
[~, max_accel_1_x] = max(true_accel_hist_1(1,epochs));
[~, max_accel_1_y] = max(true_accel_hist_1(2,epochs));
[~, max_accel_1_z] = max(true_accel_hist_1(3,epochs));
[~, max_accel_2_x] = max(true_accel_hist_2(1,epochs));
[~, max_accel_2_y] = max(true_accel_hist_2(2,epochs));
[~, max_accel_2_z] = max(true_accel_hist_2(3,epochs));


figure()
t = tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile(t)

plot(1:(duration-1), (true_accel_hist_m(1, 2:duration) - pred_accel_hist_m(1, 2:duration))./true_accel_hist_m(1, max_accel_m_x)*100, "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(1:(duration-1), (true_accel_hist_1(1, 2:duration) - pred_accel_hist_1(1, 2:duration))./true_accel_hist_1(1, max_accel_1_x)*100, "square-", 'MarkerSize', 3, 'Color', color2)
plot(1:(duration-1), (true_accel_hist_2(1, 2:duration) - pred_accel_hist_2(1, 2:duration))./true_accel_hist_2(1, max_accel_2_x)*100, "+-", 'MarkerSize', 3, 'Color', color3)
hold off
ylabel('Gravitational Accel Diff (%)')
xlabel('Epochs (\Deltat)')
subtitle('\Delta a_X')
xlim([0, duration])
nexttile(t)
plot(1:(duration-1), (true_accel_hist_m(2, 2:duration) - pred_accel_hist_m(2, 2:duration))./true_accel_hist_m(2, max_accel_m_y)*100, "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(1:(duration-1), (true_accel_hist_1(2, 2:duration) - pred_accel_hist_1(2, 2:duration))./true_accel_hist_1(2, max_accel_1_y)*100, "square-", 'MarkerSize', 3, 'Color', color2)
plot(1:(duration-1), (true_accel_hist_2(2, 2:duration) - pred_accel_hist_2(2, 2:duration))./true_accel_hist_2(2, max_accel_2_y)*100, "+-", 'MarkerSize', 3, 'Color', color3)
hold off
xlabel('Epochs (\Deltat)')
subtitle('\Delta a_Y')
xlim([0, duration])
nexttile(t)
plot(1:(duration-1), (true_accel_hist_m(3, 2:duration) - pred_accel_hist_m(3, 2:duration))./true_accel_hist_m(3, max_accel_m_z)*100, "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(1:(duration-1), (true_accel_hist_1(3, 2:duration) - pred_accel_hist_1(3, 2:duration))./true_accel_hist_1(3, max_accel_1_z)*100, "square-", 'MarkerSize', 3, 'Color', color2)
plot(1:(duration-1), (true_accel_hist_2(3, 2:duration) - pred_accel_hist_2(3, 2:duration))./true_accel_hist_2(3, max_accel_2_z)*100, "+-", 'MarkerSize', 3, 'Color', color3)
hold off
xlabel('Epochs (\Deltat)')
subtitle('\Delta a_Z')
legend('Mothership', 'Daughtership 1', 'Daughtership 2')
xlim([0, duration])


% Cost components
figure()
plot(epochs, pos_cost_hist(epochs), "o-", 'MarkerSize', 3)
hold on
plot(epochs, dot_cost_hist(epochs), "+-", 'MarkerSize', 3)
plot(epochs, dv_cost_1_hist(epochs), ">-", 'MarkerSize', 3)
plot(epochs, dv_cost_2_hist(epochs), "diamond-", 'MarkerSize', 3)
%plot(epochs, e_cost_hist(epochs), "square-", 'MarkerSize', 3)
%plot(epochs, a_cost_hist(epochs), "x-", 'MarkerSize', 3)
plot(epochs, i_cost_hist(epochs), "pentagram-", 'MarkerSize', 3)
hold off
xlabel('Epochs (\Deltat)')
ylabel('Cost')
legend(["Position", "Antipodal Angle", "\DeltaV_1", "\DeltaV_2", "Inclination"])
%legend(["Position", "Antipodal Angle", "\DeltaV_1", "\DeltaV_2", "Eccentricity", "Semimajor Axis", "Inclination"])
xlim([0, duration])

% bistatic angle

figure()
plot(1:(duration-1), bistatic_angle_true_hist(1:duration-1), "+-", 'MarkerSize', 3, "Color", color1)
hold on
plot(1:(duration-1), bistatic_angle_hat_hist(1:duration-1), "square-", 'MarkerSize', 3, "Color", color3)
hold off
xlabel('Epochs (\Deltat)')
ylabel('Antipodal Angle (deg)')
%subtitle('Antipodal Angle vs Epoch')
legend('True', 'Estimated')
xlim([0, duration])



% orbital element history of reference with initial terminator plane only

figure()

% 1. Semi-major axis
subplot(5,1,1)
plot(1:ref_duration_1, a_hist_1, 'LineWidth', 1.8)
hold on
plot(1:ref_duration_2, a_hist_2, 'LineWidth', 1.8)
ylabel('a (km)')
title('Orbital Elements in Terminator Frame')
grid on

% 2. Eccentricity
subplot(5,1,2)
plot(1:ref_duration_1, e_hist_1, 'LineWidth', 1.8)
hold on
plot(1:ref_duration_2, e_hist_2, 'LineWidth', 1.8)
hold off
ylabel('e')
grid on

% 3. Inclination
subplot(5,1,3)
plot(1:ref_duration_1, 180/pi*i_hist_1, 'LineWidth', 1.8)
hold on
plot(1:ref_duration_2, 180/pi*i_hist_2, 'LineWidth', 1.8)
hold off
ylabel('i (deg)')
grid on

% 4. RAAN
subplot(5,1,4)
plot(1:ref_duration_1, 180/pi*raan_hist_1, 'LineWidth', 1.8)
hold on
plot(1:ref_duration_2, 180/pi*raan_hist_2, 'LineWidth', 1.8)
hold off
ylabel('RAAN (deg)')
grid on

% 5. Argument of periapsis
subplot(5,1,5)
plot(1:ref_duration_1, 180/pi*aop_hist_1, 'LineWidth', 1.8)
hold on
plot(1:ref_duration_2, 180/pi*aop_hist_2, 'LineWidth', 1.8)
hold off
ylabel('\omega (deg)')
xlabel('Epoch (s)')
grid on
legend('D1','D2')

% actual orbital elements

figure()

spline_x = griddedInterpolant(1:length(s_sun_to_apophis(1,epochs)), s_sun_to_apophis(1,epochs), 'spline');
spline_y = griddedInterpolant(1:length(s_sun_to_apophis(2,epochs)), s_sun_to_apophis(2,epochs), 'spline');
spline_z = griddedInterpolant(1:length(s_sun_to_apophis(3,epochs)), s_sun_to_apophis(3,epochs), 'spline');

offset_1_hist = offset_hist(1,epochs);
offset_2_hist = offset_hist(2,epochs);
[a_hist_actual_1, e_hist_actual_1, i_hist_actual_1, raan_hist_actual_1, aop_hist_actual_1, r_rot_1, v_rot_1] = compute_elements_over_time(mu_apophis, s_hist_1(:, epochs), spline_x, spline_y, spline_z, offset_1, offset_1_hist);
[a_hist_actual_2, e_hist_actual_2, i_hist_actual_2, raan_hist_actual_2, aop_hist_actual_2, r_rot_2, v_rot_2] = compute_elements_over_time(mu_apophis, s_hist_2(:, epochs), spline_x, spline_y, spline_z, offset_2, offset_2_hist);

% 1. Semi-major axis
t = tiledlayout(5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile(t)
plot(epochs, a_hist_actual_1(epochs), "+-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color2)
hold on
plot(epochs, a_hist_actual_2(epochs), "square-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color3)
ylabel('a (km)')
%title('Orbital Elements in Terminator Frame')
grid on
xlim([0, duration])

% 2. Eccentricity
nexttile(t)
plot(epochs, e_hist_actual_1(epochs), "+-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color2)
hold on
plot(epochs, e_hist_actual_2(epochs), "square-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color3)
hold off
ylabel('e')
grid on
xlim([0, duration])

% 3. Inclination
nexttile(t)
plot(epochs, i_hist_actual_1(epochs), "+-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color2)
hold on
plot(epochs, i_hist_actual_2(epochs), "square-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color3)
hold off
ylabel('i (deg)')
grid on
xlim([0, duration])

% 4. RAAN
nexttile(t)
plot(epochs, raan_hist_actual_1(epochs), "+-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color2)
hold on
plot(epochs, raan_hist_actual_2(epochs), "square-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color3)
hold off
ylabel('RAAN (deg)')
grid on
xlim([0, duration])

% 5. Argument of periapsis
nexttile(t)
plot(epochs, aop_hist_actual_1(epochs), "+-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color2)
hold on
plot(epochs, aop_hist_actual_2(epochs), "square-", 'MarkerSize', 3, 'LineWidth', 1.8, "Color", color3)
hold off
ylabel('\omega (deg)')
xlabel('Epoch (\Deltat)')
grid on
legend('D1','D2')
xlim([0, duration])


% residual and cov plots in SAM frame

% get residuals

res_m  = s_hist_m  - shat_post_BLS_hist_m;
res_1  = s_hist_1  - shat_post_BLS_hist_1;
res_2  = s_hist_2  - shat_post_BLS_hist_2;

res_m_v = s_hist_m(4:6,:) - shat_post_BLS_hist_m(4:6,:);
res_1_v = s_hist_1(4:6,:) - shat_post_BLS_hist_1(4:6,:);
res_2_v = s_hist_2(4:6,:) - shat_post_BLS_hist_2(4:6,:);

% state residuals SAM

N = length(epochs);

res_m_SAM  = zeros(3,N);
res_1_SAM  = zeros(3,N);
res_2_SAM  = zeros(3,N);

res_m_v_SAM = zeros(3,N);
res_1_v_SAM = zeros(3,N);
res_2_v_SAM = zeros(3,N);

for k = 1:N
    Rk = R_SAM(:,:,k);

    res_m_SAM(:,k)  = Rk * res_m(1:3,epochs(k));
    res_1_SAM(:,k)  = Rk * res_1(1:3,epochs(k));
    res_2_SAM(:,k)  = Rk * res_2(1:3,epochs(k));

    res_m_v_SAM(:,k) = Rk * res_m_v(:,epochs(k));
    res_1_v_SAM(:,k) = Rk * res_1_v(:,epochs(k));
    res_2_v_SAM(:,k) = Rk * res_2_v(:,epochs(k));
end

% covariance SAM

cov_m_SAM  = zeros(3,3,N);
cov_1_SAM  = zeros(3,3,N);
cov_2_SAM  = zeros(3,3,N);

cov_m_v_SAM = zeros(3,3,N);
cov_1_v_SAM = zeros(3,3,N);
cov_2_v_SAM = zeros(3,3,N);

for k = 1:N
    Rk = R_SAM(:,:,k);

    cov_m_SAM(:,:,k)  = Rk * cov_hist_m(1:3,1:3,epochs(k)) * Rk';
    cov_1_SAM(:,:,k)  = Rk * cov_hist_1(1:3,1:3,epochs(k)) * Rk';
    cov_2_SAM(:,:,k)  = Rk * cov_hist_2(1:3,1:3,epochs(k)) * Rk';

    cov_m_v_SAM(:,:,k) = Rk * cov_hist_m(4:6,4:6,epochs(k)) * Rk';
    cov_1_v_SAM(:,:,k) = Rk * cov_hist_1(4:6,4:6,epochs(k)) * Rk';
    cov_2_v_SAM(:,:,k) = Rk * cov_hist_2(4:6,4:6,epochs(k)) * Rk';
end

% extract rotated 1sigma diagonals
cov_m_sig  = sqrt([ squeeze(cov_m_SAM(1,1,:))'; squeeze(cov_m_SAM(2,2,:))'; squeeze(cov_m_SAM(3,3,:))' ]);
cov_1_sig  = sqrt([ squeeze(cov_1_SAM(1,1,:))'; squeeze(cov_1_SAM(2,2,:))'; squeeze(cov_1_SAM(3,3,:))' ]);
cov_2_sig  = sqrt([ squeeze(cov_2_SAM(1,1,:))'; squeeze(cov_2_SAM(2,2,:))'; squeeze(cov_2_SAM(3,3,:))' ]);

cov_m_v_sig = sqrt([ squeeze(cov_m_v_SAM(1,1,:))'; squeeze(cov_m_v_SAM(2,2,:))'; squeeze(cov_m_v_SAM(3,3,:))' ]);
cov_1_v_sig = sqrt([ squeeze(cov_1_v_SAM(1,1,:))'; squeeze(cov_1_v_SAM(2,2,:))'; squeeze(cov_1_v_SAM(3,3,:))' ]);
cov_2_v_sig = sqrt([ squeeze(cov_2_v_SAM(1,1,:))'; squeeze(cov_2_v_SAM(2,2,:))'; squeeze(cov_2_v_SAM(3,3,:))' ]);

% deltaV SAM

% daughtership delta V covariance

dv1 = delta_v_hat_hist_1(:, epochs(mask_1));
dv1_cov_x = reshape(sqrt(delta_v_cov_hist_1(1,1,epochs(mask_1))), [],1);
dv1_cov_y = reshape(sqrt(delta_v_cov_hist_1(2,2,epochs(mask_1))), [],1);
dv1_cov_z = reshape(sqrt(delta_v_cov_hist_1(3,3,epochs(mask_1))), [],1);

dv2 = delta_v_hat_hist_2(:, epochs(mask_2));
dv2_cov_x = reshape(sqrt(delta_v_cov_hist_2(1,1,epochs(mask_2))), [],1);
dv2_cov_y = reshape(sqrt(delta_v_cov_hist_2(2,2,epochs(mask_2))), [],1);
dv2_cov_z = reshape(sqrt(delta_v_cov_hist_2(3,3,epochs(mask_2))), [],1);

dv1_res = delta_v_hist_1(:,epochs(mask_1)) + delta_v_error_hist_1(:,epochs(mask_1)) - dv1;
dv2_res = delta_v_hist_2(:,epochs(mask_2)) + delta_v_error_hist_2(:,epochs(mask_2)) - dv2;

dv1_res_SAM = zeros(3,sum(mask_1));
dv2_res_SAM = zeros(3,sum(mask_2));

dv1_cov_SAM = zeros(3,3,sum(mask_1));
dv2_cov_SAM = zeros(3,3,sum(mask_2));

k = epochs(mask_1);
for i = 1:sum(mask_1)    
    Rk = R_SAM(:,:,k(i));

    dv1_res_SAM(:,i) = Rk * dv1_res(:,i);
    dv1_cov_SAM(:,:,i) = Rk * delta_v_cov_hist_1(:,:,k(i)) * Rk';
end

k = epochs(mask_2);
for i = 1:sum(mask_2)
    Rk = R_SAM(:,:,k(i));

    dv2_res_SAM(:,i) = Rk * dv2_res(:,i);
    dv2_cov_SAM(:,:,i) = Rk * delta_v_cov_hist_2(:,:,k(i)) * Rk';
end

dv1_sig = sqrt([ squeeze(dv1_cov_SAM(1,1,:))'; squeeze(dv1_cov_SAM(2,2,:))'; squeeze(dv1_cov_SAM(3,3,:))']);
dv2_sig = sqrt([ squeeze(dv2_cov_SAM(1,1,:))'; squeeze(dv2_cov_SAM(2,2,:))'; squeeze(dv2_cov_SAM(3,3,:))']);

% mothership SAM plot

figure()
max_m_x = 1.5*max(res_m_SAM(1,10:end) + 3*cov_m_sig(1,10:end));
min_m_x = 1.5*min(res_m_SAM(1,10:end) - 3*cov_m_sig(1,10:end));
max_m_y = 1.5*max(res_m_SAM(2,10:end) + 3*cov_m_sig(2,10:end));
min_m_y = 1.5*min(res_m_SAM(2,10:end) - 3*cov_m_sig(2,10:end));
max_m_z = 1.5*max(res_m_SAM(3,10:end) + 3*cov_m_sig(3,10:end));
min_m_z = 1.5*min(res_m_SAM(3,10:end) - 3*cov_m_sig(3,10:end));

max_m_vx = 1.5*max(res_m_v_SAM(1,10:end)*1e6 + 3*cov_m_v_sig(1,10:end)*1e6);
min_m_vx = 1.5*min(res_m_v_SAM(1,10:end)*1e6 - 3*cov_m_v_sig(1,10:end)*1e6);
max_m_vy = 1.5*max(res_m_v_SAM(2,10:end)*1e6 + 3*cov_m_v_sig(2,10:end)*1e6);
min_m_vy = 1.5*min(res_m_v_SAM(2,10:end)*1e6 - 3*cov_m_v_sig(2,10:end)*1e6);
max_m_vz = 1.5*max(res_m_v_SAM(3,10:end)*1e6 + 3*cov_m_v_sig(3,10:end)*1e6);
min_m_vz = 1.5*min(res_m_v_SAM(3,10:end)*1e6 - 3*cov_m_v_sig(3,10:end)*1e6);

t = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

nexttile(t)
plot(epochs, res_m_SAM(1,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_SAM(1,:) + 3*cov_m_sig(1,:), "r--")
plot(epochs, res_m_SAM(1,:) - 3*cov_m_sig(1,:), "r--")
hold off
subtitle('r_X (SAM)')
xlim([0 duration])
ylim([min_m_x max_m_x])
ylabel('Position (km)')

nexttile(t)
plot(epochs, res_m_SAM(2,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_SAM(2,:) + 3*cov_m_sig(2,:), "r--")
plot(epochs, res_m_SAM(2,:) - 3*cov_m_sig(2,:), "r--")
hold off
subtitle('r_Y (SAM)')
xlim([0 duration])
ylim([min_m_y max_m_y])

nexttile(t)
plot(epochs, res_m_SAM(3,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_SAM(3,:) + 3*cov_m_sig(3,:), "r--")
plot(epochs, res_m_SAM(3,:) - 3*cov_m_sig(3,:), "r--")
hold off
subtitle('r_Z (SAM)')
xlim([0 duration])
ylim([min_m_z max_m_z])

nexttile(t)
plot(epochs, res_m_v_SAM(1,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_v_SAM(1,:)*1e6 + 3*cov_m_v_sig(1,:)*1e6, "r--")
plot(epochs, res_m_v_SAM(1,:)*1e6 - 3*cov_m_v_sig(1,:)*1e6, "r--")
hold off
subtitle('v_X (SAM)')
xlim([0 duration])
ylim([min_m_vx max_m_vx])
ylabel('Velocity (mm/s)')
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_m_v_SAM(2,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_v_SAM(2,:)*1e6 + 3*cov_m_v_sig(2,:)*1e6, "r--")
plot(epochs, res_m_v_SAM(2,:)*1e6 - 3*cov_m_v_sig(2,:)*1e6, "r--")
hold off
subtitle('v_Y (SAM)')
xlim([0 duration])
ylim([min_m_vy max_m_vy])
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_m_v_SAM(3,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_m_v_SAM(3,:)*1e6 + 3*cov_m_v_sig(3,:)*1e6, "r--")
plot(epochs, res_m_v_SAM(3,:)*1e6 - 3*cov_m_v_sig(3,:)*1e6, "r--")
hold off
subtitle('v_Z (SAM)')
legend('Mothership','3\sigma formal uncertainty')
xlabel('Epochs (\Deltat)')
xlim([0 duration])
ylim([min_m_vz max_m_vz])

% D1 state SAM plot

figure()
t = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

nexttile(t)
plot(epochs, res_1_SAM(1,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_SAM(1,:) + 3*cov_1_sig(1,:), "r--")
plot(epochs, res_1_SAM(1,:) - 3*cov_1_sig(1,:), "r--")
hold off
subtitle('r_X (SAM)')
xlim([0 duration])
ylabel('Position (km)')

nexttile(t)
plot(epochs, res_1_SAM(2,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_SAM(2,:) + 3*cov_1_sig(2,:), "r--")
plot(epochs, res_1_SAM(2,:) - 3*cov_1_sig(2,:), "r--")
hold off
subtitle('r_Y (SAM)')
xlim([0 duration])

nexttile(t)
plot(epochs, res_1_SAM(3,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_SAM(3,:) + 3*cov_1_sig(3,:), "r--")
plot(epochs, res_1_SAM(3,:) - 3*cov_1_sig(3,:), "r--")
hold off
subtitle('r_Z (SAM)')
xlim([0 duration])

nexttile(t)
plot(epochs, res_1_v_SAM(1,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_v_SAM(1,:)*1e6 + 3*cov_1_v_sig(1,:)*1e6, "r--")
plot(epochs, res_1_v_SAM(1,:)*1e6 - 3*cov_1_v_sig(1,:)*1e6, "r--")
hold off
subtitle('v_X (SAM)')
xlim([0 duration])
ylabel('Velocity (mm/s)')
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_1_v_SAM(2,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_v_SAM(2,:)*1e6 + 3*cov_1_v_sig(2,:)*1e6, "r--")
plot(epochs, res_1_v_SAM(2,:)*1e6 - 3*cov_1_v_sig(2,:)*1e6, "r--")
hold off
subtitle('v_Y (SAM)')
xlim([0 duration])
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_1_v_SAM(3,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_1_v_SAM(3,:)*1e6 + 3*cov_1_v_sig(3,:)*1e6, "r--")
plot(epochs, res_1_v_SAM(3,:)*1e6 - 3*cov_1_v_sig(3,:)*1e6, "r--")
hold off
subtitle('v_Z (SAM)')
legend('Daughtership 1','3\sigma formal uncertainty')
xlabel('Epochs (\Deltat)')
xlim([0 duration])

% D2 state SAM plot

figure()
t = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

nexttile(t)
plot(epochs, res_2_SAM(1,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_SAM(1,:) + 3*cov_2_sig(1,:), "r--")
plot(epochs, res_2_SAM(1,:) - 3*cov_2_sig(1,:), "r--")
hold off
subtitle('r_X (SAM)')
xlim([0 duration])
ylabel('Position (km)')

nexttile(t)
plot(epochs, res_2_SAM(2,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_SAM(2,:) + 3*cov_2_sig(2,:), "r--")
plot(epochs, res_2_SAM(2,:) - 3*cov_2_sig(2,:), "r--")
hold off
subtitle('r_Y (SAM)')
xlim([0 duration])

nexttile(t)
plot(epochs, res_2_SAM(3,:), "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_SAM(3,:) + 3*cov_2_sig(3,:), "r--")
plot(epochs, res_2_SAM(3,:) - 3*cov_2_sig(3,:), "r--")
hold off
subtitle('r_Z (SAM)')
xlim([0 duration])

nexttile(t)
plot(epochs, res_2_v_SAM(1,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_v_SAM(1,:)*1e6 + 3*cov_2_v_sig(1,:)*1e6, "r--")
plot(epochs, res_2_v_SAM(1,:)*1e6 - 3*cov_2_v_sig(1,:)*1e6, "r--")
hold off
subtitle('v_X (SAM)')
xlim([0 duration])
ylabel('Velocity (mm/s)')
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_2_v_SAM(2,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_v_SAM(2,:)*1e6 + 3*cov_2_v_sig(2,:)*1e6, "r--")
plot(epochs, res_2_v_SAM(2,:)*1e6 - 3*cov_2_v_sig(2,:)*1e6, "r--")
hold off
subtitle('v_Y (SAM)')
xlim([0 duration])
xlabel('Epochs (\Deltat)')

nexttile(t)
plot(epochs, res_2_v_SAM(3,:)*1e6, "o-", 'MarkerSize',3,'Color',color1)
hold on
plot(epochs, res_2_v_SAM(3,:)*1e6 + 3*cov_2_v_sig(3,:)*1e6, "r--")
plot(epochs, res_2_v_SAM(3,:)*1e6 - 3*cov_2_v_sig(3,:)*1e6, "r--")
hold off
subtitle('v_Z (SAM)')
legend('Daughtership 2','3\sigma formal uncertainty')
xlabel('Epochs (\Deltat)')
xlim([0 duration])

% D1 deltaV cov SAM plot

figure()
t = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile(t)
errorbar(epochs(mask_1), dv1_res_SAM(1,:)*1e6, 3*dv1_sig(1,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_1), dv1_res_SAM(1,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_X (SAM)')
ylabel('\DeltaV (mm/s)')
xlim([0 duration])

nexttile(t)
errorbar(epochs(mask_1), dv1_res_SAM(2,:)*1e6, 3*dv1_sig(2,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_1), dv1_res_SAM(2,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_Y (SAM)')
xlabel('Epochs (\Deltat)')
xlim([0 duration])

nexttile(t)
errorbar(epochs(mask_1), dv1_res_SAM(3,:)*1e6, 3*dv1_sig(3,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_1), dv1_res_SAM(3,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_Z (SAM)')
legend('3\sigma formal uncertainty','Daughtership 1 \DeltaV')
xlim([0 duration])

% D2 deltaV cov SAM plot

figure()
t = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile(t)
errorbar(epochs(mask_2), dv2_res_SAM(1,:)*1e6, 3*dv2_sig(1,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_2), dv2_res_SAM(1,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_X (SAM)')
ylabel('\DeltaV (mm/s)')
xlim([0 duration])

nexttile(t)
errorbar(epochs(mask_2), dv2_res_SAM(2,:)*1e6, 3*dv2_sig(2,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_2), dv2_res_SAM(2,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_Y (SAM)')
xlabel('Epochs (\Deltat)')
xlim([0 duration])

nexttile(t)
errorbar(epochs(mask_2), dv2_res_SAM(3,:)*1e6, 3*dv2_sig(3,:)*1e6, ...
    'LineStyle','None','Color','red','LineWidth',1.2)
hold on
scatter(epochs(mask_2), dv2_res_SAM(3,:)*1e6, 32, color1, "filled", 'o')
hold off
subtitle('\DeltaV_Z (SAM)')
legend('3\sigma formal uncertainty','Daughtership 2 \DeltaV')
xlim([0 duration])


% SAM total position plot

figure()
t = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% r_X (SAM)

nexttile(t)
plot(epochs, s_m_SAM(1,:), "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(epochs, s_1_SAM(1,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot(epochs, s_2_SAM(1,:), "+-", 'MarkerSize', 3, 'Color', color3)
hold off
ylabel('Position (km)')
subtitle('r_X (SAM)')
xlim([0, duration])

% r_Y (SAM)

nexttile(t)
plot(epochs, s_m_SAM(2,:), "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(epochs, s_1_SAM(2,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot(epochs, s_2_SAM(2,:), "+-", 'MarkerSize', 3, 'Color', color3)
hold off
subtitle('r_Y (SAM)')
xlim([0, duration])


% r_Z (SAM)

nexttile(t)
plot(epochs, s_m_SAM(3,:), "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(epochs, s_1_SAM(3,:), "square-", 'MarkerSize', 3, 'Color', color2)
plot(epochs, s_2_SAM(3,:), "+-", 'MarkerSize', 3, 'Color', color3)
hold off
ylabel('Position (km)')
xlabel('Epoch (\Deltat)')
xlim([0, duration])

% dmin

nexttile(t)
plot(epochs, dmin_hist_m, "o-", 'MarkerSize', 3, 'Color', color1)
hold on
plot(epochs, dmin_hist_1, "square-", 'MarkerSize', 3, 'Color', color2)
plot(epochs, dmin_hist_2, "+-", 'MarkerSize', 3, 'Color', color3)
hold off
xlabel('Epoch (\Deltat)')
subtitle('Min Distance to Ref')
legend('Mothership', 'Daughtership 1', 'Daughtership 2')
xlim([0, duration])

% true and predicted acceleration (SAM frame)

% rotate true & predicted accelerations into SAM frame

true_accel_m_SAM = zeros(3, duration);
true_accel_1_SAM = zeros(3, duration);
true_accel_2_SAM = zeros(3, duration);

pred_accel_m_SAM = zeros(3, duration);
pred_accel_1_SAM = zeros(3, duration);
pred_accel_2_SAM = zeros(3, duration);

for k = 1:duration
    Rk = R_SAM(:,:,k);

    true_accel_m_SAM(:,k) = Rk * true_accel_hist_m(:,k);
    true_accel_1_SAM(:,k) = Rk * true_accel_hist_1(:,k);
    true_accel_2_SAM(:,k) = Rk * true_accel_hist_2(:,k);

    pred_accel_m_SAM(:,k) = Rk * pred_accel_hist_m(:,k);
    pred_accel_1_SAM(:,k) = Rk * pred_accel_hist_1(:,k);
    pred_accel_2_SAM(:,k) = Rk * pred_accel_hist_2(:,k);
end

% percent acceleration difference (SAM frame)

% find max accel indices in SAM frame
[~, max_accel_m_x] = max(true_accel_m_SAM(1,2:end));
[~, max_accel_m_y] = max(true_accel_m_SAM(2,2:end));
[~, max_accel_m_z] = max(true_accel_m_SAM(3,2:end));

[~, max_accel_1_x] = max(true_accel_1_SAM(1,2:end));
[~, max_accel_1_y] = max(true_accel_1_SAM(2,2:end));
[~, max_accel_1_z] = max(true_accel_1_SAM(3,2:end));

[~, max_accel_2_x] = max(true_accel_2_SAM(1,2:end));
[~, max_accel_2_y] = max(true_accel_2_SAM(2,2:end));
[~, max_accel_2_z] = max(true_accel_2_SAM(3,2:end));


% delta V magnitudes without error (mm/s)

delta_v_1_mag_no_error = sum(vecnorm(delta_v_hist_1, 2, 1)*10^6)
delta_v_2_mag_no_error = sum(vecnorm(delta_v_hist_2, 2, 1)*10^6)

% delta V magnitudes with error (mm/s)

delta_v_1_mag_with_error = sum(vecnorm(delta_v_hist_1 + delta_v_error_hist_1, 2, 1)*10^6)
delta_v_2_mag_with_error = sum(vecnorm(delta_v_hist_2 + delta_v_error_hist_2, 2, 1)*10^6)