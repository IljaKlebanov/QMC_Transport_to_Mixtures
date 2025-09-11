clear; clc;
tic

% === User-defined parameters ===
% The following setup takes 1 min on my slow laptop: dim = 50; num_centers = 700; num_samples = 1000; 
dim = 2;                % Dimension (use 2 or 3 for plots)
num_centers = 7;        % Number of mixture components
num_samples = 30;      % Number of points (or level for sparse grid)
point_type = 'sparse_grid';     % 'MC', 'qMC', 'sparse_grid'

% === Fix seed for reproducibility ===
rng(1);


% % === Define Gaussian Mixture ===
% dim = 2;                % Dimension (use 2 or 3 for plots)
% num_centers = 3;        % Number of mixture components
% mixture.w = [0.3, 0.4, 0.3];
% mixture.a = [2, -1,  0;
%     -1,  0, -2];
% mixture.B = cat(3, ...
%     2/3 * eye(2), ...
%     2/3 * [1, 0; -1, 1], ...
%     2/3 * [2, 0; 0, 0.5]);


% === Generate random Gaussian mixture ===
% mixture.w = ones(1, num_centers) / num_centers;
raw_weights = rand(1, num_centers);
mixture.w = raw_weights / sum(raw_weights);
mixture.a = 3*randn(dim, num_centers);  % centers
% Random covariance via scaled Wishart
df = dim + 4;
mixture.B = zeros(dim, dim, num_centers);
for j = 1:num_centers
    C = wishrnd(eye(dim) / df, df) * dim;
    mixture.B(:,:,j) = chol(C, 'lower');  % use lower Cholesky for consistency
end

% === Transport points ===
[samp_1, weights, N_actual] = transport_points(mixture, point_type, num_samples);
fprintf('Number of transported points: %d\n', N_actual);

% === Plot Result (Final Figure Only) ===
switch dim
    case 1
        % === 1D: density curve and point markers ===
        margin = 1.5;
        x_min = min(samp_1) - margin;
        x_max = max(samp_1) + margin;
        x_vals = linspace(x_min, x_max, 1000);
        [rho_vals, ~] = evaluate_density_and_velocity_t(1, x_vals, mixture);

        figure('Position', [200, 200, 600, 300]); hold on;
        plot(x_vals, rho_vals, 'LineWidth', 2, 'Color', [0.2, 0.5, 0.8]);

        if strcmp(point_type, 'sparse_grid')
            pl = plot_weighted_sparse_grid(samp_1, weights);
            legend({'Density $\rho_1$', 'sparse grid samples'}, ...
                'Interpreter', 'latex', 'FontSize', 14);
        else
            plot(samp_1, -0.01 * ones(size(samp_1)), 'k|', 'MarkerSize', 10);
            legend({'Density $\rho_1$', 'transported points'}, ...
                'Interpreter', 'latex', 'FontSize', 14);
        end

        xlabel('x'); ylabel('density');
        ylim([-0.05, max(rho_vals)*1.1]);
        title('1D Transport: Density and Samples');
        grid on;

    case 2
        % === 2D: contour + transported points ===
        plot_grid_size = 100;
        margin = 1.5;
        mins = min(samp_1, [], 2) - margin;
        maxs = max(samp_1, [], 2) + margin;
        xx = linspace(mins(1), maxs(1), plot_grid_size);
        yy = linspace(mins(2), maxs(2), plot_grid_size);
        [XX, YY] = meshgrid(xx, yy);
        grid_points = [reshape(XX,1,[]); reshape(YY,1,[])];
        [rho, ~] = evaluate_density_and_velocity_t(1, grid_points, mixture);
        rho = reshape(rho, plot_grid_size, plot_grid_size);
        contour_lines = linspace(min(rho(:)), max(rho(:))*0.95, 25);

        figure('Position', [200, 200, 600, 600]); hold on; axis equal;
        contour(XX, YY, rho, contour_lines, 'LineWidth', 1.5);

        if strcmp(point_type, 'sparse_grid')
            pl = plot_weighted_sparse_grid(samp_1, weights);
            legend(pl, 'sparse grid transported by $T=\Phi_{1}$', ...
                'Interpreter', 'latex', 'Location', 'NorthEast', 'FontSize', 16);
        else
            pl = plot(samp_1(1,:), samp_1(2,:), 'b.', 'MarkerSize', 20);
            legend(pl, 'points transported by $T=\Phi_{1}$', ...
                'Interpreter', 'latex', 'Location', 'NorthEast', 'FontSize', 16);
        end

        xlabel('x'); ylabel('y');
        title('2D Transported Points and Target Density');
        grid on;

    case 3
        % === 3D: isosurface + transported points ===
        plot_grid_size = 50;
        margin = 1.5;
        mins = min(samp_1, [], 2) - margin;
        maxs = max(samp_1, [], 2) + margin;
        x_vals = linspace(mins(1), maxs(1), plot_grid_size);
        y_vals = linspace(mins(2), maxs(2), plot_grid_size);
        z_vals = linspace(mins(3), maxs(3), plot_grid_size);
        [XX, YY, ZZ] = meshgrid(x_vals, y_vals, z_vals);
        grid_points = [XX(:)'; YY(:)'; ZZ(:)'];
        [rho_vals, ~] = evaluate_density_and_velocity_t(1, grid_points, mixture);
        rho_vol = reshape(rho_vals, size(XX));
        iso_level = prctile(rho_vals, 85);

        figure('Position', [200, 200, 700, 600]); hold on;
        p = patch(isosurface(XX, YY, ZZ, rho_vol, iso_level));
        isonormals(XX, YY, ZZ, rho_vol, p);
        p.FaceColor = [0.2 0.5 0.9]; p.EdgeColor = 'none'; p.FaceAlpha = 0.4;

        if strcmp(point_type, 'sparse_grid')
            plot_weighted_sparse_grid(samp_1, weights);
        else
            scatter3(samp_1(1,:), samp_1(2,:), samp_1(3,:), 15, 'k', 'filled');
        end

        xlabel('x'); ylabel('y'); zlabel('z');
        title('3D Transport with High-Density Isosurface');
        axis equal; view(135, 30); grid on;
        camlight; lighting gouraud;
        legend('High-density region', 'Transported points');

    otherwise
        fprintf('Transport done. No plot available for dim = %d.\n', dim);
end

toc