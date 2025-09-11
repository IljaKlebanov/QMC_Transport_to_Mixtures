function [samp_1, weights, num_points] = transport_points(mixture, point_type, num_samples)
%TRANSPORT_POINTS  Transport nodes from N(0,I) to a Gaussian mixture.
%   [samp_1, weights, num_points] = transport_points(mixture, point_type, num_samples)
%   generates initial nodes according to the chosen rule, then transports them
%   via the ODE flow to time t=1.
%
%   Inputs
%     mixture     - struct with fields:
%                     w : (1 x J) mixture weights
%                     a : (d x J) centers
%                     B : (d x d x J), each B(:,:,j) is the lower Cholesky
%                         factor of the component covariance at t=1
%     point_type  - 'MC', 'qMC', or 'sparse_grid'
%     num_samples - number of points (for MC/qMC) or level (for sparse grids)
%
%   Outputs
%     samp_1      - (d x N) transported nodes
%     weights     - (1 x N) quadrature weights (uniform for MC/qMC)
%     num_points  - number of generated nodes


d = size(mixture.a, 1);

switch point_type
    case 'MC'
        samp_0  = randn(d, num_samples);
        weights = ones(1, num_samples) / num_samples;

    case 'qMC'
%         U = rhalton_owen(num_samples, d);              % d x num_samples in (0,1)

        p = sobolset(d);
        p = scramble(p,'MatousekAffineOwen');
        U = net(p,num_samples)';                  % d x num_samples in (0,1)
%         U = 1 - abs(2*U - 1);                     % tent transform

        samp_0  = erfinv(2*U - 1) * sqrt(2);
        weights = ones(1, num_samples) / num_samples;

    case 'sparse_grid'
%         level   = num_samples;
%         knots   = @(n) knots_gaussian(n,0,1);
%         S       = smolyak_grid(d, level, knots, @lev2knots_lin);
%         Sr      = reduce_sparse_grid(S);
%         samp_0  = Sr.knots;
%         weights = Sr.weights;
        level = num_samples;
        knots=@(n) knots_normal_leja(n,0,1,'sym_line'); % knots              
        S  = smolyak_grid(d,level,knots,@lev2knots_lin); % grid
        Sr = reduce_sparse_grid(S);
        samp_0  = Sr.knots;
        weights = Sr.weights;

    otherwise
        error('Unknown point_type: %s', point_type);
end

num_points = size(samp_0, 2);

rhs = @(t, x) rhs_transport(t, x, mixture);
[~, path] = ode45(rhs, [0, 1], samp_0(:));

samp_1 = reshape(path(end,:).', d, []);
end
