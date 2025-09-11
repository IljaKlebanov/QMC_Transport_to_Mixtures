function [density, velocity] = evaluate_density_and_velocity_t(t, x, mixture)
%EVALUATE_DENSITY_AND_VELOCITY_T  Interpolated density and velocity at time t.
%   [density, velocity] = evaluate_density_and_velocity_t(t, x, mixture)
%   computes the Gaussian-mixture density and corresponding velocity field
%   at interpolation time t in [0,1].
%
%   Inputs
%     t        - scalar in [0, 1]
%     x        - (d x N) evaluation points
%     mixture  - struct with fields:
%                  w : (1 x J) mixture weights
%                  a : (d x J) centers
%                  B : (d x d x J), each B(:,:,j) is the lower Cholesky
%                      factor of the component covariance at t=1
%
%   Outputs
%     density  - (1 x N) mixture density at time t
%     velocity - (d x N) velocity field at time t


[d, N] = size(x);
J = numel(mixture.w);

rho  = zeros(1, N);
flux = zeros(d, N);

c0 = (2*pi)^(-d/2);

for j = 1:J
    a_j  = mixture.a(:, j);
    B_j  = mixture.B(:, :, j);
    A_jt = t * B_j + (1 - t) * eye(d);

    % Pull back
    x0   = A_jt \ (x - t * a_j);
    v_jt = a_j + (B_j - eye(d)) * x0;

    % Sigma_t = A_jt*A_jt'
    R = chol(A_jt*A_jt.', 'lower');          % R*R' = Sigma_t
    log_det_inv_sqrt = -sum(log(diag(R)));   % log(det(Sigma_t)^(-1/2))

    D = x - t * a_j;
    y = R' \ (R \ D);                        % = Sigma_t^{-1} * D
    qf = sum(D .* y, 1);

    rho_j = c0 * exp(log_det_inv_sqrt - 0.5 * qf);

    rho  = rho  + mixture.w(j) * rho_j;
    flux = flux + mixture.w(j) * v_jt .* rho_j;
end

density = rho;
velocity = flux ./ rho;
end
