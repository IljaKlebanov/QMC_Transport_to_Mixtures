function dxdt = rhs_transport(t, x, mixture)
%RHS_TRANSPORT  ODE RHS for transport to a Gaussian mixture (stable solves).
%   dxdt = rhs_transport(t, x, mixture) evaluates the velocity field at time t
%   for the stacked state x (d*N x 1). The global Gaussian constant cancels and
%   is omitted.
%
%   Inputs
%     t        - scalar in [0,1]
%     x        - (d*N x 1) state vector storing N points stacked column-wise
%     mixture  - struct with fields:
%                  w : (1 x J) mixture weights
%                  a : (d x J) centers
%                  B : (d x d x J), each B(:,:,j) is the lower Cholesky
%                      factor of the component covariance at t=1
%
%   Output
%     dxdt     - (d*N x 1) velocity evaluated at the current state


d  = size(mixture.a, 1);
x  = x(:);
N  = numel(x) / d;
X  = reshape(x, d, N);

J    = numel(mixture.w);
rho  = zeros(1, N);
flux = zeros(d, N);

I = eye(d);

for j = 1:J
    a_j  = mixture.a(:, j);
    B_j  = mixture.B(:, :, j);
    A_jt = t * B_j + (1 - t) * I;

    % Pull back and velocity contribution for component j
    X0   = A_jt \ (X - t * a_j);
    v_jt = a_j + (B_j - I) * X0;

    % Sigma_t = A_jt*A_jt'  → Cholesky and triangular solves
    R = chol(A_jt * A_jt.', 'lower');         % R*R' = Sigma_t
    log_det_inv_sqrt = -sum(log(diag(R)));    % log det(Sigma_t)^(-1/2)

    D = X - t * a_j;
    y = R' \ (R \ D);                          % y = Sigma_t^{-1} * D
    q = sum(D .* y, 1);

    rho_j = exp(log_det_inv_sqrt - 0.5 * q);

    wj   = mixture.w(j);
    rho  = rho  + wj * rho_j;
    flux = flux + wj * v_jt .* rho_j;
end

V    = flux ./ rho;
dxdt = V(:);
end
