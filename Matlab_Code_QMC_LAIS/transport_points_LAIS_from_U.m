function Y = transport_points_LAIS_from_U(means, sigma, U01)
% Transport qMC points for LAIS (equal weights, σ^2 I) via a single ode45 call.
% means : d x J component means
% sigma : scalar std (covariance σ^2 I)
% U01   : d x N points in [0,1] (e.g., Sobol), reused across runs

[d, N] = size(U01);
Z0 = erfinv(2*U01 - 1) * sqrt(2);  % N(0,1)
X0 = sigma * Z0;                   % N(0, σ^2 I)

C   = make_rhs_cache_LAIS(means, sigma);
rhs = @(t,z) rhs_transport_LAIS(t, z, C);

[~, sol] = ode45(rhs, [0, 1], X0(:));   % default tolerances (no opts)
Y = reshape(sol(end,:).', d, N);
end
