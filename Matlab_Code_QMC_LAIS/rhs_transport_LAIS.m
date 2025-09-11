function dxdt = rhs_transport_LAIS(t, x, C)
% Velocity for LAIS transport (equal weights, σ^2 I), using precomputed cache C.
% t  : scalar
% x  : vectorized state (length d*N), reshaped to d x N
% C  : struct from make_rhs_cache_LAIS

d   = size(C.means,1);
X   = reshape(x, d, []);
sx  = sum(X.^2, 1);              % 1 x N
mx  = C.mt * X;                  % J x N

% E = -(||x||^2)/(2σ^2) + (t/σ^2)(m^T x) - (t^2)||m||^2/(2σ^2)
E = bsxfun(@minus,  C.invs2 * (t * mx), C.inv2s2 * sx);       % JxN - 1xN
E = bsxfun(@minus,  E, (t^2) * C.mm2_scaled);                 % subtract Jx1
E = bsxfun(@minus,  E, max(E, [], 1));                        % stabilize
W = exp(E);                                                   % J x N
den  = sum(W, 1);                                            % 1 x N
flux = C.means * W;                                          % d x N
V    = bsxfun(@rdivide, flux, den);                          % d x N

dxdt = V(:);
end
