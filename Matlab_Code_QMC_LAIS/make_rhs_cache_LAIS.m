function C = make_rhs_cache_LAIS(means, sigma)
% Precompute constants for LAIS RHS with equal weights and σ^2 I.
C.means      = means;           % d x J
C.mt         = means.';         % J x d
C.invs2      = 1/(sigma^2);
C.inv2s2     = 0.5 * C.invs2;
C.mm2_scaled = sum(means.^2, 1).' * C.inv2s2;  % J x 1, ||m_j||^2/(2σ^2)
end
