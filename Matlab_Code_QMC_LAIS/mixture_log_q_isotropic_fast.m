function logq = mixture_log_q_isotropic_fast(Y, means, sigma)
% LOG of unnormalized proposal density for equal-weight isotropic mixture,
% via compiled mvnpdf (constants drop out after weight normalization).
% Y     : d x N
% means : d x J
% sigma : scalar

[~, N] = size(Y);
Sigma = (sigma^2) * eye(size(means,1));
den   = zeros(1, N);
Yt    = Y.';  % N x d

for j = 1:size(means,2)
    den = den + mvnpdf(Yt, means(:,j).', Sigma).';
end
logq = log(den);
end
