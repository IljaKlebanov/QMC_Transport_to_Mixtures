function [logp, p, DIM, mu_true, Marglike_true] = target(X, typeT)
% TARGET  Log-density and density of benchmark targets (mixtures of Gaussians).
%   [logp, p, DIM, mu_true, Marglike_true] = target(X, typeT)
%   - X: N-by-DIM matrix (rows are samples). If X is scalar NaN, returns only meta info.
%   - typeT: integer selector (1..5) as in the original code.
%   - logp, p: N-by-1 vectors (log-density and density).
%   - DIM: dimension of the target.
%   - mu_true: reference posterior mean used in LAIS experiments (unchanged).
%   - Marglike_true: placeholder 1 (kept for compatibility).
%
% Implementation details:
%   - Mixtures evaluated via log-sum-exp with Cholesky factors of covariances.
%   - No calls to mvnpdf; numerically stable and usually faster.
%   - Accepts both N-by-D and D-by-N (auto-transposes to N-by-D).

persistent CACHED
if isempty(CACHED) || ~isfield(CACHED, 'typeT') || CACHED.typeT ~= typeT
    CACHED = setup_params(typeT);
end
DIM           = CACHED.DIM;
mu_true       = CACHED.mu_true;
Marglike_true = 1;

% Meta-query: dimensions only
if isscalar(X) && isnan(X)
    logp = []; p = [];
    return
end

% Ensure N-by-DIM
if size(X,2) ~= DIM && size(X,1) == DIM
    X = X.';  % D-by-N -> N-by-D
end
if size(X,2) ~= DIM
    error('X must be N-by-%d or %d-by-N.', DIM, DIM);
end
N = size(X,1);
K = CACHED.K;

% Per-component log-density: log N(x; mu_k, Sigma_k)
% log N = const_k - 0.5 * || L_k \ (x - mu_k) ||^2, with L_k lower Cholesky
logNk = zeros(N, K);
for k = 1:K
    L  = CACHED.L(:,:,k);         % DIM-by-DIM, lower
    xc = (X - CACHED.mu(k,:));    % N-by-DIM
    R  = (L \ xc.').';            % N-by-DIM (solve per row via transpose trick)
    quad = sum(R.^2, 2);          % N-by-1
    logNk(:,k) = CACHED.logc(k) - 0.5 * quad;
end

% Mixture via log-sum-exp with weights
Z      = logNk + CACHED.logw;     % N-by-K
Zmax   = max(Z, [], 2);
logp   = Zmax + log(sum(exp(Z - Zmax), 2));
if nargout > 1
    p = exp(logp);
end
end

% ---------- parameters per target type (means, covariances, Cholesky, etc.) ----------
function C = setup_params(typeT)
C.typeT = typeT;

switch typeT
    case 1  % 2D, five-component mixture (equal weights)
        C.DIM = 2;
        mu = [ -10 -10;
                 0  16;
                13   8;
                 -9  7;
                14 -14 ];
        Sigma(:,:,1) = [2  0.6;  0.6 1];
        Sigma(:,:,2) = [2 -0.4; -0.4 2];
        Sigma(:,:,3) = [2  0.8;  0.8 2];
        Sigma(:,:,4) = [3  0;    0   0.5];
        Sigma(:,:,5) = [2 -0.1; -0.1 2];
        w = ones(5,1) / 5;
        C.mu_true = [1.6; 1.4];

    case 2  % 2D, single Gaussian
        C.DIM = 2;
        mu = [0 16];
        Sigma = zeros(2,2,1); Sigma(:,:,1) = [3 0; 0 3];
        w = 1;
        C.mu_true = [0; 16];

    case 3  % 2D, two-component mixture (equal weights)
        C.DIM = 2;
        mu = [ 5  0;
               0 16 ];
        Sigma(:,:,1) = [2  0.6; 0.6 1];
        Sigma(:,:,2) = [3  0;   0   3];
        w = [0.5; 0.5];
        C.mu_true = [2.5; 8];

    case 4  % 4D, single isotropic Gaussian with variance 4
        C.DIM = 4;
        mu = [0 16 5 -5];
        sig = 4;
        Sigma = zeros(4,4,1); Sigma(:,:,1) = sig * eye(4);
        w = 1;
        C.mu_true = [0; 16; 5; -5];

    case 5  % 10D, single isotropic Gaussian with variance 4, mean 5*ones
        C.DIM = 10;
        mu = 5 * ones(1,10);
        sig = 4;
        Sigma = zeros(10,10,1); Sigma(:,:,1) = sig * eye(10);
        w = 1;
        C.mu_true = 5 * ones(10,1);

    otherwise
        error('Unknown typeT = %d.', typeT);
end

% Store component means row-wise (K rows) for fast row-subtraction
K = size(mu,1);
C.K   = K;
C.mu  = mu;                    % K-by-DIM
C.L   = zeros(C.DIM, C.DIM, K);
C.logc = zeros(1, K);
for k = 1:K
    L = chol(Sigma(:,:,k), 'lower');
    C.L(:,:,k) = L;
    C.logc(k)  = -0.5*C.DIM*log(2*pi) - sum(log(diag(L)));   % log normalizer
end
C.logw = log(w(:)).';          % 1-by-K
end
