% LAIS with QMC — error vs total samples (vary M), averaged over R runs.
% Upper layer: parallel MH chains (Martino et al., 2016).
% Lower layer: DM-LAIS vs TQMC-LAIS using ode45 transport.
% Uses fast mvnpdf-based log-sum for proposal q. Upper layer fixed across M.

clear all; close all; clc; rng default
tic
% Target
typeTar = 1;
[~,~,d,mu_true] = target(NaN, typeTar);

% LAIS configuration
N  = 10;              % chains
T  = 20;              % MH epochs
sig_prop        = 5;  % MH proposal std
sig_lower_layer = 13; % lower-layer σ

% Upper layer once (as in original M-plot)
mu_tot = proposal_chains_mh(N, T, sig_prop, typeTar); % d x (N*T)

% Replications and x-axis
R     = 20;
MList = floor(1.8.^(1:9));
Ntot  = MList * N * T;

% Storage
ErrDM = zeros(numel(MList), R);
ErrTQ = zeros(numel(MList), R);

for r = 1:R
    r
    % One scrambled Sobol set per run (reused across M)
    p  = sobolset(d); p = scramble(p,'MatousekAffineOwen');
    U_big = net(p, max(Ntot))';                          % d x maxN

    for im = 1:numel(MList)
        M = MList(im);

        % ---- DM-LAIS ----
        ErrDM(im,r) = dm_lais_full_error(mu_tot, M, sig_lower_layer, typeTar, mu_true);

        % ---- TQMC-LAIS (ode45 transport, reuse Sobol slice) ----
        num_samp = N * T * M;
        U01 = U_big(:, 1:num_samp);
        Y = transport_points_LAIS_from_U(mu_tot, sig_lower_layer, U01);

        logDEN = mixture_log_q_isotropic_fast(Y, mu_tot, sig_lower_layer);
        logNUM = target(Y.', typeTar); logNUM = logNUM(:).';
        w      = exp(logNUM - logDEN); w = w / sum(w);
        x_est  = Y * w.';
        ErrTQ(im,r) = norm(x_est - mu_true);
    end
end

save('cache_LAIS_over_M.mat','MList','Ntot','ErrDM','ErrTQ','N','T','sig_prop','sig_lower_layer','typeTar','R','d','mu_true','mu_tot')

%% ---- Plot: error vs total samples (log–log, LaTeX labels, no title) ----

% load('cache_LAIS_over_M.mat','MList','Ntot','ErrDM','ErrTQ','N','T','sig_prop','sig_lower_layer','typeTar','R','d','mu_true','mu_tot')

% Geometric means and bands
[DM_g, DM_lo, DM_hi] = geom_stats(ErrDM);
[TQ_g, TQ_lo, TQ_hi] = geom_stats(ErrTQ);
myfig = figure('Position',[200,200,500,400]);
delt = 0.01;
ax = axes('Units','normalized','Position',[13*delt, 13*delt, 1-14*delt, 1-14*delt]);
hold(ax,'on'); grid(ax,'on'); set(ax,'XScale','log','YScale','log');

shadeband(Ntot, DM_lo, DM_hi, [0 0 1]);
shadeband(Ntot, TQ_lo, TQ_hi, [1 0 0]);

loglog(Ntot, DM_g, 'b', 'LineWidth', 2);
loglog(Ntot, TQ_g, 'r', 'LineWidth', 2);
loglog(Ntot, 50  * Ntot.^(-0.5), 'k--', 'LineWidth', 1.5);
loglog(Ntot, 500 * Ntot.^(-1),   'k-.', 'LineWidth', 1.5);

xlabel('$N = C \cdot T \cdot M$', 'Interpreter','latex','FontSize',12);
ylabel('$\| \hat{S}^{\textup{snIS}}[f] - \mathbf{E}_{\mathbf{P}_{\textup{tar}}}[f]\|$', 'Interpreter','latex','FontSize',12);

h1 = legend('DM-LAIS','TQMC-LAIS','$\mathcal{O}(N^{-1/2})$','$\mathcal{O}(N^{-1})$');
set(h1,'Interpreter','latex','Location','SouthWest','Fontsize',15)

set(myfig,'Units','Inches'); pos = get(myfig,'Position');
set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(myfig,'Lais_with_QMC_over_M','-dpdf','-r0')


toc

% ---------------- local helpers ----------------
function mu_tot = proposal_chains_mh(N, T, sig_prop, typeTar)
[~,~,d] = target(NaN, typeTar);
x = -4 + 8*rand(N,d);
Sigma = sig_prop^2 * eye(d);
[logf0,~] = target(x, typeTar);
mu_tot = zeros(d, N*T);
col = 0;
for t = 1:T
    x_prop = mvnrnd(x, Sigma);
    logf1  = target(x_prop, typeTar);
    alpha  = min(1, exp(logf1 - logf0));
    u      = rand(N,1);
    acc    = (u <= alpha);
    x(acc,:)   = x_prop(acc,:);
    logf0(acc) = logf1(acc);
    for i = 1:N
        col = col + 1;
        mu_tot(:,col) = x(i,:).';
    end
end
end

function err = dm_lais_full_error(mu_tot, M, sigma, typeTar, mu_true)
d = size(mu_tot,1); J = size(mu_tot,2); %#ok<NASGU>
Z = sigma * randn(d, M*size(mu_tot,2));
X = zeros(d, M*size(mu_tot,2));
for j = 1:size(mu_tot,2)
    idx = (j-1)*M + (1:M);
    X(:,idx) = mu_tot(:,j) + Z(:,idx);
end
logDEN = mixture_log_q_isotropic_fast(X, mu_tot, sigma);
logNUM = target(X.', typeTar); logNUM = logNUM(:).';
w      = exp(logNUM - logDEN); w = w / sum(w);
x_est  = X * w.';
err    = norm(x_est - mu_true);
end

function [g, lo, hi] = geom_stats(ErrMat)
mu  = mean(log(ErrMat), 2).';
sig = std( log(ErrMat), 0, 2).';
g  = exp(mu); lo = exp(mu - sig); hi = exp(mu + sig);
end
