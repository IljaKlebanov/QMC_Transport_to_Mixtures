%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%   LAYERED ADAPTIVE IMPORTANCE SAMPLING (LAIS) with QMC  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MAIN_QMC_LAIS_density_and_samples.m

clear all
close all
clc

tic

% PRODUCE DM-LAIS AND QMC-LAIS SAMPLES (using LAIS-specific routines)
rng default
N = 10;                     % number of parallel chains
T = 20;                     % number of vertical and horizontal steps per epoch
M = 2;                      % samples per proposal pdfs in the lower layer
sig_prop = 5;               % std of the proposal pdfs of upper layer
typeTar = 1;
sig_lower_layer = 13;       % σ for lower-layer proposals (σ^2 I)
[~,~,d] = target(NaN, typeTar);

% Upper layer: centers via parallel MH (Martino et al., 2016)
mu_tot = proposal_chains_mh(N, T, sig_prop, typeTar);   % d x (N*T)
J = size(mu_tot, 2);

% DM-LAIS points: draw M per center from N(m_j, σ^2 I)
LAIS_points = zeros(d, M*J);
for j = 1:J
    idx = (j-1)*M + (1:M);
    LAIS_points(:, idx) = mu_tot(:, j) + sig_lower_layer * randn(d, M);
end

% TQMC-LAIS points: transport qMC through LAIS velocity field
p = sobolset(d); p = scramble(p, 'MatousekAffineOwen');
U01 = net(p, N*T*M)';                                   % d x (N*T*M)
QMC_points = transport_points_LAIS_from_U(mu_tot, sig_lower_layer, U01);

% PLOTTING: COMPUTE THE DENSITY
range = [-25,25,-25,25];
[XX,YY] = meshgrid( linspace(range(1),range(2),500) , linspace(range(3),range(4),600) );
logden = target([XX(:),YY(:)], typeTar)';
ZZ = exp(reshape(logden, size(XX,1), size(XX,2)));

% PLOTTING: DM-LAIS
myfig = figure('Position',[200,200,500,400]);
delt = 0.01; axes('Units', 'normalized', 'Position',[1*delt, 1*delt, 1-2*delt, 1-2*delt]); hold on;
contour(XX,YY,ZZ,15)
hold on
plot(mu_tot(1,:),mu_tot(2,:),'r.','MarkerSize',20)
plot(LAIS_points(1,:),LAIS_points(2,:),'k.','MarkerSize',7)
axis(range)
h1 = legend('target density $\rho_{\textup{tar}}$','centers $m_{i}$','DM-LAIS samples $z_{n}$');
set(h1,'Interpreter','latex','Location','SouthWest','Fontsize',12)
set(myfig,'Units','Inches'); pos = get(myfig,'Position'); set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(myfig,'Density_and_DM_Lais','-dpdf','-r0')

% PLOTTING: QMC-LAIS
myfig = figure('Position',[200,200,500,400]);
delt = 0.01; axes('Units', 'normalized', 'Position',[1*delt, 1*delt, 1-2*delt, 1-2*delt]); hold on;
contour(XX,YY,ZZ,15)
hold on
plot(mu_tot(1,:),mu_tot(2,:),'r.','MarkerSize',20)
plot(QMC_points(1,:),QMC_points(2,:),'k.','MarkerSize',7)
axis(range)
h1 = legend('target density $\rho_{\textup{tar}}$','centers $m_{i}$','TQMC-LAIS points $\tilde{z}_{n}$');
set(h1,'Interpreter','latex','Location','SouthWest','Fontsize',12)
set(myfig,'Units','Inches'); pos = get(myfig,'Position'); set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(myfig,'Density_and_QMC_Lais','-dpdf','-r0')

toc

% ---------------- local helper (upper layer) ----------------
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
