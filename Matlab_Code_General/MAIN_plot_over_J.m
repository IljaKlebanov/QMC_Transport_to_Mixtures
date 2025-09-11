% function MAIN_plot_over_J
% MC, TQMC, CQMC over J at fixed d and N with a costly integrand.
% Plots:
%   (1) Relative error vs J (geometric mean across M runs)
%   (2) Relative error vs total runtime [s] with J labels

clear all
close all
tic

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % SETUP FOR COSTLY INTEGRAND:
% fun_scalar = make_costly_func_stiff_vdp_int();
% K = 7;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP FOR CHEAP INTEGRAND:
fun_scalar = @(y) cos(0.3 + sum(y)/numel(y));
K = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% produce_cache_for_plot_over_J(K)

S = load('cache_for_plot_over_J.mat', 'd','N','J_list','M','Mixtures','Y_qmc','T_transport');
d = S.d; N = S.N; J_list = S.J_list; M = S.M;
Mixtures = S.Mixtures; Y_qmc = S.Y_qmc; T_transport = S.T_transport;



% Reference via TQMC with 1000 transported points (excluded from runtime)
N_ref = 2^(K+2);

RelErr_MC   = zeros(1, numel(J_list));
RelErr_TQMC = zeros(1, numel(J_list));
RelErr_CQMC = zeros(1, numel(J_list));
Time_MC     = zeros(1, numel(J_list));
Time_TQMC   = zeros(1, numel(J_list));
Time_CQMC   = zeros(1, numel(J_list));

for ij = 1:numel(J_list)
    J   = J_list(ij)
    mix = Mixtures{ij};

    % Reference (not timed)
    [Y_ref, ~] = transport_points(mix, 'qMC', N_ref);
    I_true     = mean(apply_fun_cols(fun_scalar, Y_ref));

    % MC: M runs
    eMC = zeros(1, M); tMC = zeros(1, M);
    for m = 1:M
        t0 = tic;
        r = randi(J, N, 1);
        Z = randn(d, N);
        Y = zeros(d, N);
        for j = 1:J
            idx = (r == j);
            if any(idx), Y(:,idx) = mix.a(:,j) + mix.B(:,:,j) * Z(:,idx); end
        end
        est   = mean(apply_fun_cols(fun_scalar, Y));
        tMC(m)= toc(t0);
        eMC(m)= abs(est - I_true) / abs(I_true);
    end

    % TQMC: M cached transports + eval time
    eTQ = zeros(1, M); tTQ = zeros(1, M);
    for m = 1:M
        Y      = Y_qmc{ij, m};
        t0     = tic;
        est    = mean(apply_fun_cols(fun_scalar, Y));
        t_eval = toc(t0);
        eTQ(m) = abs(est - I_true) / abs(I_true);
        tTQ(m) = T_transport(ij, m) + t_eval;
    end

    % CQMC: component-wise QMC (no transport); M runs
    eCQ  = zeros(1, M); tCQ = zeros(1, M);
    Ns_c = max(1, round(N / J));
    for m = 1:M
        t0 = tic;
        p = sobolset(d); p = scramble(p,'MatousekAffineOwen');
        U = net(p, Ns_c)'; Z = erfinv(2*U - 1) * sqrt(2);
        vals_c = zeros(1, J);
        for j = 1:J
            Yj = mix.a(:,j) + mix.B(:,:,j) * Z;
            vals_c(j) = mean(apply_fun_cols(fun_scalar, Yj));
        end
        est   = mean(vals_c);
        tCQ(m)= toc(t0);
        eCQ(m)= abs(est - I_true) / abs(I_true);
    end

    % Aggregate
    RelErr_MC(ij)   = exp(mean(log(eMC)));
    RelErr_TQMC(ij) = exp(mean(log(eTQ)));
    RelErr_CQMC(ij) = exp(mean(log(eCQ)));
    Time_MC(ij)     = mean(tMC);
    Time_TQMC(ij)   = mean(tTQ);
    Time_CQMC(ij)   = mean(tCQ);
end

%% ===== Plot 1: Error vs J (with bands), match style and filename =====
delt = 0.005;
myfig = figure('Position',[200,200,600,500]);
ax = axes('Units','normalized','Position',[22*delt, 22*delt, 1-23*delt, 1-23*delt]);
hold(ax,'on'); grid(ax,'on'); set(ax,'XScale','log','YScale','log');

co = get(ax,'ColorOrder');
cMC  = co(1,:); cTQ = co(2,:); cCQ = co(3,:);

% Geometric-mean curves
hMC = loglog(J_list, RelErr_MC,  'o-', 'LineWidth', 2, 'Color', cMC);
hTQ = loglog(J_list, RelErr_TQMC,'*-', 'LineWidth', 2, 'Color', cTQ);
hCQ = loglog(J_list, RelErr_CQMC,'d-', 'LineWidth', 2, 'Color', cCQ);

xlabel('$J$','Interpreter','latex','FontSize',16);
ylabel('Relative error','Interpreter','latex','FontSize',16);
h1 = legend('MC','TQMC','CQMC');
set(h1,'Interpreter','latex','Location','NorthWest','Fontsize',20)

set(myfig,'Units','Inches');
pos = get(myfig,'Position');
set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(myfig, append('MC_QMC_error_over_J_dim_', num2str(d)), '-dpdf','-r0')

%% ===== Plot 2: Error vs runtime (no bands) with J labels =====
myfig = figure('Position',[200,200,600,500]);
ax = axes('Units','normalized','Position',[22*delt, 22*delt, 1-23*delt, 1-23*delt]);
hold(ax,'on'); grid(ax,'on');

plot(Time_MC,   RelErr_MC,   'o-', 'LineWidth', 2, 'Color', cMC);
plot(Time_TQMC, RelErr_TQMC, '*-', 'LineWidth', 2, 'Color', cTQ);
plot(Time_CQMC, RelErr_CQMC, 'd-', 'LineWidth', 2, 'Color', cCQ);

xlabel('Total runtime [s]','Interpreter','latex','FontSize',16);
ylabel('Relative error','Interpreter','latex','FontSize',16);
h2 = legend('MC','TQMC','CQMC');
set(h2,'Interpreter','latex','Location','NorthEast','Fontsize',20)

for k = 1:numel(J_list)
    text(Time_MC(k)*1.01,   RelErr_MC(k)*1.03,   sprintf('J=%d', J_list(k)), 'Interpreter','latex', 'FontSize', 12);
    text(Time_TQMC(k)*1.01, RelErr_TQMC(k)*0.97, sprintf('J=%d', J_list(k)), 'Interpreter','latex', 'FontSize', 12);
    text(Time_CQMC(k)*1.01, RelErr_CQMC(k)*1.00, sprintf('J=%d', J_list(k)), 'Interpreter','latex', 'FontSize', 12);
end

set(myfig,'Units','Inches');
pos = get(myfig,'Position');
set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(myfig, sprintf('Error_vs_Runtime_dim_%d_costly_lowN', d), '-dpdf','-r0')


toc
% end

% -------- Local utilities --------
function vals = apply_fun_cols(fun_scalar, Y)
N = size(Y,2); vals = zeros(1, N);
for i = 1:N, vals(i) = fun_scalar(Y(:,i)); end
end

function fun = make_costly_func_stiff_vdp_int()
mu = 10; tEnd = 5; opts = odeset('RelTol',1e-6,'AbsTol',1e-6);
fun = @(y) stiff_vdp_eval(y(:), mu, tEnd, opts);
end

function val = stiff_vdp_eval(y, mu, tEnd, opts)
if numel(y) ~= 2, error('y must be a 2-vector'); end
f = @(t,x) [ x(2); mu*(1 - x(1)^2)*x(2) - x(1) + 0.01*sin(30*t) ];
F = @(t,xt) [ f(t, xt(1:2)); sum(xt(1:2).^2) ];
x0 = [y; 0]; [~,XT] = ode15s(F, [0 tEnd], x0, opts); val = XT(end,3);
end
