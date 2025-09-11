function MAIN_convergence_plots_over_N_varying_d_J
% Compare MC and transported QMC over N for grids of d and J.
% Plots match the original formatting and filenames, with a 2D density plot when d=2.

clear all
close all
tic

% % If not yet performed before:
% produce_cache_for_convergence_plots_over_N

S = load('cache_for_convergence_plots_over_N.mat', ...
         'dim_list','J_list','N_list','M','Mixtures','Y_qmc');
dim_list = S.dim_list; J_list = S.J_list; N_list = S.N_list;
M = S.M; Mixtures = S.Mixtures; Y_qmc = S.Y_qmc;

% Choose one scalar integrand: 'f1','f2','f3','f4','f4tilde','f5','f5tilde','f6','f6tilde','f7','f8','f9','f10','f10tilde'
choose_f = 'f9';
f = @(Y) select_scalar_integrand(Y, choose_f);

% Seed for reproducibility of MC runs
seed=1; randn('state',seed); rand('state',seed);

for dim_ind = 1:numel(dim_list)
    dim = dim_list(dim_ind);
    for num_centers_ind = 1:numel(J_list)
        num_centers = J_list(num_centers_ind);
        mix = Mixtures{dim_ind, num_centers_ind};

        % Reference integral via per-component MC (reused across N)
        num_samp_true_int = max(1, round(2^20 / num_centers));
        true_int_temp = zeros(1, num_centers);
        for j = 1:num_centers
            Z = randn(dim, num_samp_true_int);
            Yj = mix.a(:,j) + mix.B(:,:,j) * Z;
            true_int_temp(j) = mean(f(Yj));
        end
        true_int = mean(true_int_temp);

        % Geometric-mean stats over M runs (relative errors)
        MC_geom  = zeros(1, numel(N_list));
        MC_lo    = zeros(1, numel(N_list));
        MC_hi    = zeros(1, numel(N_list));
        QMC_geom = zeros(1, numel(N_list));
        QMC_lo   = zeros(1, numel(N_list));
        QMC_hi   = zeros(1, numel(N_list));

        for N_ind = 1:numel(N_list)
            num_samples = N_list(N_ind);

            % MC: M runs
            eMC = zeros(1, M);
            for m = 1:M
                samp_0 = randn(dim, num_samples);
                r = randi(num_centers, num_samples, 1);
                MC_samples = zeros(dim, num_samples);
                for j = 1:num_samples
                    MC_samples(:,j) = mix.a(:,r(j)) + mix.B(:,:,r(j)) * samp_0(:,j);
                end
                MC_int = mean(f(MC_samples));
                eMC(m) = abs(MC_int - true_int);
            end
            eMC_rel = eMC / abs(true_int);

            % TQMC: M cached runs
            eTQ = zeros(1, M);
            for m = 1:M
                Y = Y_qmc{dim_ind, num_centers_ind, N_ind, m};
                QMC_int = mean(f(Y));
                eTQ(m) = abs(QMC_int - true_int);
            end
            eTQ_rel = eTQ / abs(true_int);

            % Log-space aggregation: geometric mean ± multiplicative 1σ
            mc_mu  = mean(log(eMC_rel));
            mc_sig = std( log(eMC_rel), 0, 2);
            q_mu   = mean(log(eTQ_rel));
            q_sig  = std( log(eTQ_rel), 0, 2);

            MC_geom(N_ind) = exp(mc_mu);
            MC_lo(N_ind)   = exp(mc_mu - mc_sig);
            MC_hi(N_ind)   = exp(mc_mu + mc_sig);

            QMC_geom(N_ind) = exp(q_mu);
            QMC_lo(N_ind)   = exp(q_mu - q_sig);
            QMC_hi(N_ind)   = exp(q_mu + q_sig);
        end

        % Plot (match original style)
        delt = 0.005;
        x_add = 0; y_add = 0; x_delt_add = 0; y_delt_add = 0;
        if dim == dim_list(1), x_add = 30; x_delt_add = 19; end
        if num_centers == J_list(end), y_add = 25; y_delt_add = 15; end

        myfig = figure('Position',[200-x_add,200-y_add,300+x_add,300+y_add]);
        ax = axes('Units','normalized','Position',[(1+x_delt_add)*delt, (1+y_delt_add)*delt, 1-(2+x_delt_add)*delt, 1-(24+y_delt_add)*delt]);
        hold(ax, 'on'); grid(ax, 'on'); set(ax, 'XScale','log','YScale','log');

        % Colors
        co = get(ax, 'ColorOrder');
        cMC  = co(1,:);
        cQMC = co(2,:);

        % Shaded multiplicative ±1σ bands (no legend entries)
        shadeband(N_list, MC_lo,  MC_hi,  cMC);
        shadeband(N_list, QMC_lo, QMC_hi, cQMC);

        % Geometric-mean curves (legend shows only these two)
        hMC  = loglog(N_list, MC_geom,  'o-', 'LineWidth', 2, 'Color', cMC);
        hQMC = loglog(N_list, QMC_geom, '*-', 'LineWidth', 2, 'Color', cQMC);

        h1 = legend([hMC hQMC], 'MC', 'TQMC');
        axis([N_list(1), N_list(end), 1e-5, 1])
        set(h1,'Interpreter','latex','Location','SouthWest','Fontsize',14)
        my_title = title(append('$d=$ ',num2str(dim),', $J=$ ',num2str(num_centers)));
        set(my_title,'Interpreter','latex','Fontsize',14)
        set(myfig,'Units','Inches');
        pos = get(myfig,'Position');
        set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        print(myfig, append('MC_QMC_error_over_N_dim_',num2str(dim),'_centers_',num2str(num_centers)), '-dpdf','-r0')

        pause(0.3)

        % Density visualization when dim = 2 (match original style)
        if dim == 2
            ran_plot = 5;
            N_1d = 200;
            plot_range = [ -ran_plot, ran_plot; -ran_plot, ran_plot ];
            xrange = linspace(plot_range(1,1), plot_range(1,2), N_1d);
            yrange = linspace(plot_range(2,1), plot_range(2,2), N_1d);
            [xvals, yvals] = meshgrid(xrange, yrange); gridvals = [xvals(:) yvals(:)];
            vals = zeros(size(gridvals, 1), 1);
            weights = ones(1, num_centers) / num_centers;
            for k = 1:num_centers
                Sigma_k = mix.B(:,:,k) * mix.B(:,:,k)';      % covariance
                vals = vals + weights(k) .* mvnpdf(gridvals, mix.a(:,k)', Sigma_k);
            end
            myfig = figure('Position',[200,200-y_add,300,300+y_add]);
            axes('Units','normalized','Position',[1*delt, (1+y_delt_add)*delt, 1-2*delt, 1-(24+y_delt_add)*delt]);
            hold on
            contour(xvals, yvals, reshape(vals, N_1d, N_1d), 20)
            plot(mix.a(1,:), mix.a(2,:), 'r.', 'Markersize', 20)
            axis(reshape(plot_range',1,4));
            set(gca,'xticklabel',{[]})
            my_title = title(append('$d=$ ',num2str(dim),', $J=$ ',num2str(num_centers)));
            set(my_title,'Interpreter','latex','Fontsize',14)
            set(myfig,'Units','Inches');
            pos = get(myfig,'Position');
            set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
            print(myfig, append('mixture_with_',num2str(num_centers),'_centers'), '-dpdf','-r0')
        end
    end
end

toc
end


% Local integrand selector (scalar f: R^d -> R)
function val = select_scalar_integrand(Y, name)
% Y is d-by-N; returns 1-by-N
d = size(Y,1);
ystar = 0.5;

switch lower(name)
    case 'f1'       % ||y - y*||_1
        val = sum(abs(Y - ystar), 1);

    case 'f2'       % ||y - y*||_2^2
        val = sum((Y - ystar).^2, 1);

    case 'f3'       % (sum_j (2 y_j - 1))^4
        val = (sum(2*Y - 1, 1)).^4;

    case 'f4'       % prod_j 2 |2 y_j - 1|
        val = prod(2*abs(2*Y - 1), 1);

    case 'f4tilde'  % f4^(1/d)
        val = (prod(2*abs(2*Y - 1), 1)).^(1/d);

    case 'f5'       % prod_j (pi/2) sin(pi y_j)
        val = prod((pi/2) * sin(pi*Y), 1);

    case 'f5tilde'  % |f5|^(1/d) * sgn(f5)
        g = prod((pi/2) * sin(pi*Y), 1);
        val = sign(g) .* abs(g).^(1/d);

    case 'f6'       % prod_j (1 + |y_j - y*_j|^2)^(-1)
        val = prod(1 ./ (1 + abs(Y - ystar).^2), 1);

    case 'f6tilde'  % f6^(1/d)
        val = (prod(1 ./ (1 + abs(Y - ystar).^2), 1)).^(1/d);

    case 'f7'       % exp( - d^{-2} ||y - y*||_2^2 )
        val = exp(- sum((Y - ystar).^2, 1) / d^2);

    case 'f8'       % exp( d^{-1} sum_j y_j )
        val = exp(sum(Y, 1) / d);

    case 'f9'       % cos( 0.3 + d^{-1} sum_j y_j )
        val = cos(0.3 + sum(Y, 1) / d);

    case 'f10'      % 1_{B_{1/2}(y*)}
        val = double(sum((Y - ystar).^2, 1) <= (1/2)^2);

    case 'f10tilde' % 1_{B_d(y*)}
        val = double(sum((Y - ystar).^2, 1) <= d^2);

    otherwise
        error('Unknown integrand name: %s', name);
end
end

