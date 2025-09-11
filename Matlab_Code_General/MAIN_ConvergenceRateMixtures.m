function MAIN_ConvergenceRateMixtures
% Convergence of transported MC, transported qMC, and transported sparse grids
% for three 2D test integrands under a fixed Gaussian mixture.

tic
rng default

% Test integrands: returns 3-by-N when fed (y1,y2)
func = @(y1,y2) [ ...
    (y1 - 0.5).^2 + (y2 - 0.5).^2 ; ...
    4 * abs(2*y1 - 1) .* abs(2*y2 - 1) ; ...
    cos(0.3 + (y1 + y2)/2) ];
num_func = 3;

% Fixed Gaussian mixture (weights w, centers a, lower factors B)
numberGauss = 3;
w  = [0.3, 0.4, 0.3];
a  = [ 2, -1,  0; ...
      -1,  0, -2];
B  = zeros(2,2,numberGauss);
B(:,:,1) = 2/3 * eye(2);
B(:,:,2) = 2/3 * [1, 0; -1, 1];
B(:,:,3) = 2/3 * [2, 0; 0, 0.5];

mixture.w = w;
mixture.a = a;
mixture.B = B;

% Reference expectation via large per-component sampling
IntList      = zeros(num_func, numberGauss);
num_samp_ref = 30000000;
for j = 1:numberGauss
    Yj = B(:,:,j) * randn(2, num_samp_ref) + a(:,j);
    IntList(:,j) = mean(func(Yj(1,:), Yj(2,:)), 2);
end
true_int = IntList * (w.');

% Sample sizes
NList       = floor(1.93.^(2:14));
NSparseList = floor(linspace(1.2, 5.3, 15).^3);

% NList       = floor(1.93.^(2:10));
% NSparseList = floor(linspace(1.2, 3.3, 15).^3);


% Replicated runs for MC and TQMC
R = 50;
ErrMC_runs  = zeros(num_func, numel(NList), R);
ErrQMC_runs = zeros(num_func, numel(NList), R);

% MC and qMC (transported): R replicated runs
for k = 1:numel(NList)
    N = NList(k)
    for r = 1:R
        [Y, ~] = transport_points(mixture, 'MC', N);
        est = mean(func(Y(1,:), Y(2,:)), 2);
        ErrMC_runs(:,k,r) = abs(true_int - est);

        [Y, ~] = transport_points(mixture, 'qMC', N);
        est = mean(func(Y(1,:), Y(2,:)), 2);
        ErrQMC_runs(:,k,r) = abs(true_int - est);
    end
end

% Sparse grids (transported): single run per level
ErrorSparseList = zeros(num_func, numel(NSparseList));
for k = 1:numel(NSparseList)
    level = NSparseList(k);
    [Y, wts] = transport_points(mixture, 'sparse_grid', level);
    est = func(Y(1,:), Y(2,:)) * (wts.');
    ErrorSparseList(:,k) = abs(true_int - est);
    NSparseList(k) = numel(wts);   % convert level → actual N for the plot
end

% Aggregate MC/TQMC statistics in log-space (geometric mean ± multiplicative 1σ)
MC_logmean  = mean(log(ErrMC_runs),  3);
MC_logstd   = std( log(ErrMC_runs),  0, 3);
QMC_logmean = mean(log(ErrQMC_runs), 3);
QMC_logstd  = std( log(ErrQMC_runs), 0, 3);

MC_geom  = exp(MC_logmean);
MC_lo    = exp(MC_logmean - MC_logstd);
MC_hi    = exp(MC_logmean + MC_logstd);

QMC_geom = exp(QMC_logmean);
QMC_lo   = exp(QMC_logmean - QMC_logstd);
QMC_hi   = exp(QMC_logmean + QMC_logstd);

% Plots
NList_Rate = NList(2:end);
prefactors = [3, 0.2; 20, 2; 0.3, 0.03];
delt = 0.005;

for f = 1:num_func
    myfig = figure('Position', [200, 200, 520, 520]);
    ax = axes('Units','normalized','Position',[13*delt, 13*delt, 1-18*delt, 1-18*delt]);
    hold(ax, 'on'); grid(ax, 'on'); set(ax, 'XScale','log','YScale','log');

    % Colors
    co = get(ax, 'ColorOrder');
    cMC  = co(1,:);
    cQMC = co(2,:);
    cSG  = co(3,:);

    % Shaded multiplicative ±1σ bands
    shadeband(NList, MC_lo(f,:),  MC_hi(f,:),  cMC);
    shadeband(NList, QMC_lo(f,:), QMC_hi(f,:), cQMC);

    % Geometric-mean curves and sparse-grid curve
    hMC  = loglog(NList,       MC_geom(f,:),  'o-', 'LineWidth', 2, 'Color', cMC);
    hQMC = loglog(NList,       QMC_geom(f,:), '*-', 'LineWidth', 2, 'Color', cQMC);
    hSG  = loglog(NSparseList, ErrorSparseList(f,:), 's-', 'LineWidth', 2, 'Color', cSG);

    % Reference rates
    hRate1 = loglog(NList_Rate, prefactors(f,1) ./ sqrt(NList_Rate), 'k-.', 'LineWidth', 1.5);
    hRate2 = loglog(NList_Rate, prefactors(f,2) * (log(NList_Rate).^2) ./ NList_Rate, 'k--', 'LineWidth', 1.5);

    xlabel('N'); ylabel('|error|');
    h1 = legend([hMC hQMC hSG hRate1 hRate2], ...
        'MC (geom. mean)', 'TQMC (geom. mean)', 'TSG', ...
        '$\mathcal{O}(N^{-1/2})$', '$\mathcal{O}(N^{-1} (\\log N)^d)$');
    set(h1, 'Interpreter','latex', 'Location','SouthWest', 'FontSize', 18);

    set(myfig,'Units','Inches');
    pos = get(myfig,'Position');
    set(myfig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(myfig, append('TransportToMixtureErrorPlot_function_', num2str(f)), '-dpdf', '-r0');
end

toc
end

