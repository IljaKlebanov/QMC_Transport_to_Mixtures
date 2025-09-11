function produce_cache_for_convergence_plots_over_N
% Cache transported TQMC samples for grids of dimensions, component counts, and N.
tic
rng default

% Configuration
dim_list = [2, 5, 20, 50];
J_list   = [2, 5, 20];
N_list   = 2.^(2:13);
M        = 20;

% Storage
Mixtures = cell(numel(dim_list), numel(J_list));
Y_qmc    = cell(numel(dim_list), numel(J_list), numel(N_list), M);

for id = 1:numel(dim_list)
    d = dim_list(id)
    for ij = 1:numel(J_list)
        J = J_list(ij)

        % Mixture: equal weights; B(:,:,j) lower Cholesky of covariance
        mix.w = ones(1, J) / J;
        mix.a = randn(d, J);
        df    = d + 4;
        mix.B = zeros(d, d, J);
        for j = 1:J
            C = wishrnd(eye(d)/df, df) * d;
            mix.B(:,:,j) = chol(C, 'lower');
        end
        Mixtures{id, ij} = mix;

        % Transported TQMC samples for all N and M runs
        for in = 1:numel(N_list)
            N = N_list(in);
            for m = 1:M
                rng(10000*id + 100*ij + 10*in + m);
                [Y, ~] = transport_points(mix, 'qMC', N);
                Y_qmc{id, ij, in, m} = Y;
            end
        end
    end
end

save('cache_for_convergence_plots_over_N.mat', ...
     'dim_list','J_list','N_list','M','Mixtures','Y_qmc', '-v7.3');
toc
end
