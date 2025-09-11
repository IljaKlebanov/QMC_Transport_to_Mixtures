function produce_cache_for_plot_over_J(K)
% Cache transported TQMC samples (and transport times) for fixed (d,N) across J.
tic
rng default

d      = 2;
% N      = 2^12;
% J_list = 2.^(0:11);
N      = 2^K;
J_list = 2.^(0:K-1);
M      = 20;

Mixtures     = cell(numel(J_list), 1);
Y_qmc        = cell(numel(J_list), M);
T_transport  = zeros(numel(J_list), M);

for ij = 1:numel(J_list)
    J = J_list(ij)

    mix.w = ones(1, J) / J;
    mix.a = randn(d, J);
    df    = d + 4;
    mix.B = zeros(d, d, J);
    for j = 1:J
        C = wishrnd(eye(d)/df, df) * d;
        mix.B(:,:,j) = chol(C, 'lower');   % lower Cholesky of covariance
    end
    Mixtures{ij} = mix;

    for m = 1:M
        rng(1000*ij + m);
        t0 = tic;
        [Y, ~] = transport_points(mix, 'qMC', N);
        T_transport(ij, m) = toc(t0);
        Y_qmc{ij, m} = Y;
    end
end

save('cache_for_plot_over_J.mat', ...
     'd','N','J_list','M','Mixtures','Y_qmc','T_transport', '-v7.3');
toc
end
