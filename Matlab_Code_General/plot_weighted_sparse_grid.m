function pl = plot_weighted_sparse_grid(samp, weights)
%PLOT_WEIGHTED_SPARSE_GRID  Visualize weighted sparse-grid nodes in 1D–3D.
%   pl = plot_weighted_sparse_grid(samp, weights) plots nodes stored column-wise
%   in samp (d x N) with signed quadrature weights (1 x N). Positive weights
%   are shown in blue, negative in cyan. Marker sizes scale with |w_i|.
%
%   Inputs
%     samp    - (d x N) node coordinates
%     weights - (1 x N) signed weights
%
%   Output
%     pl      - graphics handles for the last positive-weight plot in the
%               given dimension (useful for legend). For 3D, this is a scatter3
%               handle; for 1D/2D, a line handle.

[d, N] = size(samp);
if ~isvector(weights) || numel(weights) ~= N
    error('weights must be a 1 x N vector matching samp''s columns.');
end
weights = reshape(weights, 1, []);  %#ok<NASGU> (keep row shape)

hold on
pl = gobjects(1, 1);

% Size scaling: mild dependence on |w| and N to keep visibility across levels
sz = 15 * (abs(weights) * N).^0.2;

switch d
    case 1
        y0 = -0.01;
        for k = 1:N
            w = weights(k);
            if w == 0, continue; end
            col = 'b'; if w < 0, col = 'c'; end
            pl = plot(samp(1,k), y0, '|', 'Color', col, 'MarkerSize', sz(k));
        end

    case 2
        for k = 1:N
            w = weights(k);
            if w == 0, continue; end
            if w > 0
                pl = plot(samp(1,k), samp(2,k), 'b.', 'MarkerSize', sz(k));
            else
                plot(samp(1,k), samp(2,k), 'c.', 'MarkerSize', sz(k));
            end
        end

    case 3
        for k = 1:N
            w = weights(k);
            if w == 0, continue; end
            if w > 0
                pl = scatter3(samp(1,k), samp(2,k), samp(3,k), sz(k), 'b', 'filled');
            else
                scatter3(samp(1,k), samp(2,k), samp(3,k), sz(k), 'c', 'filled');
            end
        end

    otherwise
        error('Only dimensions d = 1, 2, or 3 are supported.');
end
end
