function shadeband(x, ylo, yhi, colorVal)
% Shaded multiplicative ±1σ band on (log) axes
x = x(:)'; ylo = ylo(:)'; yhi = yhi(:)';
xx = [x, fliplr(x)];
yy = [yhi, fliplr(ylo)];
patch('XData', xx, 'YData', yy, ...
      'FaceColor', colorVal, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end