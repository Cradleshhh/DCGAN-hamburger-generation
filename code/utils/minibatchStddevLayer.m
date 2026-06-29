function layer = minibatchStddevLayer(layerName)
    fcn = @(X) minibatchStddevFcn(X);
    layer = functionLayer(fcn, 'Name', layerName, 'Formattable', true);
end

function Y = minibatchStddevFcn(X)
    mu    = mean(X, 4);
    s     = sqrt(mean((X - mu).^2, 4) + 1e-8);
    s_avg = mean(s, 'all');
    s_map = repmat(s_avg, [size(X,1), size(X,2), 1, size(X,4)]);
    Y     = cat(3, X, s_map);
end
