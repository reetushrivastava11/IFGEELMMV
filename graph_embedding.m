% Graph embedding: fuzzy-weighted LFDA graph matrices and the graph
% regularization matrix used by the model.
function GE = graph_embedding(train_data, train_lbls, fuzzy_weights)

    [Sw, Sb] = matrices_LFDA_fuzzy(train_data, train_lbls, fuzzy_weights);
    GE = pinv(Sb + 1e-3 * eye(size(Sb, 1))) * Sw;

end


% -------------------------------------------------------------------------
% Fuzzy-weighted LFDA graph matrices and graph regularization matrix.
% -------------------------------------------------------------------------
function [Sw, Sb] = matrices_LFDA_fuzzy(data, lbls, fuzzy_weights)

    [d, N] = size(data);

    if N < 3
        Sw = eye(d) * 1e-6;
        Sb = eye(d) * 1e-6;
        return;
    end

    norms = sqrt(sum(data.^2, 1)) + 1e-12;
    data_norm = data ./ norms;

    Dmat = sum(data_norm'.^2, 2) + sum(data_norm'.^2, 2)' ...
           - 2 * (data_norm' * data_norm);
    Dmat = max(Dmat, 0);

    k = min(7, N - 1);
    sorted_d = sort(Dmat, 2);
    sigma2 = max(mean(sorted_d(:, k+1)) + 1e-12, 1e-8);

    Amat = exp(-Dmat / (2 * sigma2));
    if ~isempty(fuzzy_weights)
        Amat = Amat .* (fuzzy_weights * fuzzy_weights');
    end

    same_class = (lbls == lbls');
    diff_class = ~same_class;

    unique_labels = unique(lbls);
    Nc_vec = zeros(N, 1);
    max_count = 0;

    for c = 1:length(unique_labels)
        cnt = sum(lbls == unique_labels(c));
        Nc_vec(lbls == unique_labels(c)) = cnt;
        max_count = max(max_count, cnt);
    end

    Ww = (Amat ./ Nc_vec) .* same_class;
    Wb = (Amat .* ((max_count ./ Nc_vec) / N)) .* diff_class;

    Ww = (Ww + Ww') / 2;
    Wb = (Wb + Wb') / 2;

    Lw = diag(sum(Ww, 2)) - Ww;
    Lb = diag(sum(Wb, 2)) - Wb;

    Sw = data * Lw * data' + 1e-6 * eye(d);
    Sb = data * Lb * data' + 1e-6 * eye(d);

end
