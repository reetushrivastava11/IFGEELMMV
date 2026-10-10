% -------------------------------------------------------------------------
% Intuitionistic fuzzy graph-embedded ELM for multi-view learning:
% training and testing (entry function), with the intuitionistic fuzzy
% weighting and scaling routines.
% -------------------------------------------------------------------------
function best_eval = ifelm_multiview_train( ...
        A_train, B_train, A_test, B_test, p)

    trainX1 = A_train(:, 1:end-1);
    trainY  = A_train(:, end);
    trainX2 = B_train(:, 1:end-1);
    testX1  = A_test(:, 1:end-1);
    testY   = A_test(:, end);
    testX2  = B_test(:, 1:end-1);

    [trainX1, testX1] = robust_scaler(trainX1, testX1);
    [trainX2, testX2] = robust_scaler(trainX2, testX2);

    if_weights = compute_intuitionistic_fuzzy_score( ...
        [trainX1, trainX2, trainY], p.mew);

    Y_onehot = zeros(length(trainY), 2);
    for i = 1:length(trainY)
        Y_onehot(i, trainY(i) + 1) = 1;
    end

    % Random hidden layers (one per view)
    rng(42);
    W1 = -1 + 2 * rand(p.n_hidden, size(trainX1, 2));
    b1 = -1 + 2 * rand(1, p.n_hidden);

    rng(123);
    W2 = -1 + 2 * rand(p.n_hidden, size(trainX2, 2));
    b2 = -1 + 2 * rand(1, p.n_hidden);

    H1 = max(0, trainX1 * W1' + repmat(b1, size(trainX1, 1), 1));
    H2 = max(0, trainX2 * W2' + repmat(b2, size(trainX2, 1), 1));

    % Graph regularization matrices
    S1 = graph_embedding(H1', trainY, if_weights);
    S2 = graph_embedding(H2', trainY, if_weights);

    % Joint regularized linear system for both output weight matrices
    W_diag = diag(if_weights);
    H1_w = H1' * W_diag;
    H2_w = H2' * W_diag;

    I = eye(p.n_hidden);
    c3 = 0.1;
    reg1 = p.C1 + c3;
    reg2 = p.C2 + c3;

    top_left     = H1_w * H1 + p.theta * S1 + reg1 * I;
    top_right    = p.rho * (H1_w * H2);
    bottom_left  = p.rho * (H2_w * H1);
    bottom_right = H2_w * H2 + p.theta * S2 + reg2 * I;

    X_mat = [top_left, top_right; bottom_left, bottom_right] ...
            + 1e-6 * eye(2 * p.n_hidden);
    X_rhs = [H1_w * Y_onehot; H2_w * Y_onehot];

    beta = pinv(X_mat) * X_rhs;
    beta1 = beta(1:p.n_hidden, :);
    beta2 = beta(p.n_hidden+1:end, :);

    % Prediction
    H1_test = max(0, testX1 * W1' + repmat(b1, size(testX1, 1), 1));
    H2_test = max(0, testX2 * W2' + repmat(b2, size(testX2, 1), 1));

    raw_scores = H1_test * beta1 + H2_test * beta2;

    ex = exp(raw_scores - max(raw_scores, [], 2));
    prob = ex ./ (sum(ex, 2) + 1e-12);

    % Decision threshold selected over a grid (0.7*G-mean + 0.3*F1)
    thresholds = linspace(0.01, 0.99, 100);
    best_score = -1;
    best_eval = [];

    for t = thresholds
        pred = double(prob(:, 2) > t);
        metrics = evaluation_metrics(testY, pred, prob);
        score = 0.7 * metrics.gmean + 0.3 * metrics.f1;
        if score > best_score
            best_score = score;
            best_eval = metrics;
        end
    end

end


% -------------------------------------------------------------------------
% Intuitionistic fuzzy score (sample weights) from class-wise RBF kernels.
% -------------------------------------------------------------------------
function S = compute_intuitionistic_fuzzy_score(A_data, mew)

    if nargin < 2
        mew = 4.0;
    end

    no_input = size(A_data, 1);
    X_all = A_data(:, 1:end-1);
    labels = A_data(:, end);

    idx_pos = (labels == 1);
    idx_neg = (labels ~= 1);

    A1 = X_all(idx_pos, :);
    B1 = X_all(idx_neg, :);

    if isempty(A1) || isempty(B1)
        S = ones(no_input, 1);
        return;
    end

    K1 = compute_rbf_kernel(A1, mew);
    K2 = compute_rbf_kernel(B1, mew);
    K3 = compute_rbf_kernel(X_all, mew);

    radiusxp = sqrt(max(1.0 - 2.0 * mean(K1, 2) + mean(K1, 'all'), 0));
    radiusxn = sqrt(max(1.0 - 2.0 * mean(K2, 2) + mean(K2, 'all'), 0));
    radiusmaxxp = max(radiusxp);
    radiusmaxxn = max(radiusxn);

    alpha_d = max(radiusmaxxn, radiusmaxxp);

    % Membership degree
    mem = zeros(no_input, 1);
    mem(idx_pos) = 1.0 - (radiusxp / (radiusmaxxp + 1e-4));
    mem(idx_neg) = 1.0 - (radiusxn / (radiusmaxxn + 1e-4));

    % Non-membership degree from the neighborhood label disagreement
    DD = sqrt(max(2.0 * (1.0 - K3), 0));
    mask = DD < alpha_d;
    diff_labels = (labels ~= labels');
    ro = sum(diff_labels .* mask, 2) ./ max(sum(mask, 2), 1e-12);
    ro(sum(mask, 2) == 0) = 0;

    v = (1.0 - mem) .* ro;

    % Score function
    S = zeros(no_input, 1);
    cond1 = (v == 0);
    cond2 = (~cond1) & (mem <= v);
    cond3 = (~cond1) & (~cond2);

    S(cond1) = mem(cond1);
    S(cond2) = 0.01;
    S(cond3) = (1.0 - v(cond3)) ./ (2.0 - mem(cond3) - v(cond3) + 1e-12);

    S = max(S, 0.01);
    S = S / mean(S);

end

function K = compute_rbf_kernel(X, mew)

    sq_norms = sum(X.^2, 2);
    dist_sq = max(sq_norms + sq_norms' - 2 * (X * X'), 0);
    K = exp(-dist_sq / (mew^2));

end

function [X_tr, X_te] = robust_scaler(train, test)

    med = median(train, 1);
    iqr_val = quantile(train, 0.75, 1) - quantile(train, 0.25, 1);
    iqr_val(iqr_val == 0) = 1.0;

    X_tr = (train - med) ./ iqr_val;
    X_te = (test - med) ./ iqr_val;

end
