% Evaluation metrics: G-mean and AUC (in percent).
% The F1-score is computed only because it is part of the decision-threshold
% selection objective used during training.
function metrics = evaluation_metrics(ACTUAL, PREDICTED, PROB)

    ACTUAL = ACTUAL(:);
    PREDICTED = PREDICTED(:);

    tp = sum((ACTUAL == 1) & (PREDICTED == 1));
    tn = sum((ACTUAL == 0) & (PREDICTED == 0));
    fp = sum((ACTUAL == 0) & (PREDICTED == 1));
    fn = sum((ACTUAL == 1) & (PREDICTED == 0));

    if (2*tp + fp + fn) == 0
        f1 = 0;
    else
        f1 = 100 * (2*tp) / (2*tp + fp + fn);
    end

    auc = 50.0;
    if nargin >= 3 && ~isempty(PROB) && length(unique(ACTUAL)) > 1
        try
            [~, ~, ~, auc_val] = perfcurve(ACTUAL, PROB(:, 2), 1);
            auc = 100 * auc_val;
        catch
            auc = 50.0;
        end
    end

    sens = tp / (tp + fn + 1e-12);
    spec = tn / (tn + fp + 1e-12);
    gmean = 100 * sqrt(sens * spec);

    metrics.gmean = gmean;
    metrics.auc   = auc;
    metrics.f1    = f1;

end
