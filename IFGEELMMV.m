% IFGEELMMV: Intuitionistic Fuzzy Graph-Embedded Extreme Learning Machine
% for Multi-View Learning.
%
% Usage: place this file in the same folder as the "datasets" folder
% (CSV files, last column = binary class label) and run the script.
% Supporting functions are in the "functions" folder.
% Output: IFGEELMMV_results.csv

clc;
clear;
close all;
warning('off', 'all');

script_folder = fileparts(mfilename('fullpath'));
folder_path   = fullfile(script_folder, 'datasets');
output_csv    = fullfile(script_folder, 'IFGEELMMV_results.csv');

addpath(fullfile(script_folder, 'functions'));

try
    if isempty(gcp('nocreate'))
        parpool();
    end
catch
    % Parallel Computing Toolbox unavailable: run serially.
end

run_on_folder(folder_path, output_csv);


% -------------------------------------------------------------------------
% Main pipeline: load each dataset, tune, evaluate, and save results.
% -------------------------------------------------------------------------
function run_on_folder(folder_path, output_csv)

    if ~isfolder(folder_path)
        error('Dataset folder not found: %s', folder_path);
    end

    files = dir(fullfile(folder_path, '*.csv'));
    num_files = length(files);

    if num_files == 0
        fprintf('No CSV files found in: %s\n', folder_path);
        return;
    end

    fprintf('Found %d CSV file(s).\n\n', num_files);

    results = {};
    count = 0;

    for file_idx = 1:num_files

        file_name = files(file_idx).name;
        full_path = fullfile(folder_path, file_name);

        fprintf('[%d/%d] Dataset: %s\n', file_idx, num_files, file_name);

        try
            raw_tbl = readtable(full_path, detectImportOptions(full_path));

            % Features (categorical columns are label-encoded)
            X_data = zeros(height(raw_tbl), width(raw_tbl) - 1);
            for c = 1:width(raw_tbl) - 1
                col_val = raw_tbl{:, c};
                if iscell(col_val) || isstring(col_val) || iscategorical(col_val)
                    [~, ~, X_data(:, c)] = unique(col_val);
                else
                    X_data(:, c) = double(col_val);
                end
            end

            % Binary labels in {0, 1}
            y_raw = raw_tbl{:, end};
            if iscell(y_raw) || isstring(y_raw) || iscategorical(y_raw)
                [u_lbls, ~, y] = unique(y_raw);
                if length(u_lbls) ~= 2
                    fprintf('Skipping non-binary dataset: %s\n', file_name);
                    continue;
                end
                y = y - 1;
            else
                y = double(y_raw);
                y(y == -1) = 0;
                if length(unique(y)) ~= 2
                    fprintf('Skipping non-binary dataset: %s\n', file_name);
                    continue;
                end
            end

            best_params = perform_grid_search_5fold(X_data, y);
            avg_m = evaluate_5fold_average(X_data, y, best_params);

            count = count + 1;
            results(count, :) = { ...
                file_name, best_params.n_hidden, ...
                sprintf('2^{%.1f}', best_params.C1_pow), ...
                sprintf('2^{%.1f}', best_params.C2_pow), ...
                best_params.rho, best_params.theta, ...
                sprintf('2^{%.1f}', best_params.mew_pow), ...
                avg_m.gmean, avg_m.auc};

        catch ME
            fprintf('Error in %s: %s\n', file_name, ME.message);
            continue;
        end
    end

    if count == 0
        fprintf('No valid binary datasets were processed.\n');
        return;
    end

    headers = {'Dataset', 'Best_h', 'Best_C1_2p', 'Best_C2_2p', 'Best_rho', ...
               'Best_theta', 'Best_mew_2p', 'Avg_Gmean', 'Avg_AUC'};

    res_table = cell2table(results, 'VariableNames', headers);
    writetable(res_table, output_csv);

    fprintf('\nResults saved to: %s\n', output_csv);
    disp(res_table);

end


% -------------------------------------------------------------------------
% Grid search (5-fold cross-validation, selection by G-mean).
% -------------------------------------------------------------------------
function best_params = perform_grid_search_5fold(X, y)

    h_range     = [30, 60, 100, 150];
    c_range     = 2.0 .^ (-10:10:30);
    rho_range   = [0.001, 0.1, 1];
    theta_range = [0.001, 0.1, 1];
    mew_range   = 2.0 .^ (-10:10:10);

    cv_folds = cvpartition(y, 'KFold', 5);

    combo_list = [];
    for h = h_range
        for c1 = c_range
            for c2 = c_range
                for rho = rho_range
                    for theta = theta_range
                        for mew = mew_range
                            combo_list = [combo_list; h, c1, c2, rho, theta, mew]; %#ok<AGROW>
                        end
                    end
                end
            end
        end
    end

    num_combos = size(combo_list, 1);
    gmean_results = zeros(num_combos, 1);
    mid = floor(size(X, 2) / 2);

    fprintf('Grid search: %d combinations\n', num_combos);

    parfor idx = 1:num_combos

        p = struct();
        p.n_hidden = combo_list(idx, 1);
        p.C1       = combo_list(idx, 2);
        p.C2       = combo_list(idx, 3);
        p.rho      = combo_list(idx, 4);
        p.theta    = combo_list(idx, 5);
        p.mew      = combo_list(idx, 6);

        fold_gmeans = zeros(5, 1);

        try
            for k = 1:5
                tr_idx = cv_folds.training(k);
                te_idx = cv_folds.test(k);

                X_tr = X(tr_idx, :);  y_tr = y(tr_idx);
                X_te = X(te_idx, :);  y_te = y(te_idx);

                A_tr = [X_tr(:, 1:mid), y_tr];
                B_tr = [X_tr(:, mid+1:end), y_tr];
                A_te = [X_te(:, 1:mid), y_te];
                B_te = [X_te(:, mid+1:end), y_te];

                eval_m = ifelm_multiview_train( ...
                    A_tr, B_tr, A_te, B_te, p);
                fold_gmeans(k) = eval_m.gmean;
            end
            gmean_results(idx) = mean(fold_gmeans);
        catch
            gmean_results(idx) = -1.0;
        end

    end

    [best_avg_gmean, max_i] = max(gmean_results);

    if best_avg_gmean < 0
        error('Grid search failed for all hyperparameter combinations.');
    end

    best_params.n_hidden = combo_list(max_i, 1);
    best_params.C1       = combo_list(max_i, 2);
    best_params.C2       = combo_list(max_i, 3);
    best_params.rho      = combo_list(max_i, 4);
    best_params.theta    = combo_list(max_i, 5);
    best_params.mew      = combo_list(max_i, 6);
    best_params.C1_pow   = log2(best_params.C1);
    best_params.C2_pow   = log2(best_params.C2);
    best_params.mew_pow  = log2(best_params.mew);

    fprintf(['Best parameters: h=%d, C1=2^{%.1f}, C2=2^{%.1f}, rho=%.3f, ' ...
             'theta=%.3f, mew=2^{%.1f} | G-mean=%.2f%%\n'], ...
        best_params.n_hidden, best_params.C1_pow, best_params.C2_pow, ...
        best_params.rho, best_params.theta, best_params.mew_pow, best_avg_gmean);

end


% -------------------------------------------------------------------------
% 5-fold cross-validated evaluation (G-mean and AUC) with the selected
% hyperparameters.
% -------------------------------------------------------------------------
function avg_results = evaluate_5fold_average(X, y, best_params)

    cv_folds = cvpartition(y, 'KFold', 5);
    mid = floor(size(X, 2) / 2);

    gm_vec = zeros(5, 1);
    auc_vec = zeros(5, 1);

    for k = 1:5
        tr_idx = cv_folds.training(k);
        te_idx = cv_folds.test(k);

        X_tr = X(tr_idx, :);  y_tr = y(tr_idx);
        X_te = X(te_idx, :);  y_te = y(te_idx);

        A_tr = [X_tr(:, 1:mid), y_tr];
        B_tr = [X_tr(:, mid+1:end), y_tr];
        A_te = [X_te(:, 1:mid), y_te];
        B_te = [X_te(:, mid+1:end), y_te];

        m = ifelm_multiview_train(A_tr, B_tr, A_te, B_te, best_params);

        gm_vec(k) = m.gmean;
        auc_vec(k) = m.auc;
    end

    avg_results.gmean = mean(gm_vec);
    avg_results.auc = mean(auc_vec);

end
