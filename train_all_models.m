
function results = train_all_models(dataset_dir, checkpoint_dir, mask_dir)

if nargin < 1 || isempty(dataset_dir)
    dataset_dir = pwd;
end
if nargin < 2 || isempty(checkpoint_dir)
    checkpoint_dir = fullfile(pwd, 'checkpoint');
end
if ~exist(checkpoint_dir, 'dir')
    mkdir(checkpoint_dir);
end

params = get_default_params();
VAL_SPLIT = 0.2;
SEED = 42;
N_TREES = 300;
IMAGE_EXTENSIONS = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};
MODEL_TYPES = {'logistic_regression', 'svm_rbf', 'random_forest'};
CKPT_NAMES = struct('logistic_regression', 'logistic_regression_model.mat', ...
                    'svm_rbf', 'svm_model.mat', ...
                    'random_forest', 'random_forest_model.mat');


%% Locate images

authentic_dir = fullfile(dataset_dir, 'authentic');
forged_dir = fullfile(dataset_dir, 'forged');
if ~exist(authentic_dir, 'dir') || ~exist(forged_dir, 'dir')
    error('train_all_models:missingdir', ...
        'Expected "%s" to contain "authentic" and "forged" subfolders.', dataset_dir);
end

authentic_files = list_images(authentic_dir, IMAGE_EXTENSIONS);
forged_files = list_images(forged_dir, IMAGE_EXTENSIONS);
fprintf('[data] %d authentic images, %d forged images\n', ...
    numel(authentic_files), numel(forged_files));

all_paths = [authentic_files; forged_files];
all_labels = [zeros(numel(authentic_files), 1); ones(numel(forged_files), 1)];
if isempty(all_paths)
    error('train_all_models:nodata', 'No images found under %s.', dataset_dir);
end


%% Feature extraction (once, shared by all models)

n = numel(all_paths);
rows = cell(n, 1);
keep = false(n, 1);
feature_names = {};

for i = 1:n
    try
        img_rgb = load_preprocess_image(all_paths{i}, params.max_dim);
    catch err
        fprintf('[warn] skipping unreadable image %s: %s\n', all_paths{i}, err.message);
        continue
    end
    feats = extract_feature_vector_rgb(img_rgb, params);
    if isempty(feature_names)
        feature_names = fieldnames(feats);
    end
    rows{i} = feats;
    keep(i) = true;
    if mod(i, 20) == 0 || i == n
        fprintf('  extracting DSP features: %d/%d\n', i, n);
    end
end

kept_paths = all_paths(keep);
labels = all_labels(keep);
rows = rows(keep);

X = zeros(numel(rows), numel(feature_names));
for i = 1:numel(rows)
    for j = 1:numel(feature_names)
        X(i, j) = rows{i}.(feature_names{j});
    end
end
y = labels;
fprintf('[features] matrix [%d x %d]\n', size(X, 1), size(X, 2));


%%  Train / validation split + standardization (shared)

rng(SEED);
cv = cvpartition(y, 'HoldOut', VAL_SPLIT, 'Stratify', true);
X_train = X(training(cv), :); y_train = y(training(cv));
X_val = X(test(cv), :); y_val = y(test(cv));

mu = mean(X_train, 1);
sigma = std(X_train, 0, 1);
sigma(sigma == 0) = 1;
X_train_s = (X_train - mu) ./ sigma;
X_val_s = (X_val - mu) ./ sigma;

w_train = balanced_weights(y_train);

figures_dir = fullfile(pwd, 'output', 'figures');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Cross-validated model comparison (shared)

cv_scores = struct();
for name = MODEL_TYPES
    [m, s] = cv_auc_for_model(name{1}, X_train_s, y_train, SEED);
    cv_scores.(name{1}) = struct('mean_auc', m, 'std_auc', s);
    fprintf('[cv] %s: AUC = %.3f +/- %.3f\n', name{1}, m, s);
end


%%  DSP localization evaluation (IoU / Dice vs. ground-truth masks)


if nargin < 3 || isempty(mask_dir)
    mask_dir = find_mask_dir(dataset_dir);
end
ious = []; dices = [];
if isempty(mask_dir)
    fprintf('[localization] no mask folder found - pass mask_dir as 3rd argument; skipping IoU/Dice\n');
else
    val_paths = kept_paths(test(cv));
    for k = find(y_val(:)' == 1)
        [~, base, ~] = fileparts(val_paths{k});
        mask_path = find_mask_file(mask_dir, base);
        if isempty(mask_path)
            continue
        end
        img_rgb = load_preprocess_image(val_paths{k}, params.max_dim);
        gt = load_mask(mask_path, size(img_rgb, 1:2));
        [~, pred_mask] = localize(img_rgb, params);
        [iou, dice] = iou_dice(pred_mask, gt);
        ious(end + 1, 1) = iou; 
        dices(end + 1, 1) = dice; 
    end
    if isempty(ious)
        fprintf('[localization] no ground-truth masks matched in %s; skipping IoU/Dice\n', mask_dir);
    end
end
if ~isempty(ious)
    fprintf('[localization] %d masked forged val images: mean IoU=%.3f  mean Dice=%.3f\n', ...
        numel(ious), mean(ious), mean(dices));
end

fprintf('[features] key forensic feature means, authentic images:\n%s\n', ...
    top_feature_lines(cell2struct(num2cell(mean(X(y == 0, :), 1)), feature_names, 2)));
fprintf('[features] key forensic feature means, forged images:\n%s\n', ...
    top_feature_lines(cell2struct(num2cell(mean(X(y == 1, :), 1)), feature_names, 2)));

%%  Train, evaluate, plot and save each model

results = struct();
for mi = 1:numel(MODEL_TYPES)
    model_label = MODEL_TYPES{mi};

    fprintf('\n=== %s ===\n', model_label);
    switch model_label
        case 'logistic_regression'
            fprintf('[train] fitting Logistic Regression on %d samples...\n', numel(y_train));
            model = fitglm(X_train_s, y_train, 'Distribution', 'binomial', 'Weights', w_train);
            y_prob = predict(model, X_val_s);
            y_pred = double(y_prob >= 0.5);
            
            importances = abs(model.Coefficients.Estimate(2:end));
            imp_label = 'Logistic Regression |standardized coefficient|';

        case 'svm_rbf'
            fprintf('[train] fitting RBF SVM on %d samples...\n', numel(y_train));
            svm_model = fitcsvm(X_train_s, y_train, 'KernelFunction', 'rbf', ...
                'ClassNames', [0 1], 'Weights', w_train);
            model = fitPosterior(svm_model);
            [y_pred, post] = predict(model, X_val_s);
            y_prob = post(:, 2);
            
            importances = permutation_importance(@(Xv) svm_scores(model, Xv), X_val_s, y_val, SEED);
            imp_label = 'SVM permutation importance (validation AUC drop)';

        case 'random_forest'
            fprintf('[train] fitting Random Forest (%d trees) on %d samples...\n', N_TREES, numel(y_train));
            model = TreeBagger(N_TREES, X_train_s, y_train, 'Method', 'classification', ...
                'Weights', w_train, 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');
            [labels_cell, scores] = predict(model, X_val_s);
            y_pred = str2double(labels_cell);
            y_prob = scores(:, 2);
            importances = model.OOBPermutedPredictorDeltaError;
            imp_label = 'Random Forest feature importance (OOB permuted delta error)';
            [~, imp_order] = sort(importances, 'descend');
            fprintf('[importance] top 5 features: %s\n', ...
                strjoin(feature_names(imp_order(1:min(5, end))), ', '));
    end

    % ---- validation metrics ----
    cm = confusionmat(y_val, y_pred, 'Order', [0 1]);
    tp = cm(2, 2); fp = cm(1, 2); fn_ = cm(2, 1); tn = cm(1, 1);
    acc = (tp + tn) / sum(cm(:));
    prec = tp / max(tp + fp, 1);
    rec = tp / max(tp + fn_, 1);
    f1 = 2 * prec * rec / max(prec + rec, eps);
    [fpr, tpr, ~, auc] = perfcurve(y_val, y_prob, 1);
    [recall_pr, precision_pr] = perfcurve(y_val, y_prob, 1, 'xCrit', 'reca', 'yCrit', 'prec');
    ap = average_precision(recall_pr, precision_pr);

    fprintf('[val] acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f ap=%.3f\n', acc, prec, rec, f1, auc, ap);

    % ---- report figures ----
    save_confusion_matrix(cm, {'authentic', 'forged'}, model_label, figures_dir);
    save_roc_curve(fpr, tpr, auc, model_label, figures_dir);
    save_pr_curve(precision_pr, recall_pr, ap, model_label, figures_dir);
    save_cv_comparison(cv_scores, model_label, figures_dir);
    save_feature_importance(importances, feature_names, imp_label, model_label, figures_dir);
    if ~isempty(ious)
        save_iou_dice_histogram(ious, dices, model_label, figures_dir);
    end
    fprintf('[figures] report figures -> %s\n', figures_dir);

    metrics = struct('model_type', model_label, ...
        'n_train', numel(y_train), 'n_val', numel(y_val), ...
        'accuracy', acc, 'precision', prec, 'recall', rec, 'f1', f1, 'auc', auc, ...
        'avg_precision', ap, 'confusion_matrix', cm, 'cv_scores', cv_scores, ...
        'feature_importance', importances, ...
        'n_localization_samples', numel(ious), 'mean_iou', mean(ious), 'mean_dice', mean(dices));

    % ---- save checkpoint ----
    bundle = struct('model', {model}, 'model_type', model_label, ...
        'mu', mu, 'sigma', sigma, 'feature_names', {feature_names}, ...
        'params', params, 'metrics', metrics);
    ckpt_path = fullfile(checkpoint_dir, CKPT_NAMES.(model_label));
    save(ckpt_path, '-struct', 'bundle');
    fprintf('[done] checkpoint -> %s\n', ckpt_path);

    results.(model_label) = metrics;
end


%%  Summary

fprintf('\n[summary] validation metrics (%d train / %d val)\n', numel(y_train), numel(y_val));
fprintf('  %-20s %6s %6s %6s %6s %6s %6s\n', 'model', 'acc', 'prec', 'rec', 'f1', 'auc', 'ap');
for mi = 1:numel(MODEL_TYPES)
    r = results.(MODEL_TYPES{mi});
    fprintf('  %-20s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n', MODEL_TYPES{mi}, ...
        r.accuracy, r.precision, r.recall, r.f1, r.auc, r.avg_precision);
end

end


%%  LOCAL FUNCTIONS (preprocessing + RGB DSP feature extraction)


function params = get_default_params()
params.max_dim = 1024;
params.block_size = 8;
params.ela_quality = 90;
params.wavelet_name = 'db8';
params.wavelet_level = 1;
end

function files = list_images(folder, extensions)
files = {};
if ~exist(folder, 'dir')
    return
end
for i = 1:numel(extensions)
    listing = dir(fullfile(folder, extensions{i}));
    for k = 1:numel(listing)
        files{end + 1} = fullfile(listing(k).folder, listing(k).name); 
    end
end
files = sort(files(:));
end

function out = merge_struct(a, b)
out = a;
fn = fieldnames(b);
for i = 1:numel(fn)
    out.(fn{i}) = b.(fn{i});
end
end

function w = balanced_weights(y)
classes = unique(y);
n = numel(y);
w = zeros(size(y));
for i = 1:numel(classes)
    cnt = sum(y == classes(i));
    w(y == classes(i)) = n / (numel(classes) * max(cnt, 1));
end
end

function img_rgb = load_preprocess_image(path, max_dim)
img = imread(path);
if size(img, 3) == 1
    img = repmat(img, [1 1 3]); 
elseif size(img, 3) == 4
    img = img(:, :, 1:3);
end
img_rgb = im2uint8(img);
[h, w, ~] = size(img_rgb);
scale = min(1.0, max_dim / max(h, w));
if scale < 1.0
    img_rgb = imresize(img_rgb, [round(h * scale), round(w * scale)], 'method', 'box');
end
end

function [feats, names] = extract_feature_vector_rgb(img_rgb, params)
feats = struct();
feats = merge_struct(feats, compute_ela_features(img_rgb, params.ela_quality));
feats = merge_struct(feats, compute_dct_features_rgb(img_rgb, params.block_size));
feats = merge_struct(feats, compute_fft_features_rgb(img_rgb));
feats = merge_struct(feats, compute_noise_features_rgb(img_rgb, params.wavelet_name, params.wavelet_level));
feats = merge_struct(feats, compute_highpass_features_rgb(img_rgb));
feats = merge_struct(feats, compute_cfa_features(img_rgb));
feats = merge_struct(feats, compute_blockiness_features_rgb(img_rgb, params.block_size));
names = fieldnames(feats);
end

% ----------------------- 1. Error Level Analysis ----------------------

function m = ela_map(img_rgb, quality)
tmp = [tempname() '.jpg'];
imwrite(img_rgb, tmp, 'jpg', 'Quality', quality);
recompressed = imread(tmp);
delete(tmp);
if size(recompressed, 3) == 1
    recompressed = repmat(recompressed, [1 1 3]);
end
diff = abs(double(img_rgb) - double(recompressed));
m = mean(diff, 3);
end

function feats = compute_ela_features(img_rgb, quality)
m = ela_map(img_rgb, quality);
lap = imfilter(m, fspecial('laplacian', 0), 'replicate', 'same');
feats = struct( ...
    'ela_mean', mean(m(:)), 'ela_std', std(m(:)), 'ela_max', max(m(:)), ...
    'ela_p95', prctile(m(:), 95), 'ela_energy', mean(m(:) .^ 2), ...
    'ela_highfreq_ratio', mean(abs(lap(:))) / (mean(m(:)) + 1e-6));
end

% ------------------- 2. Block-DCT / Benford's-law ----------------------

function p = benford_expected()
d = 1:9;
p = log10(1 + 1 ./ d);
end

function [fd_map, digits] = block_dct_ac_energy_map(channel, block)
[h, w] = size(channel);
h_c = floor(h / block) * block;
w_c = floor(w / block) * block;
g = double(channel(1:h_c, 1:w_c));
n_rows = h_c / block; n_cols = w_c / block;
energy = zeros(n_rows, n_cols, 'single');
digit_chunks = {};
for by = 1:block:h_c
    row = (by - 1) / block + 1;
    for bx = 1:block:w_c
        col = (bx - 1) / block + 1;
        d = dct2(g(by:by + block - 1, bx:bx + block - 1));
        d_flat = d(:);
        ac = abs(d_flat(2:end));
        energy(row, col) = sum(ac .^ 2);
        nz = ac(ac >= 1.0);
        if ~isempty(nz)
            fd = floor(nz ./ (10 .^ floor(log10(nz))));
            digit_chunks{end + 1} = fd; 
        end
    end
end
fd_map = imresize(energy, [h, w], 'nearest');
if isempty(digit_chunks); digits = 1; else; digits = vertcat(digit_chunks{:}); end
end

function feats = compute_dct_features_rgb(img_rgb, block)
digits_all = []; energy_all = [];
for c = 1:3
    [emap, dg] = block_dct_ac_energy_map(img_rgb(:, :, c), block);
    energy_all = [energy_all; double(emap(:))];
    digits_all = [digits_all; dg(:)]; 
end
counts = zeros(1, 9);
for d = 1:9
    counts(d) = sum(digits_all == d);
end
counts = counts / max(sum(counts), 1.0);
expected = benford_expected();
chi_sq = sum((counts - expected) .^ 2 ./ (expected + 1e-9));
nz = counts > 0;
kl_div = sum(counts(nz) .* log((counts(nz) + 1e-9) ./ expected(nz)));
feats = struct( ...
    'dct_benford_chi2', chi_sq, 'dct_benford_kl', kl_div, ...
    'dct_energy_mean', mean(energy_all), 'dct_energy_std', std(energy_all), ...
    'dct_energy_cv', std(energy_all) / (mean(energy_all) + 1e-6));
end

% ------------------- 3. FFT / global spectral features -----------------

function mag = fft_log_magnitude(channel)
f = fftshift(fft2(double(channel)));
mag = log1p(abs(f));
end

function feats = compute_fft_features_rgb(img_rgb)
low_all = []; mid_all = []; high_all = []; mag_pool = []; power_pool = [];
for c = 1:3
    channel = img_rgb(:, :, c);
    mag = fft_log_magnitude(channel);
    [h, w] = size(mag);
    cy = floor(h / 2) + 1; cx = floor(w / 2) + 1;
    [xx, yy] = meshgrid(1:w, 1:h);
    r = sqrt((yy - cy) .^ 2 + (xx - cx) .^ 2);
    r_norm = r / max(r(:));
    low_all = [low_all; mag(r_norm < 0.15)]; 
    mid_all = [mid_all; mag(r_norm >= 0.15 & r_norm < 0.5)]; 
    high_all = [high_all; mag(r_norm >= 0.5)]; 
    mag_pool = [mag_pool; mag(:)]; 
    power = abs(fft2(double(channel))) .^ 2;
    pf = power(:); pf = pf(pf > 0);
    power_pool = [power_pool; pf]; 
end
log_p = log(power_pool);
flatness = exp(mean(log_p)) / (mean(power_pool) + 1e-9);
high_thr = prctile(mag_pool, 99.5);
peak_ratio = mean(high_all > high_thr);
feats = struct( ...
    'fft_low_mean', mean(low_all), 'fft_mid_mean', mean(mid_all), ...
    'fft_high_mean', mean(high_all), ...
    'fft_high_low_ratio', mean(high_all) / (mean(low_all) + 1e-6), ...
    'fft_spectral_flatness', flatness, 'fft_highfreq_peak_ratio', peak_ratio);
end

% ----------------- 4. Wavelet noise-residual features -------------------

function residual = noise_residual_map(channel, wavelet, level)
g = double(channel);
[C, S] = wavedec2(g, level, wavelet);
n_approx = S(1, 1) * S(1, 2);
C_zeroed = C; C_zeroed(1:n_approx) = 0;
rec = waverec2(C_zeroed, S, wavelet);
residual = rec(1:size(g, 1), 1:size(g, 2));
end

function var_map = local_variance_map(arr, block)
arr = double(arr);
[h, w] = size(arr);
h_c = floor(h / block) * block; w_c = floor(w / block) * block;
a = arr(1:h_c, 1:w_c);
n_rows = h_c / block; n_cols = w_c / block;
var_small = zeros(n_rows, n_cols, 'single');
for by = 1:block:h_c
    row = (by - 1) / block + 1;
    for bx = 1:block:w_c
        col = (bx - 1) / block + 1;
        patch = a(by:by + block - 1, bx:bx + block - 1);
        var_small(row, col) = var(patch(:), 1);
    end
end
var_map = imresize(var_small, [h, w], 'nearest');
end

function feats = compute_noise_features_rgb(img_rgb, wavelet, level)
r_all = []; vm_all = [];
for c = 1:3
    residual = noise_residual_map(img_rgb(:, :, c), wavelet, level);
    var_map = local_variance_map(residual, 16);
    r_all = [r_all; double(residual(:))]; 
    vm_all = [vm_all; double(var_map(:))];
end
feats = struct( ...
    'noise_residual_std', std(r_all), 'noise_residual_energy', mean(r_all .^ 2), ...
    'noise_var_map_mean', mean(vm_all), 'noise_var_map_std', std(vm_all), ...
    'noise_var_map_cv', std(vm_all) / (mean(vm_all) + 1e-6));
end

% ---------------- 5. High-pass / Laplacian edge consistency -------------

function hp = highpass_map(channel)
k = [0 1 0; 1 -4 1; 0 1 0];
hp = imfilter(double(channel), k, 'replicate', 'same');
end

function feats = compute_highpass_features_rgb(img_rgb)
hp_all = []; vm_all = [];
for c = 1:3
    hp = highpass_map(img_rgb(:, :, c));
    var_map = local_variance_map(hp, 16);
    hp_all = [hp_all; hp(:)]; 
    vm_all = [vm_all; double(var_map(:))];
end
feats = struct( ...
    'hp_energy', mean(hp_all .^ 2), 'hp_mean_abs', mean(abs(hp_all)), ...
    'hp_var_map_cv', std(vm_all) / (mean(vm_all) + 1e-6));
end

% --------------------- 6. CFA / demosaicing artifacts --------------------

function feats = compute_cfa_features(img_rgb)
g = double(img_rgb(:, :, 2));
pred = zeros(size(g));
pred(2:end-1, 2:end-1) = (g(1:end-2, 2:end-1) + g(3:end, 2:end-1) + ...
                          g(2:end-1, 1:end-2) + g(2:end-1, 3:end)) / 4.0;
residual = abs(g - pred);
residual = residual(2:end-1, 2:end-1);
even_rows = residual(1:2:end, :); odd_rows = residual(2:2:end, :);
grid_energy_diff = abs(mean(even_rows(:)) - mean(odd_rows(:)));
feats = struct( ...
    'cfa_residual_mean', mean(residual(:)), 'cfa_residual_std', std(residual(:)), ...
    'cfa_grid_energy_diff', grid_energy_diff);
end

% -------------------------- 7. JPEG blockiness ----------------------------

function feats = compute_blockiness_features_rgb(img_rgb, block)
boundary_all = []; interior_all = [];
for c = 1:3
    g = double(img_rgb(:, :, c));
    gx = abs(diff(g, 1, 2));
    w1 = size(gx, 2);
    cols = 1:w1;
    on_grid = mod(cols, block) == 0;
    if any(on_grid)
        boundary_all = [boundary_all; reshape(gx(:, on_grid), [], 1)];
    end
    if any(~on_grid)
        interior_all = [interior_all; reshape(gx(:, ~on_grid), [], 1)]; 
    end
end
if isempty(boundary_all); boundary_energy = 0.0; else; boundary_energy = mean(boundary_all); end
if isempty(interior_all); interior_energy = 1e-6; else; interior_energy = mean(interior_all); end
feats = struct('blockiness_ratio', boundary_energy / (interior_energy + 1e-6));
end


%% LOCAL FUNCTIONS (report figures + model comparison)


function path = save_confusion_matrix(cm, class_names, model_label, out_dir)
fig = figure('Visible', 'off', 'Position', [100 100 400 400]);
imagesc(cm); colormap(fig, 'summer'); colorbar;
set(gca, 'XTick', 1:numel(class_names), 'XTickLabel', class_names, ...
         'YTick', 1:numel(class_names), 'YTickLabel', class_names);
xlabel('Predicted'); ylabel('True');
title(sprintf('Confusion matrix (validation split): %s', model_label), 'Interpreter', 'none');
for i = 1:size(cm, 1)
    for j = 1:size(cm, 2)
        text(j, i, num2str(cm(i, j)), 'HorizontalAlignment', 'center', 'Color', 'k');
    end
end
path = fullfile(out_dir, sprintf('confusion_matrix_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function path = save_roc_curve(fpr, tpr, auc, model_label, out_dir)
fig = figure('Visible', 'off', 'Position', [100 100 450 450]);
plot(fpr, tpr, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (AUC=%.3f)', strrep(model_label, '_', ' '), auc));
hold on
plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5], 'DisplayName', 'chance');
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC curve (validation split)'); legend('Location', 'southeast');
path = fullfile(out_dir, sprintf('roc_curve_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function path = save_pr_curve(precision, recall, ap, model_label, out_dir)
fig = figure('Visible', 'off', 'Position', [100 100 450 450]);
plot(recall, precision, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (AP=%.3f)', strrep(model_label, '_', ' '), ap));
xlabel('Recall'); ylabel('Precision');
title('Precision-Recall curve (validation split)'); legend('Location', 'southwest');
path = fullfile(out_dir, sprintf('pr_curve_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function path = save_cv_comparison(cv_scores, model_label, out_dir)

names = fieldnames(cv_scores);
means = cellfun(@(n) cv_scores.(n).mean_auc, names);
stds = cellfun(@(n) cv_scores.(n).std_auc, names);

fig = figure('Visible', 'off', 'Position', [100 100 550 400]);
bar(means, 'FaceColor', [0.51 0.45 0.70]);
hold on
errorbar(1:numel(means), means, stds, 'k.', 'LineWidth', 1);
set(gca, 'XTick', 1:numel(names), 'XTickLabel', strrep(names, '_', ' '));
xtickangle(15);
ylabel('5-fold cross-validated ROC-AUC'); title('Model selection: candidate classifiers');
ylim([0 1]);
path = fullfile(out_dir, sprintf('cv_comparison_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function path = save_feature_importance(importances, feature_names, x_label, model_label, out_dir, top_k)

if nargin < 6 || isempty(top_k)
    top_k = 15;
end
[sorted_imp, order] = sort(importances(:), 'descend');
n = min(top_k, numel(order));
order = order(1:n);
sorted_imp = sorted_imp(1:n);

fig = figure('Visible', 'off', 'Position', [100 100 700 600]);
barh(flipud(sorted_imp(:)));
set(gca, 'YTick', 1:n, 'YTickLabel', flipud(feature_names(order)));
set(gca, 'TickLabelInterpreter', 'none');
xlabel(x_label, 'Interpreter', 'none'); title('Top DSP features by importance');
path = fullfile(out_dir, sprintf('feature_importance_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function txt = top_feature_lines(feats)
keys = {'ela_max', 'ela_energy', 'noise_var_map_cv', 'hp_var_map_cv', 'dct_benford_kl'};
lines = {};
for i = 1:numel(keys)
    k = keys{i};
    if isfield(feats, k)
        lines{end + 1} = sprintf('  - %s: %.4f', k, feats.(k)); 
    end
end
txt = strjoin(lines, newline);
end

function ap = average_precision(recall, precision)
ok = ~isnan(recall) & ~isnan(precision);
[r, order] = sort(recall(ok));
p = precision(ok);
p = p(order);
ap = trapz(r, p);
end

function [mean_auc, std_auc] = cv_auc_for_model(name, X, y, seed)

rng(seed);
c = cvpartition(y, 'KFold', max(2, min(5, min(sum(y == 0), sum(y == 1)))), 'Stratify', true);
aucs = zeros(c.NumTestSets, 1);
for k = 1:c.NumTestSets
    tri = training(c, k); tei = test(c, k);
    model = fit_cv_model(name, X(tri, :), y(tri), balanced_weights(y(tri)));
    y_prob = predict_cv_scores(name, model, X(tei, :));
    [~, ~, ~, aucs(k)] = perfcurve(y(tei), y_prob, 1);
end
mean_auc = mean(aucs);
std_auc = std(aucs);
end

function model = fit_cv_model(name, X, y, w)
switch name
    case 'logistic_regression'
        model = fitglm(X, y, 'Distribution', 'binomial', 'Weights', w);
    case 'svm_rbf'
        m = fitcsvm(X, y, 'KernelFunction', 'rbf', 'ClassNames', [0 1], 'Weights', w);
        model = fitPosterior(m);
    case 'random_forest'
        model = TreeBagger(300, X, y, 'Method', 'classification', 'Weights', w);
    otherwise
        error('cv_auc_for_model:model', 'Unknown model type: %s', name);
end
end

function y_prob = predict_cv_scores(name, model, X)
switch name
    case 'logistic_regression'
        y_prob = predict(model, X);
    case 'svm_rbf'
        [~, post] = predict(model, X);
        y_prob = post(:, 2);
    case 'random_forest'
        [~, scores] = predict(model, X);
        y_prob = scores(:, 2);
    otherwise
        error('cv_auc_for_model:model', 'Unknown model type: %s', name);
end
end

function s = svm_scores(model, X)
[~, post] = predict(model, X);
s = post(:, 2);
end

function imp = permutation_importance(score_fn, X, y, seed)
rng(seed);
[~, ~, ~, base_auc] = perfcurve(y, score_fn(X), 1);
imp = zeros(1, size(X, 2));
for j = 1:size(X, 2)
    drops = zeros(1, 5);
    for r = 1:5
        Xp = X;
        Xp(:, j) = X(randperm(size(X, 1)), j);
        [~, ~, ~, auc_p] = perfcurve(y, score_fn(Xp), 1);
        drops(r) = base_auc - auc_p;
    end
    imp(j) = mean(drops);
end
end

%%  LOCAL FUNCTIONS (DSP localization + IoU/Dice evaluation)


function [heat, mask] = localize(img_rgb, params)

ela = ela_map(img_rgb, params.ela_quality);
ela_smooth = imgaussfilt(ela, 1.1, 'FilterSize', 5);

noise_var = 0;
dct_energy = 0;
for c = 1:3
    channel = img_rgb(:, :, c);
    residual = noise_residual_map(channel, params.wavelet_name, params.wavelet_level);
    noise_var = noise_var + double(local_variance_map(residual, 16)) / 3;
    dct_energy = dct_energy + double(block_dct_ac_energy_map(channel, params.block_size)) / 3;
end

combined = local_zscore(ela_smooth) + local_zscore(noise_var) + local_zscore(dct_energy);
combined = imgaussfilt(combined, 1.4, 'FilterSize', 7);

lo = prctile(combined(:), 1);
hi = prctile(combined(:), 99);
heat = single(min(max((combined - lo) / (hi - lo + 1e-6), 0), 1));

mask = heatmap_to_mask(heat, 80);
end

function z = local_zscore(map_)
z = (map_ - mean(map_(:))) / (std(map_(:)) + 1e-6);
end

function mask = heatmap_to_mask(heat, min_area)
heat_u8 = uint8(round(heat * 255));
level = graythresh(heat_u8);
mask = imbinarize(heat_u8, level);
mask = imclose(mask, ones(5, 5));
mask = imopen(mask, ones(3, 3));
mask = bwareaopen(mask, min_area, 8);
mask = uint8(mask);
end

function [iou, dice] = iou_dice(pred_mask, gt_mask)

pred = logical(pred_mask);
gt = logical(gt_mask);

inter = sum(pred(:) & gt(:));
union_ = sum(pred(:) | gt(:));
if union_ > 0
    iou = inter / union_;
else
    iou = double(sum(pred(:)) == 0);
end

denom = sum(pred(:)) + sum(gt(:));
if denom > 0
    dice = 2 * inter / denom;
else
    dice = double(sum(pred(:)) == 0);
end
end

function path = save_iou_dice_histogram(ious, dices, model_label, out_dir)

fig = figure('Visible', 'off', 'Position', [100 100 900 400]);
tiledlayout(1, 2);
nexttile;
histogram(ious, 20, 'FaceColor', [0.30 0.45 0.69]);
xline(mean(ious), '--k', sprintf('mean=%.3f', mean(ious)));
title('IoU distribution');
nexttile;
histogram(dices, 20, 'FaceColor', [0.77 0.31 0.32]);
xline(mean(dices), '--k', sprintf('mean=%.3f', mean(dices)));
title('Dice distribution');
sgtitle('DSP-only localization performance across forged validation images');
path = fullfile(out_dir, sprintf('iou_dice_histogram_%s.png', model_label));
exportgraphics(fig, path, 'Resolution', 150);
close(fig);
end

function mask_dir = find_mask_dir(dataset_dir)

candidates = {fullfile(dataset_dir, 'masks'), ...
              fullfile(dataset_dir, 'train_masks'), ...
              fullfile(dataset_dir, '..', 'train_masks')};
mask_dir = '';
for i = 1:numel(candidates)
    if isfolder(candidates{i})
        mask_dir = candidates{i};
        return
    end
end
end

function mask_path = find_mask_file(mask_dir, base)
mask_path = '';
for ext = {'.npy', '.png'}
    candidate = fullfile(mask_dir, [base ext{1}]);
    if isfile(candidate)
        mask_path = candidate;
        return
    end
end
end

function mask = load_mask(mask_path, target_hw)
[~, ~, ext] = fileparts(mask_path);
if strcmpi(ext, '.npy')
    raw = read_npy(mask_path);
    if ndims(raw) == 3
        m = squeeze(any(raw > 0, 1));
    else
        m = raw > 0;
    end
else
    m = imread(mask_path);
    if size(m, 3) > 1
        m = any(m > 0, 3);
    else
        m = m > 0;
    end
end
mask = uint8(m);
if ~isequal(size(mask), target_hw)
    mask = uint8(imresize(mask, target_hw, 'nearest'));
end
end

function data = read_npy(filename)

fid = fopen(filename, 'rb');
if fid < 0
    error('read_npy:open', 'Could not open file: %s', filename);
end
cleaner = onCleanup(@() fclose(fid)); 

magic = fread(fid, 6, 'uint8=>char')';
if ~isequal(magic, char([147 78 85 77 80 89])) % \x93NUMPY
    error('read_npy:magic', 'Not a valid .npy file: %s', filename);
end

major = fread(fid, 1, 'uint8=>double');
fread(fid, 1, 'uint8=>double'); 
if major == 1
    header_len = fread(fid, 1, 'uint16=>double');
else
    header_len = fread(fid, 1, 'uint32=>double');
end
header = fread(fid, header_len, 'uint8=>char')';

descr = regexp(header, "'descr':\s*'([^']+)'", 'tokens', 'once');
descr = descr{1};
fortran = regexp(header, "'fortran_order':\s*(True|False)", 'tokens', 'once');
fortran_order = strcmp(fortran{1}, 'True');
shape_str = regexp(header, "'shape':\s*\(([^)]*)\)", 'tokens', 'once');
shape = str2double(strsplit(strtrim(shape_str{1}), ','));
shape = shape(~isnan(shape));
if isempty(shape)
    shape = 1;
end

raw = fread(fid, prod(shape), [npy_dtype_to_matlab(descr) '=>double']);
if numel(shape) <= 1
    data = raw;
    return
end
if fortran_order
    data = reshape(raw, shape);
else
    
    data = reshape(raw, fliplr(shape));
    data = permute(data, numel(shape):-1:1);
end
end

function matlab_type = npy_dtype_to_matlab(descr)
map = struct( ...
    'f4', 'single', 'f8', 'double', ...
    'i1', 'int8', 'i2', 'int16', 'i4', 'int32', 'i8', 'int64', ...
    'u1', 'uint8', 'u2', 'uint16', 'u4', 'uint32', 'u8', 'uint64', ...
    'b1', 'uint8');
key = descr(2:end); 
if strcmp(key, '?') || strcmp(descr(end), '?')
    matlab_type = 'uint8';
elseif isfield(map, key)
    matlab_type = map.(key);
else
    error('read_npy:dtype', 'Unsupported .npy dtype: %s', descr);
end
end
