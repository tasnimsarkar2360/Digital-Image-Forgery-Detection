function result = infer_random_forest(image_path, threshold, checkpoint_path, show_figure, save_dir)

here = fileparts(mfilename('fullpath'));
test_dir = fullfile(here, 'test');

if nargin < 1 || isempty(image_path)
    image_path = prompt_for_test_image(test_dir);
else
    image_path = resolve_image_path(image_path, test_dir);
end

if nargin < 2 || isempty(threshold)
    threshold = 0.5;
end
if nargin < 3 || isempty(checkpoint_path)
    checkpoint_path = fullfile(here, 'checkpoint', 'random_forest_model.mat');
end
if nargin < 4 || isempty(show_figure)
    show_figure = true;
end
if nargin < 5
    save_dir = fullfile(here, 'output', 'figures');
end

if ~isfile(checkpoint_path)
    error('infer_random_forest:nocheckpoint', ...
        'No checkpoint found at %s - run train_random_forest first.', checkpoint_path);
end
bundle = load(checkpoint_path);

img_rgb = load_preprocess_image(image_path, bundle.params.max_dim);
feats = extract_feature_vector_rgb(img_rgb, bundle.params);

x = zeros(1, numel(bundle.feature_names));
for j = 1:numel(bundle.feature_names)
    x(j) = feats.(bundle.feature_names{j});
end
x_scaled = (x - bundle.mu) ./ bundle.sigma;

[~, scores] = predict(bundle.model, x_scaled);
prob_forged = double(scores(2));

if prob_forged >= threshold
    verdict = 'forged';
else
    verdict = 'authentic';
end

fprintf('%s\n', image_path);
fprintf('  model:        random_forest\n');
fprintf('  verdict:      %s\n', upper(verdict));
fprintf('  P(forged):    %.4f  (threshold %.2f)\n', prob_forged, threshold);

if show_figure || ~isempty(save_dir)
    fig_pipe = make_pipeline_figure(img_rgb, image_path, bundle.params, show_figure);
    if ~isempty(save_dir)
        save_report_figures(fig_pipe, image_path, save_dir);
    end
    if ~show_figure
        close(fig_pipe);
    end
end

[heat, mask, ela] = localize(img_rgb, bundle.params);
fprintf('  flagged px:   %d (%.2f%% of image)\n', nnz(mask), 100 * nnz(mask) / numel(mask));

if show_figure || ~isempty(save_dir)
    fig_loc = save_localization_figure(img_rgb, ela, heat, mask, image_path, verdict, save_dir, show_figure);
    if ~show_figure
        close(fig_loc);
    end
end

result = struct('verdict', verdict, 'prob_forged', prob_forged, ...
    'features', feats, 'image', img_rgb, 'image_path', image_path, ...
    'heatmap', heat, 'mask', mask);

end


%%  LOCAL FUNCTIONS (image selection, display, preprocessing, RGB DSP feature extraction)


function image_path = prompt_for_test_image(test_dir)
if ~isfolder(test_dir)
    error('infer_random_forest:notestdir', 'Test folder not found: %s', test_dir);
end
exts = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(test_dir, exts{i}))]; 
end
if isempty(files)
    error('infer_random_forest:notestimages', 'No images found in %s', test_dir);
end
names = sort({files.name});
fprintf('Images available in %s:\n', test_dir);
for i = 1:numel(names)
    fprintf('  %2d) %s\n', i, names{i});
end
fprintf('   0) Browse for another file...\n');
resp = strtrim(input(sprintf( ...
    'Select an image [1-%d], type a filename, or 0 to browse: ', numel(names)), 's'));
if isempty(resp)
    error('infer_random_forest:badselection', 'No image selected.');
end

num = str2double(resp);
if ~isnan(num) && num == 0
    [f, p] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Image files'}, ...
        'Select an image', test_dir);
    if isequal(f, 0)
        error('infer_random_forest:badselection', 'No image selected.');
    end
    image_path = fullfile(p, f);
elseif ~isnan(num) && num == round(num) && num >= 1 && num <= numel(names)
    image_path = fullfile(test_dir, names{num});
else
    image_path = resolve_image_path(resp, test_dir);
end
end

function image_path = resolve_image_path(image_path, test_dir)

if isfile(image_path)
    return
end
candidate = fullfile(test_dir, image_path);
if isfile(candidate)
    image_path = candidate;
else
    error('infer_random_forest:nofile', ...
        'Image not found: %s (also checked %s)', image_path, candidate);
end
end

function fig = make_pipeline_figure(img_rgb, image_path, params, show_figure)

dct_maps = cell(1, 3); fft_maps = cell(1, 3); res_maps = cell(1, 3); hp_maps = cell(1, 3);
for c = 1:3
    dct_maps{c} = block_dct_ac_energy_map(img_rgb(:, :, c), params.block_size);
    fft_maps{c} = fft_log_magnitude(img_rgb(:, :, c));
    res_maps{c} = noise_residual_map(img_rgb(:, :, c), params.wavelet_name, params.wavelet_level);
    hp_maps{c} = abs(highpass_map(img_rgb(:, :, c)));
end
stages = {img_rgb, 'Original (resized, RGB)'; ...
          ela_map(img_rgb, params.ela_quality), 'ELA residual'; ...
          rgb_composite(dct_maps{:}), 'Block-DCT AC energy (RGB)'; ...
          rgb_composite(fft_maps{:}), 'log |FFT| (RGB)'; ...
          rgb_composite(res_maps{:}), 'Wavelet noise residual (RGB)'; ...
          rgb_composite(hp_maps{:}), 'Laplacian high-pass |.| (RGB)'};

fig = figure('Name', 'DSP preprocessing stages', 'NumberTitle', 'off', ...
    'Visible', on_off(show_figure), 'Position', [100 100 1300 800]);
tiledlayout(2, 3);
for i = 1:size(stages, 1)
    nexttile;
    data = stages{i, 1};
    if ndims(data) == 3
        imshow(data);
    else
        imshow(data, []); colormap(gca, 'parula');
    end
    title(stages{i, 2}, 'FontSize', 10);
    axis off
end
[~, base, ~] = fileparts(image_path);
sgtitle(['DSP preprocessing stages: ' base], 'Interpreter', 'none');
end

function rgb_img = rgb_composite(map_r, map_g, map_b)

rgb_img = cat(3, normalize01(map_r), normalize01(map_g), normalize01(map_b));
end

function out = normalize01(a)
a = double(a);
lo = min(a(:)); hi = max(a(:));
if hi - lo < 1e-12
    out = zeros(size(a));
else
    out = (a - lo) / (hi - lo);
end
end
function save_report_figures(fig_pipe, image_path, save_dir)

if ~isfolder(save_dir)
    mkdir(save_dir);
end
[~, base, ~] = fileparts(image_path);

save_png(fig_pipe, fullfile(save_dir, sprintf('preprocessing_pipeline_%s.png', base)));
end

function save_png(fig, out_path)
exportgraphics(fig, out_path, 'Resolution', 150);
fprintf('  figure saved: %s\n', out_path);
end

function s = on_off(flag)
if flag
    s = 'on';
else
    s = 'off';
end
end

function out = merge_struct(a, b)
out = a;
fn = fieldnames(b);
for i = 1:numel(fn)
    out.(fn{i}) = b.(fn{i});
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


%%  LOCAL FUNCTIONS (DSP forgery localization)


function [heat, mask, ela] = localize(img_rgb, params)

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

function rgb = heatmap_to_rgb(heat)

heat_u8 = uint8(round(min(max(heat, 0), 1) * 255));
rgb = uint8(round(ind2rgb(heat_u8, hot(256)) * 255));
end

function img_out = overlay_mask(img_rgb, mask, alpha)

if nargin < 3
    alpha = 0.45;
end
base = double(img_rgb);
img_out = base;
m = logical(mask);
for c = 1:3
    ch = base(:, :, c);
    red = 255 * (c == 1);
    ch(m) = (1 - alpha) * ch(m) + alpha * red;
    img_out(:, :, c) = ch;
end
img_out = uint8(img_out);
end

function fig = save_localization_figure(img_rgb, ela, heat, mask, image_path, verdict, save_dir, show_figure)

if strcmp(verdict, 'forged') && any(mask(:))
    region_img = overlay_mask(img_rgb, mask);
    region_title = 'Localized region (DSP, Otsu)';
else
    region_img = img_rgb;
    region_title = 'No region flagged (authentic verdict)';
end

fig = figure('Name', 'DSP forgery localization', 'NumberTitle', 'off', ...
    'Visible', on_off(show_figure), 'Position', [100 100 1500 420]);
tiledlayout(1, 4, 'TileSpacing', 'compact');
nexttile; imshow(img_rgb); title('Original');
nexttile; imshow(ela, []); colormap(gca, 'hot'); title('ELA map');
nexttile; imshow(heatmap_to_rgb(heat)); title('DSP anomaly heatmap');
nexttile; imshow(region_img); title(region_title);

[~, base, ~] = fileparts(image_path);
sgtitle(['DSP localization: ' base ' (' verdict ')'], 'Interpreter', 'none');

if ~isempty(save_dir)
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end
    save_png(fig, fullfile(save_dir, sprintf('localization_%s.png', base)));
end
end
