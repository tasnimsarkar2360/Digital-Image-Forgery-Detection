
clear; clc;

%%  USER CONFIGURATION

AUTHENTIC_DIR_NAME = 'authentic';
FORGED_DIR_NAME    = 'forged';
OUTPUT_DIR_NAME    = 'output';

MAX_DIM          = 1024;   
BLOCK_SIZE       = 8;      
ELA_JPEG_QUALITY = 90;     
WAVELET_NAME     = 'db8'; 
WAVELET_LEVEL    = 1;
SAVE_STAGE_FIGURES = true; 

IMAGE_EXTENSIONS = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};


%%  RESOLVE FOLDERS

here = fileparts(mfilename('fullpath'));

authentic_dir = resolve_dir(AUTHENTIC_DIR_NAME, here);
forged_dir    = resolve_dir(FORGED_DIR_NAME, here);
output_dir    = fullfile(pwd, OUTPUT_DIR_NAME);
figures_dir   = fullfile(output_dir, 'figures');

if ~exist(output_dir, 'dir');  mkdir(output_dir);  end
if SAVE_STAGE_FIGURES && ~exist(figures_dir, 'dir'); mkdir(figures_dir); end

fprintf('[paths] authentic: %s\n', authentic_dir);
fprintf('[paths] forged:    %s\n', forged_dir);
fprintf('[paths] output:    %s\n', output_dir);

%%  BUILD FILE LIST

authentic_files = list_images(authentic_dir, IMAGE_EXTENSIONS);
forged_files    = list_images(forged_dir, IMAGE_EXTENSIONS);

fprintf('[data] %d authentic images, %d forged images\n', ...
    numel(authentic_files), numel(forged_files));

all_paths  = [authentic_files; forged_files];
all_labels = [zeros(numel(authentic_files), 1); ones(numel(forged_files), 1)];
all_class  = [repmat({'authentic'}, numel(authentic_files), 1); ...
              repmat({'forged'}, numel(forged_files), 1)];

if isempty(all_paths)
    error('extract_dataset_features:nodata', ...
        ['No images found. Put images in "%s" and "%s" folders ' ...
         '(next to this script, or in the current MATLAB folder).'], ...
        AUTHENTIC_DIR_NAME, FORGED_DIR_NAME);
end

%% FEATURE EXTRACTION LOOP

n = numel(all_paths);
feat_rows  = cell(n, 1);
keep       = false(n, 1);
feature_names = {};

for i = 1:n
    p = all_paths{i};
    try
        img_rgb = load_preprocess_image(p, MAX_DIM); % always HxWx3, RGB
    catch err
        fprintf('[warn] skipping unreadable image %s: %s\n', p, err.message);
        continue
    end

    feats = struct();
    feats = merge_struct(feats, compute_ela_features(img_rgb, ELA_JPEG_QUALITY));
    feats = merge_struct(feats, compute_dct_features_rgb(img_rgb, BLOCK_SIZE));
    feats = merge_struct(feats, compute_fft_features_rgb(img_rgb));
    feats = merge_struct(feats, compute_noise_features_rgb(img_rgb, WAVELET_NAME, WAVELET_LEVEL));
    feats = merge_struct(feats, compute_highpass_features_rgb(img_rgb));
    feats = merge_struct(feats, compute_cfa_features(img_rgb));
    feats = merge_struct(feats, compute_blockiness_features_rgb(img_rgb, BLOCK_SIZE));

    if isempty(feature_names)
        feature_names = fieldnames(feats);
    end

    feat_rows{i} = feats;
    keep(i) = true;

    if SAVE_STAGE_FIGURES
        try
            save_stage_figure(img_rgb, p, all_class{i}, figures_dir, ...
                ELA_JPEG_QUALITY, BLOCK_SIZE, WAVELET_NAME, WAVELET_LEVEL);
        catch err
            fprintf('[warn] could not save stage figure for %s: %s\n', p, err.message);
        end
    end

    if mod(i, 10) == 0 || i == n
        fprintf('  [%d/%d] %s (%s)\n', i, n, p, all_class{i});
    end
end

all_paths  = all_paths(keep);
all_labels = all_labels(keep);
all_class  = all_class(keep);
feat_rows  = feat_rows(keep);

%%  BUILD FEATURE TABLE AND WRITE TO EXCEL

n_kept = numel(feat_rows);
n_feat = numel(feature_names);
X = zeros(n_kept, n_feat);
for i = 1:n_kept
    for j = 1:n_feat
        X(i, j) = feat_rows{i}.(feature_names{j});
    end
end

[~, filenames, exts] = cellfun(@fileparts, all_paths, 'UniformOutput', false);
filenames = strcat(filenames, exts);

T = table(filenames, all_class, all_labels, 'VariableNames', {'filename', 'class', 'label'});
Tfeat = array2table(X, 'VariableNames', feature_names);
T = [T, Tfeat];

xlsx_path = fullfile(output_dir, 'dsp_features.xlsx');
writetable(T, xlsx_path);

fprintf('[done] %d images processed (%d authentic, %d forged)\n', ...
    n_kept, sum(all_labels == 0), sum(all_labels == 1));
fprintf('[done] numeric features -> %s\n', xlsx_path);
if SAVE_STAGE_FIGURES
    fprintf('[done] preprocessing-stage figures -> %s\n', figures_dir);
end

%%  LOCAL FUNCTIONS


function d = resolve_dir(name, script_dir)
candidate = fullfile(pwd, name);
if exist(candidate, 'dir')
    d = candidate;
    return
end
candidate2 = fullfile(script_dir, name);
if exist(candidate2, 'dir')
    d = candidate2;
    return
end
error('extract_dataset_features:missingdir', ...
    'Could not find a "%s" folder in "%s" or "%s".', name, pwd, script_dir);
end

function files = list_images(folder, extensions)
files = {};
if ~exist(folder, 'dir')
    return
end
for i = 1:numel(extensions)
    listing = dir(fullfile(folder, extensions{i}));
    for k = 1:numel(listing)
        files{end + 1} = fullfile(listing(k).folder, listing(k).name); %#ok<AGROW>
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

% --------------------------- Preprocessing ---------------------------

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
    'ela_mean', mean(m(:)), ...
    'ela_std', std(m(:)), ...
    'ela_max', max(m(:)), ...
    'ela_p95', prctile(m(:), 95), ...
    'ela_energy', mean(m(:) .^ 2), ...
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

n_rows = h_c / block;
n_cols = w_c / block;
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
if isempty(digit_chunks)
    digits = 1;
else
    digits = vertcat(digit_chunks{:});
end
end

function feats = compute_dct_features_rgb(img_rgb, block)
digits_all = [];
energy_all = [];
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
    'dct_benford_chi2', chi_sq, ...
    'dct_benford_kl', kl_div, ...
    'dct_energy_mean', mean(energy_all), ...
    'dct_energy_std', std(energy_all), ...
    'dct_energy_cv', std(energy_all) / (mean(energy_all) + 1e-6));
end

% ------------------- 3. FFT / global spectral features -----------------


function mag = fft_log_magnitude(channel)
f = fftshift(fft2(double(channel)));
mag = log1p(abs(f));
end

function feats = compute_fft_features_rgb(img_rgb)
low_all = []; mid_all = []; high_all = []; high_region_all = [];
power_flat_all = [];
mag_thr_pool = [];

for c = 1:3
    channel = img_rgb(:, :, c);
    mag = fft_log_magnitude(channel);
    [h, w] = size(mag);
    cy = floor(h / 2) + 1;
    cx = floor(w / 2) + 1;
    [xx, yy] = meshgrid(1:w, 1:h);
    r = sqrt((yy - cy) .^ 2 + (xx - cx) .^ 2);
    r_norm = r / max(r(:));

    low_all = [low_all; mag(r_norm < 0.15)]; 
    mid_all = [mid_all; mag(r_norm >= 0.15 & r_norm < 0.5)]; 
    high_all = [high_all; mag(r_norm >= 0.5)]; 
    mag_thr_pool = [mag_thr_pool; mag(:)]; 
    high_region_all = [high_region_all; mag(r_norm >= 0.5)]; 

    power = abs(fft2(double(channel))) .^ 2;
    pf = power(:);
    pf = pf(pf > 0);
    power_flat_all = [power_flat_all; pf]; 
end

log_p = log(power_flat_all);
gmean = exp(mean(log_p));
amean = mean(power_flat_all);
flatness = gmean / (amean + 1e-9);

high_thr = prctile(mag_thr_pool, 99.5);
peak_ratio = mean(high_region_all > high_thr);

feats = struct( ...
    'fft_low_mean', mean(low_all), ...
    'fft_mid_mean', mean(mid_all), ...
    'fft_high_mean', mean(high_all), ...
    'fft_high_low_ratio', mean(high_all) / (mean(low_all) + 1e-6), ...
    'fft_spectral_flatness', flatness, ...
    'fft_highfreq_peak_ratio', peak_ratio);
end

% ----------------- 4. Wavelet noise-residual features -------------------

function residual = noise_residual_map(channel, wavelet, level)
g = double(channel);
[C, S] = wavedec2(g, level, wavelet);
n_approx = S(1, 1) * S(1, 2);
C_zeroed = C;
C_zeroed(1:n_approx) = 0;
rec = waverec2(C_zeroed, S, wavelet);
residual = rec(1:size(g, 1), 1:size(g, 2));
end

function var_map = local_variance_map(arr, block)
arr = double(arr);
[h, w] = size(arr);
h_c = floor(h / block) * block;
w_c = floor(w / block) * block;
a = arr(1:h_c, 1:w_c);

n_rows = h_c / block;
n_cols = w_c / block;
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
    'noise_residual_std', std(r_all), ...
    'noise_residual_energy', mean(r_all .^ 2), ...
    'noise_var_map_mean', mean(vm_all), ...
    'noise_var_map_std', std(vm_all), ...
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
    'hp_energy', mean(hp_all .^ 2), ...
    'hp_mean_abs', mean(abs(hp_all)), ...
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

even_rows = residual(1:2:end, :);
odd_rows = residual(2:2:end, :);
grid_energy_diff = abs(mean(even_rows(:)) - mean(odd_rows(:)));

feats = struct( ...
    'cfa_residual_mean', mean(residual(:)), ...
    'cfa_residual_std', std(residual(:)), ...
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

if isempty(boundary_all)
    boundary_energy = 0.0;
else
    boundary_energy = mean(boundary_all);
end
if isempty(interior_all)
    interior_energy = 1e-6;
else
    interior_energy = mean(interior_all);
end

feats = struct('blockiness_ratio', boundary_energy / (interior_energy + 1e-6));
end

% ------------------------ Preprocessing-stage figure -----------------------

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

function save_stage_figure(img_rgb, src_path, class_name, figures_dir, ...
    ela_quality, block_size, wavelet_name, wavelet_level)

ela = ela_map(img_rgb, ela_quality);

dct_maps = cell(1, 3); fft_maps = cell(1, 3); res_maps = cell(1, 3); hp_maps = cell(1, 3);
for c = 1:3
    dct_maps{c} = block_dct_ac_energy_map(img_rgb(:, :, c), block_size);
    fft_maps{c} = fft_log_magnitude(img_rgb(:, :, c));
    res_maps{c} = noise_residual_map(img_rgb(:, :, c), wavelet_name, wavelet_level);
    hp_maps{c} = abs(highpass_map(img_rgb(:, :, c)));
end
dct_rgb = rgb_composite(dct_maps{1}, dct_maps{2}, dct_maps{3});
fft_rgb = rgb_composite(fft_maps{1}, fft_maps{2}, fft_maps{3});
res_rgb = rgb_composite(res_maps{1}, res_maps{2}, res_maps{3});
hp_rgb = rgb_composite(hp_maps{1}, hp_maps{2}, hp_maps{3});

fig = figure('Visible', 'off', 'Position', [100 100 1300 800]);
tiledlayout(2, 3);

nexttile; imshow(img_rgb); title('Original (preprocessed, RGB)', 'FontSize', 10); axis off
nexttile; imshow(ela, []); colormap(gca, 'hot'); title('ELA residual', 'FontSize', 10); axis off
nexttile; imshow(dct_rgb); title('Block-DCT AC energy (RGB)', 'FontSize', 10); axis off
nexttile; imshow(fft_rgb); title('log |FFT| (RGB)', 'FontSize', 10); axis off
nexttile; imshow(res_rgb); title('Wavelet noise residual (RGB)', 'FontSize', 10); axis off
nexttile; imshow(hp_rgb); title('Laplacian high-pass |.| (RGB)', 'FontSize', 10); axis off

[~, base, ~] = fileparts(src_path);
sgtitle(['DSP preprocessing stages: ' base ' (' class_name ')'], 'Interpreter', 'none');
out_name = sprintf('%s_%s.png', class_name, base);
exportgraphics(fig, fullfile(figures_dir, out_name), 'Resolution', 150);
close(fig);
end
