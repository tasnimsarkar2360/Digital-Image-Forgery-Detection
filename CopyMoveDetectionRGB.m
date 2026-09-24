
clc; clear; close all;

%% ----------------------- Parameters -----------------------
IMAGE_FILE = '15733.png';
MAX_DIM    = Inf;   
B          = 8;     
NCOEF      = 10;    
QSCALE     = 2;     
MATCH_TOL  = 0;     
SEARCH_WIN = 5;     
MIN_STD    = 3;     
MIN_SHIFT  = 32;    
MIN_VOTES  = 200;   
MIN_AREA   = 500;   

Q75 = [8    6    5    8    12    20    26    31;       % JPEG quantization matrix (75% quality)
       6    6    7    10    13    29    30    28;
       7    7    8    12    20    29    35    28;
       7    9    11    15    26    44    40    31;
       9    11    19    28    34   55   52    39;
       12    18    28    32    41   52   57    46;
       25    32    39    44   52   61   60   52;
       36    46    48    49   56   50   52    50 ];

%% ----------------------- Read image -----------------------
[i1, map] = imread(IMAGE_FILE);
if ~isempty(map), i1 = im2uint8(ind2rgb(i1, map)); end   
if size(i1, 3) == 1, i1 = repmat(i1, [1 1 3]); end       
if size(i1, 3) == 4, i1 = i1(:, :, 1:3); end           
figure, imshow(i1), title('Original');

sc = min(1, MAX_DIM / max(size(i1, 1), size(i1, 2)));
if sc < 1, i2 = imresize(i1, sc); else, i2 = i1; end
x = 255 * im2double(i2) - 128;                           
[H, W, ~] = size(x);
nbr = H - B + 1; nbc = W - B + 1;                        

%% ----------------------- Low-frequency block DCT (all blocks at once) -----------------------
D = dctmtx(B);
[cz, rz] = meshgrid(1:B);                                
sz = rz + cz;
key = sz * 2 * B + (mod(sz, 2) == 0) .* rz + (mod(sz, 2) == 1) .* cz;
[~, zz] = sort(key(:));
[ru, cu] = ind2sub([B B], zz(1:NCOEF));                  

feat = zeros(nbr * nbc, 3 * NCOEF);
blkStd = zeros(nbr, nbc);
box = ones(B) / B^2;
for c = 1:3
    ch = x(:, :, c);
    for k = 1:NCOEF
        basis = D(ru(k), :)' * D(cu(k), :);              
        coef = conv2(ch, rot90(basis, 2), 'valid');      
        feat(:, (c-1)*NCOEF + k) = round(coef(:) / (Q75(ru(k), cu(k)) * QSCALE));
    end
    m  = conv2(ch, box, 'valid');
    m2 = conv2(ch.^2, box, 'valid');
    blkStd = blkStd + sqrt(max(m2 - m.^2, 0)) / 3;
end
[rr, cc] = ndgrid(1:nbr, 1:nbc);
pos = [rr(:) cc(:)];                                    

keep = blkStd(:) >= MIN_STD;                             
feat = feat(keep, :); pos = pos(keep, :);

%% ----------------------- Lexicographic sort + matching -----------------------
[featSorted, order] = sortrows(feat);
posSorted = pos(order, :);
nRows = size(featSorted, 1);

pairs = zeros(0, 4); shifts = zeros(0, 2);
for w = 1:SEARCH_WIN
    a = (1:nRows - w)'; b = a + w;
    ok = max(abs(featSorted(a, :) - featSorted(b, :)), [], 2) <= MATCH_TOL;
    a = a(ok); b = b(ok);
    s = posSorted(b, :) - posSorted(a, :);
    flip = s(:, 1) < 0 | (s(:, 1) == 0 & s(:, 2) < 0);  
    s(flip, :) = -s(flip, :);
    far = sqrt(sum(s.^2, 2)) >= MIN_SHIFT;
    pairs  = [pairs;  posSorted(a(far), :) posSorted(b(far), :)]; 
    shifts = [shifts; s(far, :)];                                  
end

%% ----------------------- Vote for shift vectors -----------------------
mask = false(H, W);
if ~isempty(shifts)
    hist2 = accumarray([shifts(:, 1) + 1, shifts(:, 2) + W + 1], 1, [H + 1, 2*W + 1]);
    pooled = conv2(hist2, ones(3), 'same');             
    votes = pooled(sub2ind(size(pooled), shifts(:, 1) + 1, shifts(:, 2) + W + 1));
    good = votes >= MIN_VOTES;

    [~, iMax] = max(pooled(:)); [py, px] = ind2sub(size(pooled), iMax);
    fprintf('Strongest shift (dy, dx) = (%d, %d) with %d votes\n', ...
        py - 1, px - W - 1, round(pooled(iMax)));

    for p = find(good)'
        r1 = pairs(p, 1); c1 = pairs(p, 2); r2 = pairs(p, 3); c2 = pairs(p, 4);
        mask(r1:r1+B-1, c1:c1+B-1) = true;
        mask(r2:r2+B-1, c2:c2+B-1) = true;
    end
    mask = imclose(mask, strel('square', 5));
    mask = bwareaopen(mask, MIN_AREA);
end

if any(mask(:))
    fprintf('Result: copy-move forgery DETECTED\n');
else
    fprintf('Result: no copy-move region found\n');
end

%% ----------------------- Display -----------------------
figure, imshow(mask); title('Copy-moved part (RGB detection)');
bigMask = imresize(mask, [size(i1, 1) size(i1, 2)], 'nearest');
overlay = im2double(i1);
R = overlay(:, :, 1); G = overlay(:, :, 2); Bl = overlay(:, :, 3);
R(bigMask)  = 0.5 * R(bigMask) + 0.5;                    
G(bigMask)  = 0.5 * G(bigMask);
Bl(bigMask) = 0.5 * Bl(bigMask);
figure, imshow(cat(3, R, G, Bl)); title('Detected copy-move region (red)');