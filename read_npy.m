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
shape_str = shape_str{1};
shape = str2double(strsplit(strtrim(shape_str), ','));
shape = shape(~isnan(shape));
if isempty(shape)
    shape = 1;
end

[matlab_type, ~] = npy_dtype_to_matlab(descr);
raw = fread(fid, prod(shape), [matlab_type '=>double']);

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

function [matlab_type, nbytes] = npy_dtype_to_matlab(descr)
map = struct( ...
    'f4', 'single', 'f8', 'double', ...
    'i1', 'int8', 'i2', 'int16', 'i4', 'int32', 'i8', 'int64', ...
    'u1', 'uint8', 'u2', 'uint16', 'u4', 'uint32', 'u8', 'uint64', ...
    'b1', 'uint8');
key = descr(2:end); 
if strcmp(key, '?') || strcmp(descr(end), '?')
    matlab_type = 'uint8';
else
    if ~isfield(map, key)
        error('read_npy:dtype', 'Unsupported .npy dtype: %s', descr);
    end
    matlab_type = map.(key);
end
nbytes = 0;
end
