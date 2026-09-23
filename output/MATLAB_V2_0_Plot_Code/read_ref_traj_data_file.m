function s = read_ref_traj_data_file(file)

fid = fopen(file,'r');
s = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));

    % skip empty or header lines
    if isempty(line) || startsWith(line,'#')
        continue
    end

    % find first comma (end of timestamp)
    idx = strfind(line, ',');
    if isempty(idx)
        continue
    end

    % everything after the first comma is numeric data
    numeric_str = line(idx(1)+1:end);

    % replace commas with spaces so sscanf can read all numbers
    numeric_str = strrep(numeric_str, ',', ' ');

    % parse: epoch, x, y, z, vx, vy, vz
    nums = sscanf(numeric_str, '%f');

    if numel(nums) == 6
        s(end+1, :) = nums.';   % keep only x y z vx vy vz
    end
end

fclose(fid);

s = s';
