function [Data] = AW_ReadPhotonsByTime(Data,range_seconds, filename )
% % [Data] = AW_ReadPhotonsByTime(Data,range_seconds, filename )
Data.MSIndex = Data.MSIndex;
if nargin == 1
    offset = 0;
    count = Data.MSIndex(end);
else
    ms_range = int32(range_seconds(:) * 1000 + 1);
    a = min(ms_range);
    b = max(ms_range);
    b = min(b, length(Data.MSIndex));
    offset = Data.MSIndex(a);
    count = Data.MSIndex(b) - offset;
end

Data.x = photonscore.file_read(filename, '/photons/x', offset, count);
Data.y = photonscore.file_read(filename, '/photons/y', offset, count);
Data.dt = photonscore.file_read(filename, '/photons/dt', offset, count);
end