function d = read_photons(filename, range_seconds)
ms = photonscore.file_read(filename, '/photons/ms');

if nargin == 1
    offset = 0;
    count = ms(end);
else
    ms_range = int32(range_seconds(:) * 1000 + 1);
    a = min(ms_range);
    b = max(ms_range);
    b = min(b, length(ms));
    offset = ms(a);
    count = ms(b) - offset;
end

d.x = photonscore.file_read(filename, '/photons/x', offset, count);
d.y = photonscore.file_read(filename, '/photons/y', offset, count);
d.dt = photonscore.file_read(filename, '/photons/dt', offset, count);
end
