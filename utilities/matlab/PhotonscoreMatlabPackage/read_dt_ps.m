function dtps = read_dt_ps(filename)
start = photonscore.file_read(filename, '/start/time');
stop = photonscore.file_read(filename, '/stop/time');
dtps = int64(start) - int64(stop);
end
