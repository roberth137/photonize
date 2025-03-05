function [data, dt_channel] = dotPhotons_toFullMS(filePath)

    tt = photonscore.read_photons(filePath);
    dt_channel = photonscore.file_info(filePath).dt_channel;
    %data.x = photonscore.file_read(filePath,'/photons/x');
    %data.y = photonscore.file_read(filePath,'/photons/y');
    %data.dt = photonscore.file_read(filePath,'/photons/dt');
    data.ms = photonscore.file_read(filePath,'/photons/ms');
    
    %crop file to full ms and shift ms counter
    full_ms_photons = data.ms(end);
    data.x = tt.x(1:full_ms_photons);
    data.y = tt.y(1:full_ms_photons);
    data.dt = tt.dt(1:full_ms_photons);
end