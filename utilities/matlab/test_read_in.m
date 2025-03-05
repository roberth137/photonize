clear all; clc;



tt = photonscore.read_photons(filePath);
dt_channel = photonscore.file_info(filePath).dt_channel;
data.x = photonscore.file_read(filePath,'/photons/x');
data.y = photonscore.file_read(filePath,'/photons/y');
data.dt = photonscore.file_read(filePath,'/photons/dt');
data.ms = photonscore.file_read(filePath,'/photons/ms');
    
    %crop file to full ms and shift ms counter
    %full_ms_photons = data.ms(end);
    %data.x = data.x(1:full_ms_photons);
    %data.y = data.y(1:full_ms_photons);
    %data.dt = data.dt(1:full_ms_photons);