%% Visualise histogram of dt values
clear all; clc;

[file,path] = uigetfile('*.photons');
fname = fullfile(path,file);

%tt = photonscore.read_photons(fname);
%dt_channel = photonscore.file_info(fname).dt_channel;
data.dt = photonscore.file_read(fname,'/photons/dt');
hist_dt = histogram(data.dt(1:100:end), 'FaceColor', 'magenta', 'EdgeColor','non');
yscale('log')
grid on;
pause;
close all