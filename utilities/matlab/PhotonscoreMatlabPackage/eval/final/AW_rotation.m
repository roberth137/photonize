function [Datax_rot, Datay_rot] = AW_rotation (Data, angle_deg)
%% Rotation Photons position
% % [x_rot, y_rot]= AW_rotation (data, angle)
% give angle in degree
% data.x data.y is photon list as strcuture
angle = deg2rad(angle_deg);
sin_theta = sin(angle);
cos_theta = cos(angle);
r=1:numel(Data.x );
Data.x_rot = uint16(((single(Data.x(r))-2048) *cos_theta)+...
                    ((single(Data.y(r))-2048) *sin_theta)+2048);
Data.y_rot = uint16(((single(Data.y(r))-2048) *cos_theta)+2048)-...
                    ((single(Data.x(r))-2048) *sin_theta);
% imagesc(photonscore.hist_2d(uint16(Data.x_rot), 0, 4096, dim, uint16(Data.y_rot)));
Datax_rot=Data.x_rot;
Datay_rot=Data.y_rot;
end