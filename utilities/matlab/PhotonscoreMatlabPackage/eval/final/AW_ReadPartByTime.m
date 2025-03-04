function [DataPart] = AW_ReadPartByTime(Data, aat1, aat2)
%% [DataPart] = AW_ReadPartByTime(Data, aat1, aat2)
%% aat1: start (time in seconds)
%% aat2: end (time in seconds)
%% Data: struct of readPhotons.m
% Author: Andre Weber
if aat1 ==0
    
    event_from_t = find(Data.ms >0,1);
    
else
    event_from_t = Data.ms(int32(aat1*1000));
end

event_to_t   = Data.ms(int32(aat2*1000));
trange       = (event_from_t:event_to_t);

if event_to_t>numel(Data.x) || event_from_t>numel(Data.x)
    error ('load more Photons of the file into memory! ms > loaded Data')
end

DataPart.x   = Data.x(trange);
DataPart.y   = Data.y(trange);
DataPart.dt  = Data.dt(trange);

end