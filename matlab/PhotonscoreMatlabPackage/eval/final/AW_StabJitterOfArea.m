function [Decay_t] = AW_StabJitterOfArea(Data, area, time_start, time_end, ...
    time_bin)
%% [Decay_Irf_t] = AW_StabJitterOfArea(Data, area, time_start, time_end, ...
% %                                         time_bin, dim)
% Output: Decay over time of rectangular area
% clear s Decay_Irf_t
dim=4096;
if time_start == 0
    duration=time_start:time_bin:time_end;
    Decay_t=NaN(numel(duration),4096);
    j=1;
    for k=time_start:time_bin:time_end-1
        if k==0
           event_from_t = 1;
        else
           event_from_t = Data.ms(int32(k*1000));
        end
        
        event_to_t   = Data.ms(int32(k*1000+time_bin*1000));
        trange = (event_from_t:event_to_t);
        
        s = dim/2-area*3 < Data.x(trange) & Data.x(trange) < dim/2+area*3 &...
            dim/2-area*3 < Data.y(trange) & Data.y(trange) < dim/2+area*3;
        tmp_dt=Data.dt(trange);
        Decay_t (j,:)= photonscore.hist_1d(tmp_dt(s), 0, 4096, 4096);   
%         Decay_t (j,:)= zoo.Hist1D(Data.dt(s), 0, 4096, 4096);
        j=j+1;
    end
else
    duration=time_start:time_bin:time_end;
    Decay_t=NaN(numel(duration),4096);
    j=1;
    for k=time_start:time_bin:time_end-1
        
        event_from_t = Data.ms(int32(k*1000));
        event_to_t   = Data.ms(int32(k*1000+time_bin*1000));
        trange = (event_from_t:event_to_t);
        
        s = dim/2-area*3 < Data.x(trange) & Data.x(trange) < dim/2+area*3 &...
            dim/2-area*3 < Data.y(trange) & Data.y(trange) < dim/2+area*3;
        
        tmp_dt=Data.dt(trange);
        Decay_t (j,:)= photonscore.hist_1d(tmp_dt(s), 0, 4096, 4096); 
%         Decay_t (j,:)= zoo.Hist1D(Data.dt(s), 0, 4096, 4096);

        j=j+1;
    end
end
end