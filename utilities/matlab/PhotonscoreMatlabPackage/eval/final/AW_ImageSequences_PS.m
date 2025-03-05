function [Counts_t, FLIM_t]= AW_ImageSequences_PS(IRF, Data, from, to,...
                                  samplingrate, dim, ch)
%% [Counts_t, FLIM_t]= AW_ImageSequences_PS(IRF, Data, from, to,...
%%                                   samplingrate, dim, ch)                                   
% from:  Start (time in seconds)
% to:    End (time in seconds)
% For IRF 
img_irf = photonscore.flim.sort(IRF.x,0, 4095,dim, IRF.y, IRF.dt); 
tau_mean_irf =  photonscore.flim.medimean(img_irf)*ch;                
                
range=[from: samplingrate :to];

FLIM_t=NaN(dim, dim, numel(range)-1);
Counts_t=NaN(dim, dim, numel(range)-1);

for i=1:numel(range)-1   
Data_p=AW_ReadPartByTime(Data, range(i), range(i+1));

img_data = photonscore.flim.sort(Data_p.x, 0,4095, dim,Data_p.y,Data_p.dt); 
tau_mean_data=  photonscore.flim.medimean(img_data)*ch;      

tau_mean_data(isnan(tau_mean_data))=0;

FLIM=tau_mean_data-tau_mean_irf;
FLIM=single(FLIM);
FLIM(FLIM<0)=0;
FLIM_t(:,:,i)=FLIM;
Counts_t(:,:,i)=single(img_data.image);
i
end
end