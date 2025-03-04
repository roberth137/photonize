function Decay_Data_Dim = AW_3DtacPixel(data,dim,mask)
%% Decay_Data_Dim = AW_3DtacPixel(data,dim,mask)
%% equal to AW_3DtacImage, but MATLAB based
%% use if no MEXfile lnt_an_matlab available
%% Thanks to Kristian Loewe
mask_lin = mask(:) > 0;
index = single(sub2ind([dim dim], data.x, data.y));
take = mask_lin(index);
idx_mask = single(find(mask_lin));
idx_mask = [idx_mask(1)-0.5; idx_mask+0.5];

if isfield(data,'dtc')
    [Decay_Data_Dim,~,~] = histcounts2(data.dtc(take),index(take),0:4096,idx_mask);
else
    [Decay_Data_Dim,~,~] = histcounts2(data.dt(take),index(take),0:4096,idx_mask);
    warning('no irf_shift_xy compensation! found')
end
Decay_Data_Dim = single(Decay_Data_Dim);
end

