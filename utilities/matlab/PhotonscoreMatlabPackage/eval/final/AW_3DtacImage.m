function Full_Decay_Data_dim = AW_3DtacImage( data, dim, mask,range_tac)
% % [Decay_Data_dim ] = AW_3DtacImage( data, dim, mask,range_tac)
lx = (int32(data.y)'-1) * dim + int32(data.x)';
if isfield(data,'dtc')
    Full_Decay_Data_dim = lnt_an.hist_2d(int32(data.dtc), 0, 4095, 4096, lx, 1, (dim*dim), dim*dim);
else
    Full_Decay_Data_dim = lnt_an.hist_2d(int32(data.dt), 0, 4095, 4096, lx, 1, (dim*dim), dim*dim);
    warning('no irf_shift_xy compensation! found')
end
Full_Decay_Data_dim = single(Full_Decay_Data_dim(range_tac, logical(mask)));
end
