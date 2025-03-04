function [ img_test ] = AW_ReshapeMaskDim(Decay_Data_dim, mask_m0, dim ,range_tac)
%% [ img_test ] = AW_ReshapeMaskDim(Decay_Data_dim, mask_m0, dim ,range_tac)
%% reshape(img_test, [dim, dim, length(range_tac) ])
%AW_RESHAPEMASKDIM 
img_test= NaN(dim*dim, length(range_tac) );
img_test(mask_m0(:),:) = Decay_Data_dim';
img_test=reshape(img_test, [dim, dim, length(range_tac) ]);
% figure;
% imagesc(sum(img_test,3));
% axis image
end
%%% test
%% test_m0= ones(128, 128);test_m0(50:100, 1:10)=0;test_m0(1:40, 30:120)=0; imagesc(test_m0)
%% figure; imagesc(AW_ReshapeMaskDim(flat(test_m0(mask_m0>0)),test_m0,128,1))