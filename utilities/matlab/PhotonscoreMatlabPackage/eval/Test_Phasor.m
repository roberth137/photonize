addpath(genpath('C:\Users\AW\Desktop\matlab-windows-opt2019'));

figure;
subplot(211)
    imagesc(fr455.image)
dim = size(fr455.image,1);
dt_channel= 12;
mask_id = 1:20*20;
mask_id=reshape(mask_id,[20 20])
tmp=zeros(dim,dim);
tmp(201:220,201:220)=mask_id;
subplot(212)
imagesc(tmp);axis square
%%
img_dec = photonscore.flim.decay_from_mask(fr455,tmp, 0, 4096, 4096);
I_ij=img_dec(685:end,2:end);
%%
close all
t_phasor = 680:3900;
clear I_ij
tau=[0.01:.1:10]*10^(-9)
for j=1:size(tau,2)
I_ij(:,j) = 2^j*exp(-t_phasor*dt_channel/10^12/tau(j))+ ...
    (.1^j)*exp(-t_phasor*dt_channel/10^12/tau(end))+(.1^j+2^j)*img_dec(t_phasor,1)';
end

subplot(211)
semilogy(I_ij)
% %
% % Phasor
h_plot=1;
subplot(212)
[g_ij,s_ij, sc] = AW_phasor(I_ij ,dt_channel, 1);
% plot(g_ij, s_ij, '.');
% xlabel('g')
% ylabel('s')
%%
t_phasor = 685:3930;
 [g_ij,s_ij, sc] = ...
        AW_phasor(squeeze(Decay_roi_T(1,t_phasor,1:2)) ,dt_channel,0);
figure;
    plot(sc(1,:),sc(2,:), ...
         'Color', [0.1 0.1 0.1 0.1],'LineWidth',2); hold on
% %
aat_range=[ 50:100 160:200];

clear g_ij s_ij
g_ij=NaN(numel(aat_range), size(Decay_roi_T,3));
s_ij=NaN(numel(aat_range), size(Decay_roi_T,3));
j=1;
for i=aat_range
    [g_ij(j,:),s_ij(j,:), ~] = ...
        AW_phasor(squeeze(Decay_roi_T(i,t_phasor,:)) ,dt_channel,0);
   
  plot(g_ij(:), s_ij(:), '.', 'Color', [.7 .7 .7 .1]);hold on;
  plot(g_ij(j,:), s_ij(j,:), '.')
    xlabel('g_{i,j}');
    ylabel('s_{i,j}')
    grid on;
    xlim([-0.01 1.01 ])
    ylim([-0.01 1.01 ])
    pause (.1);
    hold off;
     j=j+1;
end
%%
figure; plot(int_roi(aat_range,:), TauMeanRois(aat_range,:))