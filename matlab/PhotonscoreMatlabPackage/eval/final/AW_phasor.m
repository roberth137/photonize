function [g_ij,s_ij, semicircel] = AW_phasor(I_ij ,dt_channel, h_plot)
%% Phasor
% [g_ij,s_ij, semicircel] = AW_phasor(I_ij ,dt_channel, h_plot)
% I_ij = decay for each pixel;I = [dt,px] 
% w = 2*pi*19.5*1000*1000;
t = (0:size(I_ij,1)-1)*dt_channel/(10^12);
w =2*pi./t(end);
%%
% tmp1= sum(I_ij(:,:).*repmat(cos(w*t),size(I_ij,2),1)')./sum(I_ij,1);
semicircel(1,:) = [1./(1+(w*t).*(w*t))]; 
semicircel(2,:) = [w*t./(1+(w*t).*(w*t))]; 
g_ij =  sum(I_ij(:,:).*repmat(cos(w*t),size(I_ij,2),1)')./sum(I_ij,1) ;
s_ij =  sum(I_ij(:,:).*repmat(sin(w*t),size(I_ij,2),1)')./sum(I_ij,1) ;

if h_plot
%     figure;
%     subplot(211)
%         plot(tmp1)
%     subplot(212)
%         plot(cos(w*t)); hold on;
%     plot(sin(w*t))
    plot(semicircel(1,:),semicircel(2,:), ...
            'Color', [0.1 0.1 0.1 0.1],'LineWidth',2); hold on;
    plot(g_ij, s_ij, '.');
xlabel('g_{i,j}');
ylabel('s_{i,j}')
grid on;
xlim([-0.01 1.01 ])
ylim([-0.01 1.01 ])
end
%% 
end