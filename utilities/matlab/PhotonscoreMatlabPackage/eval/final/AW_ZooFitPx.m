function [x1, x0] = AW_ZooFitPx(Decay_Data_dim, Decay_IRF_dim, ...
    tau, a, bg, tau_ref, irf_shift,...
    lo, up)
% [x1] = AW_ZooFitPx(Decay_Data_dim, Decay_IRF_dim, ...
%     tau, a, bg, tau_ref, irf_shift,...
%     lo, up)
%% structures:
%% Decay_Data_dim (tac_range,pixels)
%% lower limits: lo.tau, lo.a lo.background lo.irf_shift lo.tau_ref
%% upper limits: up.tau, up.a up.background up.irf_shift up.tau_ref
%% Input Parameters for ZOO.FitDecay
% if isempty(lo) && isempty(up)
%
%     lo.tau = tau * 0.10;
%     up.tau = tau * 5.00;
%     lo.background = 0 ;
%     up.background = max(max(Decay_Data_dim(1:80,:))) /size(Decay_Data_dim,1) ;
%     lo.a = 0;
%     up.a = nanmax(nansum(Decay_Data_dim,1));
%     lo.irf_shift = -10;
%     up.irf_shift = 10;
%     lo.tau_ref = 1;
%     up.tau_ref = 10;
%
% else
%%
% % % NADH_working parameters:
% % %     tau = [18.0475   49.8359  183.0675  554.1968]';
% % %     a = [1 1 1 1]' * 100;
% % %     bg = 0.1;
% % %     x0.tau = tau;
% % %     lo.tau = x0.tau * 0.1;
% % %     up.tau = x0.tau * 1000;
% % %     
% % %     x0.a = a';
% % %     lo.a = x0.a * 0;
% % %     up.a = x0.a * 100000000;
% % %     
% % %     x0.irf_shift =0.5;
% % %     lo.irf_shift = -10;
% % %     up.irf_shift = 10;
% % %     
% % %     x0.background = bg;
% % %     lo.background = 0;
% % %     up.background = 1e9;
% % %     
% % %     x0.tau_ref = 4.25;
% % %     lo.tau_ref = 0;
% % %     up.tau_ref = 10;
%%
if isempty(lo) && isempty(up)
    warning ('default setting for parameter limits up and low are set')
    lo.tau = tau * 0.10;
    up.tau = tau * 5.00;
    lo.background = 0 ;
    up.background = max(max(Decay_Data_dim(1:80,:))) /size(Decay_Data_dim,1) ;
    lo.a = zeros(size(a));
    up.a = ones(size(a))*nanmax(nansum(Decay_Data_dim,1));
    lo.irf_shift = -10;
    up.irf_shift = 10;
    lo.tau_ref = 1;
    up.tau_ref = 10;
    x0.tau = tau ;
    x0.a = a;
    x0.background =bg;
    x0.irf_shift = irf_shift;
    x0.tau_ref = tau_ref;
    assert(isequal(size(tau), size(a)), 'Size Tau nonequal Size Alpha');
    assert(all(size(lo)==size(up)), 'Size low nonequal Size up')
else
    assert(isequal(size(tau), size(a)), 'Size Tau nonequal Size Alpha');
    assert(all(size(lo)==size(up)), 'Size low nonequal Size up')
    %     clear x0 x1 % lo up
    %% Startparameter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x0.tau = tau ;
    x0.a = a;
    x0.background =bg;
    x0.irf_shift = irf_shift;
    x0.tau_ref = tau_ref;
    
    assert(isequal(sort(fieldnames(lo)), sort(fieldnames(up))));
    assert(all(size(lo)==size(up)))
end
%     tic;
x1 = zoo.FitDecay(Decay_IRF_dim, Decay_Data_dim , x0, lo, up);
%     fprintf('kFits / second = %f\n', size(x1.tau, 2) / 1000 / toc);

% bg=1/size(Decay_Data_dim,1);
% tau=[18.0475   49.8359  183.0675  554.1968]'%tau_mean_Mask
% a = [1/4 1/4 1/4 1/4]*mean(sum(Decay_Data_dim(:,:),1));
% irf_shift = -1.2626;

end