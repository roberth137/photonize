function [FitResults_t_ROI_nr] = AW_easyFit (decay, irf, ch,...
    tau, tau_fix, alpha, alpha_fix, ...
    irf_shift, irf_shift_fix,...
    tau_ref, tau_ref_fix,...
    bg, bg_fix)
% %  function [FitResults_t_ROI_nr] = AW_easyFit (decay, irf, ch,...
% %     tau, tau_fix, alpha, alpha_fix, ...
% %     irf_shift, irf_shift_fix,...
% %     tau_ref, tau_ref_fix,...
% %     bg, bg_fix)

%% tau_ref=1; %% for Reference Dye with known lifetime for IRF
%% tau = [0.35 1.62 3.41]'/ch*1000; % Lifetime i.e. 3 exp Lifetime
%% alpha = [ones(size(tau))]' * 1000; % Contribution i.e. 3 exp Lifetime
% %         a = All photons (area under decay)
%%  bg = 3;   % background
%%  irf_shift=-0.8; % missmatch IRF data in ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initial conditions
clear x0 lo up % initial fitting parameters and limits
x0.tau = tau;
if tau_fix==0
    lo.tau = x0.tau * 0.1;
    up.tau = x0.tau * 1000;
end

x0.a = alpha';
if alpha_fix==0
    lo.a = x0.a * 0;
    up.a = x0.a * 1000000000;
end

x0.irf_shift =irf_shift;
if irf_shift_fix==0
    lo.irf_shift =  x0.irf_shift-10;
    up.irf_shift = x0.irf_shift+ 10;
end

x0.background = bg;
if bg_fix==0
    lo.background = 0;
    up.background = 1e9;
end

x0.tau_ref = tau_ref;
if tau_ref_fix==0
    lo.tau_ref = 0;
    up.tau_ref = 10;
end

FitResults  = zoo.FitDecay(irf, decay, x0, lo, up);
%         FitModel = zeros(size(decay));
%         I2_chi = zeros(2, length(FitResults.tau_ref));
[FitModel, I2_chi, Residuals]   = AW_ZooResiduals(irf, decay, ...
    FitResults, 1); I2_chi(3)

FitResults_t_ROI_nr.TauMean = ...
    sum(FitResults.a.*FitResults.tau/sum(FitResults.a))*ch;
clear FitResStr
for i = 1:size(tau,1)
    FitResStr{i} = ['\tau_', num2str(i),'=',...
        num2str(round(FitResults.tau(i,1)*ch)),' ps ; ',...
        '\alpha_', num2str(i),'=',...
        num2str(round(FitResults.a(i,1)/sum(FitResults.a)*100)),'%'];
end
FitResStr{size(tau,1)+1}=['\tau_{mean} = ',num2str(FitResults_t_ROI_nr.TauMean/1000),'ns'];
FitResStr{size(tau,1)+2}=['\chi^2_{max} = ',num2str(I2_chi(3))];
subplot(211)
text(size(decay,1)/3*2,max(decay(:)*0.1),FitResStr', 'FontSize', 18)
%         text(600,500,['\tau_{mean} = ',num2str(FitResults_t_ROI_nr{...
%             time_idx,ROI_t_50_nr}.TauMean/1000),'ns'], 'FontSize', 18, ...
%             'Units', 'pixels')


FitResults_t_ROI_nr.FitResults = FitResults;
FitResults_t_ROI_nr.FitModel   = FitModel;
FitResults_t_ROI_nr.I2_chi = I2_chi;
FitResults_t_ROI_nr.Residuals = Residuals;
FitResults_t_ROI_nr.FitResStr = FitResStr;
FitResults_t_ROI_nr.TauMean = ...
    sum(FitResults.a.*FitResults.tau/sum(FitResults.a))*ch;

end
