function [ tau_range_exp, Fit_spectrum, Fit_model, Fit_Res_MEM, Fit_chi_MEM] = ...
    AW_ZooMEMFitInf(decay,irf,tau_range, shift,tau_ref, ch,iterations, plotFigure)
 %% [ tau_range_exp, spectrum, model, Res_MEM, chi_final] = ...
 %%   AW_ZooMEMFitInf(decay,irf,tau_range, shift,tau_ref, ch,iterations, plotFigure)

%% DataFitting
channels = length(irf);
% tau_range_lin =tau_range';
% %exponential
tau_range_exp = exp(tau_range * log(max(tau_range)) / max(tau_range));
tau_range_exp = tau_range_exp(:);
% 
% if plotFigure
%     figure
%     plot(tau_range_exp*ch/1000)
%     xlabel('channels')
%     ylabel('lifetimes [ns]')
% end

    Fit_model = zoo.ExpConvolve(irf, shift, channels, tau_range_exp, tau_ref);
    Fit_model = abs(Fit_model);
    %%
    for i=1:size(Fit_model, 2)
        temp = Fit_model(:, i);
        Fit_model(:, i) = temp / sum(temp);
    end
    %  keyboard;
    Fit_model(:,size(Fit_model, 2)+1) = ones(channels, 1)/sum(ones(channels, 1));
    Fit_model= abs(Fit_model);
    m_Inf = 1 - sum(Fit_model, 1);
    m_Inf(end) = 0;
    m_Inf = m_Inf';
    
ff = ones(numel(tau_range_exp)+1, 1);
% %
% figure
% imagesc(model)
% set(gca, 'XTick', tau_range_exp(1:50:end))
% set(gca, 'XTickLabel', tau_range_exp(1:50:end)*ch/1000)

%%
x=ff;
    
    for i=1:iterations
        x = zoo.em.ref.UnmixInf(Fit_model, m_Inf, decay, x);
        %         ll(i) = zoo.PoissonLogL(decay, Fit_model*x);
    end
    
    Fit_spectrum=x;
    if plotFigure
        figure;
        subplot(311)
        semilogy(decay)
        hold on;
        plot(Fit_model*x, '.-');
        ylabel('Counts', 'FontSize', 16)
        hold off;
        set(gca, 'FontSize', 14)
        xlim([0 numel(irf)]);
        yticki = 10.^(round(log10(1+min([irf(:); decay(:)]))) : ...
        ceil(log10(1+max([decay(:)]))));
        ylim([yticki(1) yticki(end)*2]);
        grid on;
        set(gca, 'YTick', yticki);
        legend('Decay', 'Fit', 'Location', 'NorthEast')
        subplot(313)
        plotSpectrumAWTE(tau_range_exp*ch, x(1:numel(tau_range_exp)));
        set(gca, 'FontSize', 14)
        % %% Residues and Modell-fits
        subplot(312)
        % set(figure(3),'position',[5 45 1275 900]);
        scF=1;
    end
    [MAX_Decay1,MAX_Decay2]=max(decay);
    
    Fit_Fit = Fit_model * x;
    temp = (decay - Fit_Fit)./sqrt(decay);
    Fit_Res_MEM=temp;
    s = ~isnan(temp) & ~isinf(temp);
    temp = temp(s);
    chi_max= sum(temp(MAX_Decay2:end)' * temp(MAX_Decay2:end))/...
        length(temp((MAX_Decay2:end)));
    chi_tot = sum(temp' * temp)/length(temp);
    
   Fit_I2_chiMEM=[zoo.PoissonLogL(decay, Fit_Fit)/length(decay(decay>0))...
            sum(((decay(decay>0) - ...
            Fit_Fit(decay>0))./sqrt(decay(decay>0))).^2)/length(decay(decay>0))];
 
    if plotFigure
        plot(Fit_Res_MEM)
        legend([ '2I*=',num2str(Fit_I2_chiMEM(1)),...
                 '; \chi^2 = ' , num2str(chi_tot),' ; \chi_{max}^2 = ' ,...
                 num2str(chi_max), '; Shift=', num2str(shift)], ...
                 'Location', 'best')
        set(gca, 'FontSize', 14)
        axheigth=get(gca, 'Position');
        axheigth(4)=axheigth(4)*scF;
        set(gca, 'Position', axheigth );
        ylabel('Resid.');
        labelss=get(gca, 'XTickLabel');
        set(gca, 'XTickLabel', []);
        grid on
        set(gca, 'XTickLabel',labelss)
        xlabel('channels [ch=12.4ps]', 'FontSize', 16)
%         title('Residues', 'FontSize', 16)
        xlim([0 numel(irf)])
    end
% %      saveas(gcf,[SaveName,'logSpect_Fit',num2str(iterations),'it.fig'])
% %      exportFigure([SaveName,'logSpect_Fit',num2str(iterations),'it.png'], ...
% %     %                 '300');
Fit_chi_MEM=[chi_tot chi_max Fit_I2_chiMEM];
end

