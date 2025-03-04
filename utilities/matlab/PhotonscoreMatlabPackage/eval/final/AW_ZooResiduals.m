function [model, I2_chi, Residuals]   = AW_ZooResiduals(irf, decay, FitResults, plotFigure)
%%  [model, I2_chi, Residuals]   = AW_ZooResiduals(irf, decay, FitResults, plotFigure)

model = zeros(size(decay));
I2_chi = zeros(3, length(FitResults.tau_ref));

if isequal(size(irf),size(decay))
     for k=1:length(FitResults.tau_ref)
     [Dmax1, Dmax2]= nanmax(decay(:,k));
         decay_max=decay(Dmax2:end,k);
         model_k = zoo.ExpConvolve(irf(:,k)/sum(irf(:,k)), FitResults.irf_shift(k), size(decay, 1), ...
            FitResults.tau(:, k), FitResults.tau_ref(k));
        model(:, k) = model_k * FitResults.a(:, k) + FitResults.background(k) / size(model, 1);
         model_max = model(Dmax2:end,k);
        I2_chi(:,k)=[zoo.PoissonLogL(decay(:,k), model(:,k))/length(decay(decay(:,k)>0,k))...
            sum(((decay(decay(:,k)>0,k) - ...
            model(decay(:,k)>0,k))./sqrt(decay(decay(:,k)>0,k))).^2)/length(decay(decay(:,k)>0,k))...
            sum(((decay_max(decay_max(:,k)>0,k) - ...
            model_max(decay_max(:,k)>0,k))./sqrt(decay_max(decay_max(:,k)>0,k))).^2)/length(decay_max(:,k))];
    end
else
    
    for k=1:length(FitResults.tau_ref)
        
        [Dmax1, Dmax2]= nanmax(decay(:,k));
         decay_max=decay(Dmax2:end,k);
        model_k = zoo.ExpConvolve(irf/sum(irf), FitResults.irf_shift(k), size(decay, 1), ...
            FitResults.tau(:, k), FitResults.tau_ref(k));
        model(:, k) = model_k * FitResults.a(:, k) + FitResults.background(k) / size(model, 1);
          model_max = model(Dmax2:end,k);
        I2_chi(:,k)=[zoo.PoissonLogL(decay(:,k), model(:,k))/length(decay(:,k))...
            sum(((decay(decay(:,k)>0,k) - ...
            model(decay(:,k)>0,k))./sqrt(decay(decay(:,k)>0,k))).^2)/length(decay(:,k))...
            sum(((decay_max(decay_max(:,k)>0,k) - ...
            model_max(decay_max(:,k)>0,k))./sqrt(decay_max(decay_max(:,k)>0,k))).^2)/length(decay_max(:,k))];
    end
end
Residuals = (decay - model)./ sqrt(decay);

    if length(FitResults.tau_ref)>20 && plotFigure
        warning('For plotoption size Decay < 20')
    end
if plotFigure && length(FitResults.tau_ref)<20

    figure;
    subplot(2,1,1);
    pdec=semilogy(decay, '.-');
    
    grid on; grid minor;
    hold on;
    
    stairs(model);
    pirf=semilogy(irf/max(irf(:))*max(decay(:)), '.k');
    hold off;
    ylabel('Counts [photons]')
    yticki = 10.^(round(log10(1+min([irf(:); decay(:)]))) : ...
        ceil(log10(1+max([decay(:)]))));
    ylim([yticki(1) yticki(end)*2])
    xlim([0 size(decay,1)])
    set(gca, 'YTick', yticki)
    clear leg1 leg2
    leg1=mat2cell(num2str(I2_chi(1,:)'), size(num2str(I2_chi(1,:)'),1), size(num2str(I2_chi(1,:)'),2))';
    l1 = legend([pdec],[leg1], 'Location', 'EastOutside');
    title( l1,'2I*')
    FigStyle
    
    subplot(2,1,2);
    pres=plot(Residuals);
    ylim([min(Residuals(isfinite(Residuals(:)))) max(Residuals(isfinite(Residuals(:))))])
    xlim([0 size(decay,1)])
    grid on; grid minor;
    ylabel('Residuals')
    xlabel('Time [channels]')
    clear leg1
    leg1=mat2cell(num2str(I2_chi(2,:)'),size(num2str(I2_chi(2,:)'),1), size(num2str(I2_chi(2,:)'),2))';
    l1 = legend([pres],[leg1], 'Location', 'EastOutside');
    title(l1,'\chi^2')
    FigStyle
end
end