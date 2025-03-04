%% TE:
%   neue plots:
%   - vernü+nftiger spektrumplot angelehnt an AW_spectraPlot
%   - Vergleich: Spektrum normalized model und model mit INF pixel
%   - Fläche unter d und Fläche unter Spektrum(tau)
%% AW_Anpassung an 100 savedIterations
% residues/sqrt(decay)
% axis
% austausch suffix Inf / norm
 close all;
savedSteps=100;
%% plot settings
% style = load('E:\Users\Public\MATLAB\MEMLifetime\Server\Hefe\With_IRF_Shift_Compensation\hgexport_style.mat');
% style = style.style;
style = load('D:\MATLAB\hgexport_style.mat');
style = style.style;
% colors
plotProp = struct();
plotProp.fewColors = hsv(size(d,2)*2);
plotProp.manyColors = jet(size(cell_x{1},3));
% plotProp.manyColors = plotProp.manyColors(1:size(x,3),:);

plotsDir = [pwd, '\plots\','chan',num2str(channels),'_mu',num2str(mu),'_sigma',num2str(sigma)];

savePlots = true;
if savePlots
    set(0, 'defaultFigurePosition',[50 0 1175 800]);
else
    set(0, 'defaultFigurePosition',[50 50 640 480])
end

cellID = 2; % 1=INF pixel model ; 2=old model

if (cellID == 2)
    
     suffix = 'normModel';
else
    suffix = 'INFpixel';
end

%% main plot (decay, residues, spectrum)
for runID = 1:length(sim)
    figure(1);
    subplot(3,1,1);
    semilogy(d(:,runID), '.b')
    hold on;
    plot(mo*cell_x{cellID}(:,runID,round(max(iterations/savedSteps+10))), 'r')
    axis([0 size(d,1) 1 max(d(:,runID))])
    title({suffix,['\tau = ',num2str(sim(runID).tau),' | counts = ',num2str(sim(runID).N)]})
    xlabel('channels')
    ylabel('counts')
    legend('Decay', 'Fit')

    subplot(3,1,2); hold on;
    title('residues')
    plot((d(:,runID)-mo*cell_x{cellID}(:,runID,round(max(iterations/savedSteps))+10))./sqrt(d(:,runID)))
%     axis([0 size(d,1) -10 10])
    xlim ([0 size(d,1)])
    xlabel('\tau [ch]')
    ylabel('residues')
    legend(['\chi² = ',num2str(cell_chi2{cellID}(end-2,runID))])

    subplot(3,1,3); hold on;
    plotSpectrumAWTE(tau_range, cell_x{cellID}(1:size(tau_range,1),runID,round(max(iterations/savedSteps))+10));
    title(['iterations = ',num2str(iterations(end)), ' | \tau = ',num2str(sim(runID).tau),' | normcounts = ',num2str(100*sim(runID).N/sum(sim(runID).N)), '%'])
   
    if savePlots
        savefilename = [num2str(runID),'_tau',num2str(sim(runID).tau),'_N',num2str(sim(runID).N)];
        saveas(gcf, [plotsDir,'fit_resid_contributions_',savefilename,'_',suffix,'.fig'],'fig');
        hgexport(gcf, [plotsDir,'fit_resid_contributions_',savefilename,'_',suffix,'.png'], style, 'Format', 'png');
    end
    close (figure(1))
end

%% likelihood, chi2, 
%% Berechne lambda
for i=1:size(cell_x{1},3)
for kl = 1:size(d,2)
    [L(i,kl) E(i,kl)] = AnalysisAWTE.klkkt(mo, cell_x{cellID}(:,kl, i), d(:,kl));
      %      stoppingRule = L - lambda*E
      %    (when stoppingRule->0)
end
end
lambda = L ./ E;
%% Estimate Taus
clear tau_est
for runID=1:size(d,2)
[tau_est{cellID, runID}] = AnalysisAWTE.estimateTau(cell_x{cellID}(1:size(tau_range,1),runID,round(max(iterations/savedSteps))+10), tau_range);
end
%%
for runID =1:length(sim)
    figure (1);
    subplot(3,1,1); hold off;
    loglog(cell_Li{cellID}(:,runID)); hold on;
    title(['\tau = ',num2str(sim(runID).tau),' | counts = ',num2str(sim(runID).N)])
    xlabel('number of iterations')
    ylabel('Likelihood function')
    legend(['Decay #', num2str(runID)])
    subplot(3,1,2); 
    title(['\chi²'])
    plot(cell_chi2{cellID}(:,runID))
    hold on;
    axis([0 size(cell_chi2{cellID},1)-1 min(cell_chi2{cellID}(1:end-2,runID)) max(cell_chi2{cellID}(1000:end-2,runID))])
    xlabel(['number of iterations'])
    ylabel('\chi²')

    subplot(3,1,3); 
    semilogy(lambda(:,runID))
    hold on;
    xlabel(['number of iterations / ', num2str(savedSteps)])
    ylabel('\lambda')
    xlim([0 size(lambda,1)])
    
    if savePlots
        savefilename = [num2str(runID),'_tau',num2str(sim(runID).tau),'_N',num2str(sim(runID).N)];
        saveas(gcf, [plotsDir,'likeli_chi2_lambda_',savefilename,'_',suffix,'.fig'],'fig');
        hgexport(gcf, [plotsDir,'likeli_chi2_lambda_',savefilename,'_',suffix,'.png'], style, 'Format', 'png');
    end
   close(figure (1))
end

%% tau_sim - tau_est
for runID = 1:length(sim)
    figure;
%     subplot(211);hold on;
    title([{'\tau_{sim} = ',num2str(sim(runID).tau),' | counts_{sim} = ',num2str(sim(runID).N)}])
%     it = [1:size(cell_x{cellID},3)];
%     legendStr = {};
% %     plot([1 it(end)], [tau(:,runID)' ; tau(:,runID)'], '-.')
%     taus = zeros(2,size(it,2));
%     for i = 1:size(it,2)
%         s = AnalysisAWTE.estimateTau(cell_x{cellID}(1:size(tau_range,1),runID,it(i)), tau_range);
%         if (length(s) == size(sim(runID).tau,2))
%             for tt = 1:length(s)
%                 taus(tt,i) = s(tt).byBarycenter;
%                 plot(it(i)*diff(iterations(1:2)), taus(tt,i)-sim(runID).tau(tt),'.', 'Color', plotProp.fewColors(tt*3,:))
%                 legendStr(tt) = {['\tau_{sim,',num2str(tt),'} = ',num2str(sim(runID).tau(tt))]};
%             end
%         end
%     end
%     legend(legendStr)
%     xlabel('number of iterations')
%     ylabel('\tau - \tau_{sim}')
%      subplot(212);
     j=1; clear leg_it;
     plotSpectrumAWTE(tau_range, cell_x{cellID}(1:size(tau_range,1),runID,round(max(iterations/savedSteps))+10));
     hold on;
     for i=1:1000:round(max(iterations/savedSteps))+10
     plot(tau_range, squeeze(cell_x{cellID}...
         (1:size(tau_range,1),runID,i)), ...
         'Color', plotProp.manyColors(i,:))
          hold on;
     leg_it{j}= ['Iterations ', num2str(i)]
     j=j+1
     end
     legend(leg_it)
  
   
%      set(gca, 'XScale','log')
if savePlots
    
    savefilename = [num2str(runID),'_tau',num2str(sim(runID).tau),'_N',num2str(sim(runID).N)];
    saveas(gcf, [plotsDir,'diff_tau_tausim',savefilename,'_',suffix,'.fig'],'fig');
    hgexport(gcf, [plotsDir,'diff_tau_tausim',savefilename,'_',suffix,'.png'], style, 'Format', 'png');
end
end
% 
% %% area(d), area(spectrum)
% % figure; hold on;
% % for nn = 1:size(N,2)
% %     subplot(size(N,2),1,nn); hold on;
% %     title(['N = ',num2str(N(nn))])
% % 
% %     legendStr = {};
% %     tauLegend = zeros(max(numberOfTaus(:,nn)), 1);
% %     area_d = sum(d(:,nn));
% %     tauCount = find(numberOfTaus(:,nn));
% %     for tt = 0:max(numberOfTaus(:,nn))-1
% %         if (tt+1 <= size(tau,1))
% %             y = find( taus(tt*7+6,nn,tauCount) );
% %             area_spec = squeeze(taus(tt*7+6,nn,tauCount(y)));
% %             plot(y, area_spec/area_d, '.');
% %             legendStr(tt+1) = {['\tau_{sim} = ',num2str(tau(tt+1))]};
% %         end
% %     end
% % %     plot(y([1 end]), [area_d/size(tau,1) area_d/size(tau,1)], '-.k');
% %     legendStr(end+1) = {'sim decay'};
% %     xlabel('number of iterations')
% %     ylabel('area')
% %     legend(legendStr)
% % end
% %% 
% % if savePlots
% %     saveas(gcf, [plotsDir,'Adecay_Aspec',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'Adecay_Aspec',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% % tau over number of interations steps
% % by maximum of peak in spectrum
% figure; hold on;
% title({'estimated \tau values by maximum of peak'})
% legendStr = {};
% 
% for i = 1:size(tau,1)
%     legendStr(i) = {['\tau_{sim ',num2str(i),'} = ' num2str(tau(i))]};
% end
% plot(iterations([1:end])', ones(size(iterations,2),1)*tau', '-.k')
% for nn = 1:size(N,2)
%     legendStr(nn+length(tau)) = {['N = ',num2str(N(nn))]};
%     for tt = 0:size(tau,1)-1
%         fin = find(taus(tt*7+1,nn,:));
%         plot(iterations(fin), squeeze(taus(tt*7+1,nn,fin)), '.', 'Color', plotProp.fewColors(nn,:));
%     end
% end
% legend(legendStr)
% xlabel('iteration steps')
% ylabel('\tau')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'tau_maxpeak',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'tau_maxpeak',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% by barycenter
% figure; hold on;
% title({'estimated \tau values by barycenter'})
% legendStr = {};
% for i = 1:size(tau,1)
%     legendStr(i) = {['\tau_{sim ',num2str(i),'} = ' num2str(tau(i))]};
% end
% plot(iterations([1:end])', ones(size(iterations,2),1)*tau', '-.k')
% for nn = 1:size(N,2)
%     legendStr(nn+size(tau,1)) = {['N = ',num2str(N(nn))]};
%     for tt = 0:size(tau,1)-1
%         fin = find(taus(tt*7+2,nn,:));
%         plot(iterations(fin), squeeze(taus(tt*7+2,nn,fin)), '.', 'Color', plotProp.fewColors(nn,:));
%     end
% end
% xlabel('iteration steps')
% ylabel('\tau')
% legend(legendStr)
% 
% if savePlots
%     saveas(gcf, [plotsDir,'tau_bary',suffix,'.fig'], 'fig');
%     hgexport(gcf, [plotsDir,'tau_bary',suffix,'.png'], style, 'Format', 'png');
% end
% 
% %% tau values and variance for each N over iter
% % figure; hold on;
% % title({'estimated \tau values (barycenter)'})
% % legendStr = {};
% % for i = 1:size(tau,1)
% %     legendStr(i) = {['\tau_{sim ',num2str(i),'}= ' num2str(tau(i))]};
% % end
% % plot(iterations([1:end])', ones(size(iterations,2),1)*tau', '-.k')
% % 
% % % taus
% % for nn = 1:size(N,2)
% %     legendStr(nn+size(tau,1)) = {['N = ',num2str(N(nn))]};
% %     for tt = 0:size(tau,1)-1
% %         fin_nonzero = find(taus(tt*7+2,nn,:));
% %         fin_var = find( abs(taus(tt*7+2,nn,fin_nonzero)-tau(tt+1))<=varMaus(tt+1,nn) );
% %         plot(iterations(fin_nonzero(fin_var)), squeeze(taus(tt*7+2,nn,fin_nonzero(fin_var))), 'o', 'Color', plotProp.fewColors(nn,:));
% %         plot(iterations(fin_nonzero), squeeze(taus(tt*7+2,nn,fin_nonzero)), '.', 'Color', plotProp.fewColors(nn,:));
% %     end
% % end
% % 
% % % variance
% % for nn = 1:size(N,2)
% % %     legendStr(nn+size(tau,1)+size(N,2)) = {['variance for N = ',num2str(N(nn))]};
% %     plot([1 size(iterations,2)], [tau+varMaus(:,nn) tau+varMaus(:,nn)]', '-.', 'Color', plotProp.fewColors(nn,:))
% %     plot([1 size(iterations,2)], [tau-varMaus(:,nn) tau-varMaus(:,nn)]', '-.', 'Color', plotProp.fewColors(nn,:))
% % end
% % 
% % xlabel('iteration steps')
% % ylabel('\tau')
% % legend(legendStr)
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'var',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'var',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% variance of decay and fit
% figure;
% tt = 1;
% for nn = 1:size(N,2)
%     legendStr = {};
%     subplot(size(tau,1),1,tt); hold on;
%     title({['\tau = ',tau(tt)],['N = ',num2str(N(nn))]})
%     plot([1 size(iterations,2)], [varHall(tt,nn) varHall(tt,nn)]', '-.', 'Color', plotProp.fewColors(nn,:))
%     legendStr(1) = {['var_{\tau = ',num2str(tau(1)),'}']};
%     for it = 1:size(iterations,2)
%         varFit = AnalysisAWTE.varTau_Hall(d(:,nn), taus(tt*7+2,nn,it), channels, ch_width);
%         plot(it, varFit, '.', 'Color', plotProp.fewColors(nn,:))
%         legendStr(1+nn) = {['N = ',num2str(N(nn))]};
%     end
% end
% 
% for tt = 1:size(tau,1)
% %     plot([1 size(iterations,2)], [varMaus(tt,nn) varMaus(tt,nn)]', '-.', 'Color', plotProp.fewColors(tt,:))
% %     plot([1 size(iterations,2)], [-varMaus(tt,nn) -varMaus(tt,nn)]', '-.', 'Color', plotProp.fewColors(tt,:))
% end
% 
% %% data, barycenter, maximum
% figure;
% title({'\tau from','* = maximum', '-- = barycenter'})
% % for i = 1:size(tau,1)
% %     legendStr(i) = {['\tau_{sim ',num2str(i),'} = ' num2str(tau(i))]};
% % end
% % plot(iterations([1:end])', ones(size(iterations,2),1)*tau', '-.k')
% legendStr = {};
% for tt = 0:size(tau,1)-1
%     subplot(size(tau,1),1,tt+1); hold on;
%     title({'\tau calculated from','* = maximum', '-- = barycenter', ['\tau = ', num2str(tau(tt+1))]})
%     for nn = 1:size(N,2)
%         legendStr(nn) = {['N = ',num2str(N(nn))]};
%     
%         fin = find(taus(tt*7+1,nn,:));
%         plot(iterations(fin), tau(tt+1)-squeeze(taus(tt*7+1,nn,fin)), '.', 'Color', plotProp.fewColors(nn,:));
%     end
% 
%     for nn = 1:size(N,2)
%         fin = find(taus(tt*7+2,nn,:));
%         plot(iterations(fin), tau(tt+1)-squeeze(taus(tt*7+2,nn,fin)), '--', 'Color', plotProp.fewColors(nn,:));
%     end
%     
%     xlabel('iteration steps')
%     ylabel(['(\tau_{',num2str(tau(tt+1)),'} - \tau)'])
%     legend(legendStr)
% end
% 
% if savePlots
%     saveas(gcf, [plotsDir,'tau_maxi_bary',suffix,'.fig'], 'fig');
%     hgexport(gcf, [plotsDir,'tau_maxi_bary',suffix,'.png'], style, 'Format', 'png');
% end
% 
% %%
% figure;
% for tt = 1:size(tau,1)
%     subplot(size(tau,1), 1, tt); hold on;
%     legendStr = {};
%     title(['\tau = ', num2str(tau(tt))])
%     for nn = 1:size(N,2)
%         plot([tau(tt)-varMaus(tt,nn) tau(tt)+varMaus(tt,nn)], [N(nn) N(nn)], '*', 'Color', plotProp.fewColors(nn,:));
%         legendStr(nn) = {['N = ',num2str(N(nn))]};
%     end
%     ylabel('variance')
%     legend(legendStr)
% end
% xlabel('[channels]')
% 
% if savePlots
%     saveas(gcf, [plotsDir,'var',suffix,'.fig'], 'fig');
%     hgexport(gcf, [plotsDir,'var',suffix,'.png'], style, 'Format', 'png');
% end
% 
% %%
% % f1 = figure; hold on;
% % ax1 = get(f1, 'CurrentAxes');
% % colors = [1 0 0;...
% %           0 0.5 0;...
% %           0 0 1;...
% %           0 0 0;...
% %           1 0 1;...
% %           0.5 1 0.3;...
% %           1 0.5 0.5];
% % 
% % for i = 1:size(colors,1)
% %     d = poissrnd(d_clean * N);
% %     [x, Li, lambda, it_sol] = EMcore(d, mo, mo_inf, x0, iterations(end), stoppingCrit);
% %     plot(ax1, N, lambda(end,:), 'Color', colors(i,:), 'Marker', 'o', 'LineStyle', 'none')
% %     
% %     % fit
% %     P = polyfit(N, lambda(end,:), 1);
% %     yfit = P(1)*N + P(2);
% %     hold on;
% %     plot(N, yfit, '-.');
% % end
% % title(ax1, {['\lambda scatter plot for ', num2str(size(colors,1)), ' noisy decays'], ['after ',size(iterations,2),' iteration steps']})
% % xlabel(ax1, 'photon counts N')
% % ylabel(ax1, '\lambda')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'noisyDecays_N_lambda',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'noisyDecays_N_lambda',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% used tau models
% % figure; hold on;
% % title('ranges of \tau used in normal and detailed model')
% % plot(tau_range);
% % plot(1:size(tau_range,1), ones(size(tau_range,1),1)*tau' ,'-.k');
% % legend('range of \tau-model', '\tau_{data}')
% % xlabel('\tau-range')
% % ylabel('\tau')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'tau',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'tau',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% distribution of tau in normal and detailed model
% % figure; hold on;
% % title({'distribution of \tau', ['after ',num2str(size(iterations,2)), ' iteration steps']})
% % plot(N, taus(:,:,end), 'or');
% % plot(N, ones(size(N,2),1)*tau', '-.k')
% % xlabel('photon counts N')
% % ylabel('\tau')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'tau_dist',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'tau_dist',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% tau, lambda over N
% % figure;
% % subplot(3,1,1); hold on;
% % plot(N, taus(1,:,end), 'or');
% % plot(N, ones(size(N))*tau, '-.k')
% % xlabel('photon counts N')
% % ylabel('\tau')
% % legend('\tau', '\tau_{data}')
% % 
% % subplot(3,1,2); hold on;
% % plot(N, lambda(end,:)', 'or')
% % xlabel('photon counts N')
% % ylabel('\lambda')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'functionsOfN',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'functionsOfN',suffix,'.png'], style, 'Format', 'png');
% % end
% %%
% % f = figure; hold on;
% % title(['\lambda after ', num2str(size(iterations,2)), ' iteration steps for different photon counts'])
% % plot(N, lambda(end,:)', 'or')
% % lsline
% % xlabel('photon counts N')
% % ylabel('\lambda')
% % legend('\lambda', 'regression')
% % 
% % if savePlots
% %     saveas(f, [plotsDir,'N_lambda',suffix,'.fig'], 'fig');
% %     hgexport(f, [plotsDir,'N_lambda',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% estimated tau after it_sol iterations for each N
% % figure; hold on;
% % title(['\tau_{est} after n-interations when \lambda hits ',num2str(1/sqrt(size(tau,1)^2))])
% % plot([1 iterations(end)], ones(size(tau,1),1)*tau', '-.k', 'LineWidth', 2)
% % for nn = 1:size(N,2)
% %     plot(it_sol(nn), taus(:,nn,it_sol(nn)), 'or');
% %     text(it_sol(nn)*ones(1,size(tau,1)), taus(:,nn,it_sol(nn)), ['   N=',num2str(N(nn))])
% % end
% % xlabel('number of iterations n');
% % ylabel('\tau');
% % legend('\tau_{data}')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'n_tauSol',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'n_tauSol',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% model, fit
% % figure; hold on;
% % title('fit after n-iterations when \lambda=1')
% % for nn = 1:size(N,2)
% %     plot(d(:,nn), '-.')
% %     plot(mo*x(:,nn,it_sol(nn)))
% % end
% % xlabel('Lifetimes \tau')
% % ylabel('counts')
% % legend('data', 'fit')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'fit',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'fit',suffix,'.png'], style, 'Format', 'png');
% % end
% 
% %% lambda over N for different amount of iteration steps
% % figure; hold on;
% % title('\lambda after different number of iteration steps n (mono/bi exp)')
% % for tt1 = 1:size(tau,1)
% %     
% %     % create decay, model and run EM-algorithm
% %     d = zeros(channels, size(N,2));
% %     for i = 1:length(N)
% %         for tt = tt1:tt1
% %             d_clean = GDecayK(tau(tt), mu, sigma, 1:channels);
% %             d(:,i) = d(:,i) + poissrnd(d_clean * N(i))';
% %         end
% %     end
% % 
% %     firstLambda = [ lambda(150,size(N,2)) lambda(200,size(N,2)) lambda(300,size(N,2)) ];
% %     
% %     [x, Li, lambda, it_sol] = EMcore(d, mo, mo_inf, x0, 300, stoppingCrit);
% %         
% % %     [lambda(150,size(N,2)) lambda(200,size(N,2)) lambda(300,size(N,2))]-firstLambda
% %     
% %     % plot
% %     plot(N, lambda(150,:), 'or')
% %     plot(N, lambda(200,:), 'ob')
% %     plot(N, lambda(300,:), 'ok')
% % end
% % lsline
% % xlabel('photon counts N')
% % ylabel('\lambda')
% % legend('n = 150', 'n = 200', 'n = 300')
% % 
% % if savePlots
% %     saveas(gcf, [plotsDir,'lambda_diffIter',suffix,'.fig'], 'fig');
% %     hgexport(gcf, [plotsDir,'lambda_diffIter',suffix,'.png'], style, 'Format', 'png');
% % end


