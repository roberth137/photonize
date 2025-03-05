%% timedependent ROI overview:
%%% MEDIAN-2D filtered Tau_mean-Image
%%% Intensity Image
%%% Tau_Mean histogram of pixels
% %%% Exponential Fitting plot of Decay
function AW_OverviewROILifetimes(ROI_t_50, ROI_t_50_nr, Legend_TIME, ...
    DataDecayMask_t, FitResults_t_ROI_nr, ...
    FLIM_Med2D, FLIM_hist, Counts_t,...
    tau_mean_range, BereichTau, range_tac, ch, ...
    FileName, SaveFolderName,EXPORT_FIGURE)

Fig_Width=get (0,'screensize');

% Legend_TIME = [samplingrate: samplingrate: samplingrate*size(Counts_t,3)];
figure ('Name', 'IntFLIM, HistRoi, DecayROI')

LIM_ROI_X =[find(squeeze(sum(sum(ROI_t_50,2),3))>0)];
LIM_ROI_Y =[find(squeeze(sum(sum(ROI_t_50,1),3))>0)];
FZ=25; % FontSize Legends
FZa=12;% FontSize Axis (plots)
if EXPORT_FIGURE
    SavePathFolder = [SaveFolderName,FileName,...
        '_Overview_TimedependentROI_',num2str(ROI_t_50_nr),'\'];
    mkdir(SavePathFolder)
    
end
for i=1:size(Counts_t,3)
    %%
    subplot(2,2,1)
    h=IntTauLab(log10(Counts_t(LIM_ROI_X,LIM_ROI_Y,i)).*...
        ROI_t_50(LIM_ROI_X,LIM_ROI_Y,i),...%.*mask_img(LIM_ROI_X,LIM_ROI_Y,ROI_nr), ...
        FLIM_Med2D(LIM_ROI_X,LIM_ROI_Y,i)/1000,...
        [0 nanmax(flat(Counts_t(...
        LIM_ROI_X,LIM_ROI_Y,:)))*0.1],...
        tau_mean_range, Legend_TIME(i) , '\tau [ns]');
    grid off;% set(gcf, 'Position',[50 50 round(Fig_Width(3)/1.33) round(900/1.35)])
    set(gca, 'FontSize',FZa);
    
    subplot(2,2,3)
    h = imagesc(Counts_t(LIM_ROI_X,LIM_ROI_Y,i).*...
        ROI_t_50(LIM_ROI_X,LIM_ROI_Y,i),...
        [0 nanmax(flat(Counts_t(LIM_ROI_X,LIM_ROI_Y,:)))]);
    axis image
    colormap jet;
    tmp_img=double(Counts_t(LIM_ROI_X,LIM_ROI_Y,i).*ROI_t_50(LIM_ROI_X,LIM_ROI_Y,i));
    tmp_img(tmp_img< 0) = 0;...
        tmp_img(tmp_img > 1) = 1;
    tmp_img = rescaleData(tmp_img, [0 1]);
    set(h,'AlphaData',tmp_img);
    set(gca,'Color',[0,0,0]);
    FigStyle;
    set(gca, 'XTick', [],'YTick', []); clear tmp_img;
    axis image; grid off;
    cbar=colorbar;
    zlab = get(cbar,'xlabel'); set(zlab,'String','photons', 'FontSize',FZ);
    set(cbar, 'FontSize', FZa)
    text(0,-130/2, FileName, 'FontSize', FZ)
    %         text(0,-50/2, Stim_Lab{i}, 'FontSize', FZ)
    
    h_sb_H=subplot(2,2,2);
    maxNum=max(flat(FLIM_hist(:,:,ROI_t_50_nr)));
    plot(BereichTau, FLIM_hist(:,i,ROI_t_50_nr), 'LineWidth', 2)
    FigStyle;
    set(gca,  'FontSize',FZa)
    legend ('\tau-Histogram')
    xlabel('\tau_{mean} [ps]', 'FontSize', FZ)
    title(['ROI ' ,num2str(ROI_t_50_nr) ])
    ylabel(' # pixel ', 'FontSize', FZ)
    ylim([0 maxNum])
    xlim([BereichTau(1) BereichTau(end)])
%     set(gcf, 'Position',[50 50 Fig_Width 900])
    %                     axes('Position',[h_sb_H.Position*0.5])
    % %                     box on
    
    subplot(2,2,4)
    semilogy(range_tac*ch/1000,  DataDecayMask_t(range_tac,i, ...
        ROI_t_50_nr));%/sum(DataDecayMask_t(range_tac,i, ...   ROI_nr))
    
    if ~isempty(FitResults_t_ROI_nr)
        
        hold on;
        plot(range_tac*ch/1000,FitResults_t_ROI_nr{i, ROI_t_50_nr}.FitModel);
        title(['Decay of Data [\tau_{mean} = ',...
            num2str(FitResults_t_ROI_nr{i, ROI_t_50_nr}.TauMean),' ns]'])
        text([(max(range_tac*ch/1000)-min(range_tac*ch/1000))/3*2.2],...
        nanmax(DataDecayMask_t(:,i,ROI_t_50_nr))*0.1,...
        FitResults_t_ROI_nr{i, ROI_t_50_nr}.FitResStr', 'FontSize', FZa);
    end
    
    FigStyle; set(gca, 'FontSize', FZa)
    xlabel('Time [ns]', 'FontSize', FZ)
    ylabel(['Intensity'],'FontSize', FZ)
    xlim([min(range_tac*ch/1000) max(range_tac*ch/1000)])
    ylim([1 max(DataDecayMask_t(:))])
    set(gcf, 'Position',[50 50 round(Fig_Width(3)/1.33) round(Fig_Width(4)/1.35)])
    
    hold off;
    
    if EXPORT_FIGURE
        exportFigure([SavePathFolder,'\', FileName,...
            '_IntTauLog_', num2str(Legend_TIME(i)), 'sTau',...
            num2str(tau_mean_range(1)),'-',num2str(tau_mean_range(2)),'ns_Timedepend',...
            '_ROI_t_50',num2str(ROI_t_50_nr),'.png'])
    end
end
end