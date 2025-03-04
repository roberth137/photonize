% %%  Analysis Photonscore files
% %%  File   : PhotonscoreAnalysis.m
% %%  Author : Andre Weber
% %% What it does:
% %% Read irf (instrumental response function) and data,
% %% makes video, ROI by threshold or mouse, gives ROIs Decays, 
% %% fitting decays + single pixelfit for ROI 
% %%%% have fun to play with it ;) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Good luck, André %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate Path %% comment out, if saved paths
clear all; close all
AnalysisFolderName = uigetdir('C:\','set path of PhotonscoreAnalysis')
    addpath(genpath(AnalysisFolderName))
    FigStyle %% stylefile for figures, Please change regarding to your interests
    close (gcf);
%%  IRF
[FileNameIrf,DataPathName, ~] = uigetfile([AnalysisFolderName,'\','*.*'],...
                                   'Find Instrumental Response Function ');
cd (DataPathName)
% %%  Data
[FileName, ~ , ~ ] = uigetfile([DataPathName, '*.*'],...
                                   'Find Data');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do something Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Initial conditions want: 1, not want 0%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim  = 512; scaleF=4096/dim; % Dimension in Pixels
ch   = 11.9;  % channel width in ps (shown in LnTCapture)
dt_irf1=520;    % effective Time window in channels (lower limit of decay)
dt_irf2=3900;   % effective Time window in channels (upper limit of decay)
MakeROI=0;       % If you want to mark an ROI by mouse and klick set to 1
EXPORT_FIGURE=0; % If you want to export png of figures set to 1
Sequence_COUNTS_TAUMEAN=1; % If you want to export TIFF images set to 1
samplingrate = 1; % Sampling rate of Resulting Imagesequences
Sequence_COUNTS_TAUMEAN_Export=1; % save .tiff-stack
SINGLE_PIXEL_FITTING=0; % Depending on PC-Power: CPU 100% !!
                        % for small dimensions or ROIs perfect,
                        % whole image can take a while for 3 exp 
                        % try first with smaller area
EXPORT_WeightedFLIM=1;
EXPORT_WeightedFLIM_ROI=1;
IMAGE_CROPtoIRFcounts=1;                    
%% Read Files 
% %% read Instrumental response function
IRF = AW_readPhotons(FileNameIrf, [1 10^9]) % load events [from to][1 10^9] 
IRF.x = IRF.x/scaleF; IRF.x(IRF.x<1)=1; % rescaling to Dimension dim
IRF.y = IRF.y/scaleF; IRF.y(IRF.y<1)=1; % rescaling to Dimension dim

cps_i=diff(IRF.MSIndex(1:1000:end)); % Counts per second IRF 

    fh_cps_i=figure ('Name','IRF cps_i','NumberTitle','off'); 
    plot(cps_i);
    hold on;
    xlabel('Time [s]')
    ylabel('Counts [Photons /s]')
    title('Intensity of IRF')
    FigStyle;
    
duration_i = length(cps_i);
Decay_Irf = zoo.Hist1D(IRF.dt, 0, 4096, 4096);

    fh_di=figure ('Name','IRF decay','NumberTitle','off'); 
    semilogy(Decay_Irf);
    hold on;
    xlabel('Time [channels]');
    hold on;
    ylabel('Counts')
    title('Decay of IRF')
    FigStyle;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do sometihing Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% focus on part of IRF choose certain part of the measurement by %%%%%%%%%%
%%% time in seconds: aat %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aat1=1;   % Start (time in seconds)
aat2=2000; % End   (time in seconds)
    % %% read partial data
IRF_p = AW_ReadPartByTime(IRF, aat1, aat2);
    figure(fh_cps_i)
    hl_cps_i = plot(aat1:aat2,cps_i(aat1:aat2), 'r')
%     hold off;
    legend(hl_cps_i, 'Chosen part of IRF')
    
Decay_Irf_p = zoo.Hist1D(IRF_p.dt, 0, 4096, 4096);    
    figure(fh_di)
 	hl_di =semilogy(Decay_Irf_p, 'r');    
    legend(hl_di, 'Chosen part of IRF')
%     hold off;
    
%% read Dataset
Data = AW_readPhotons(FileName);
Data.x = Data.x/scaleF; Data.x(Data.x<1)=1; % rescaling to Dimension dim
Data.y = Data.y/scaleF; Data.y(Data.y<1)=1; % rescaling to Dimension dim

cps_d = diff(Data.MSIndex(1:1000:end));
    
    fh_cps_d = figure('Name', 'Data cps_d', 'NumberTitle', 'off');
    plot(cps_d);
    hold on;
    xlabel('Time [s]')
    ylabel('Counts [Photons /s]')
    title('Intensity of Data')
    FigStyle;

Decay_Data = zoo.Hist1D(Data.dt, 0, 4096, 4096);

    fh_dd = figure('Name', 'Data Decay', 'NumberTitle', 'off'); 
    semilogy(Decay_Data);
    hold on;
    xlabel('Time [channels]')
    ylabel('Counts')
    title('Decay of Data')
    FigStyle;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do sometihing Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Choose certain part of the measurement by time in seconds: aat %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aat1 = 479; % Start (time in seconds)
aat2 = 526;%numel(Int_t); % End (time in seconds)
    % %% read partial data
Data_p = AW_ReadPartByTime(Data, aat1, aat2);
Decay_Data_p = zoo.Hist1D(Data_p.dt, 0, 4096, 4096);

    figure(fh_cps_d);
    hl_cps_d = plot(aat1:aat2,cps_d(aat1:aat2), 'r');
    legend(hl_cps_d,'Chosen Part')
%     hold off;
    figure(fh_dd); 
    hl_dd = semilogy(Decay_Data_p);
%     hold off;
    legend(hl_dd, 'Chosen Part')
    
%% Show image Intensity and Mean Lifetime
% For partial IRF 
[counts_i, mean_i] = lnt_mean_tau(dim, ...
    [dt_irf1 dt_irf2], IRF_p.x, IRF_p.y,...
    IRF_p.dt);
tau_mean_irf = mean_i./counts_i*ch;
    % figure;
    % imagesc(tau_mean_irf, [nanmean(tau_mean_irf(:))*0.99  nanmean(tau_mean_irf(:))*1.1 ]); colorbar;
mask_m0 = counts_i>mean(counts_i(counts_i>0)*0.5);
    % imagesc(mask_m0)
% For partial data  
[counts_d, mean_d] = lnt_mean_tau(dim, ...
    [dt_irf1 dt_irf2], Data_p.x, Data_p.y,...
    Data_p.dt);
tau_mean_data=mean_d./counts_d*ch;
tau_mean_data(isnan(tau_mean_data))=0;

% %% imagesc(tau_mean_data, [nanmean(tau_mean_data(:))*0.99  nanmean(tau_mean_data(:))*1.1 ]); colorbar;
FLIM=tau_mean_data-tau_mean_irf;
FLIM=double(FLIM.*mask_m0);
FLIM(FLIM<0)=0;
img=double(counts_d.*mask_m0);

    figure('Name', 'Counts mean-Lifetime ', 'NumberTitle', 'off');
    hs=subplot(121);
    imagesc(img,[min(flat(img(mask_m0))) max(flat(img(mask_m0)))]);
    axis square; cbar=colorbar;
    zlab = get(cbar,'xlabel'); set(zlab,'String','counts', 'FontSize',35);
     FigStyle;  grid off
    set(gca, 'FontSize', 10)
   
    
    subplot(122)
    imagesc(FLIM, [0 5*1000]);
    axis square;  cbar=colorbar;
    zlab = get(cbar,'xlabel'); set(zlab,'String','\tau [ps]', 'FontSize',35);
    % title('Overview', 'Position' [gca(hs])
     FigStyle;  grid off;
     set(gca, 'FontSize', 10);
     
     if EXPORT_FIGURE
      exportFigure([strtok(FileName,'_'),'IMGintFLIM','.png'], 300)
     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % Video / imagesequence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Sequence_COUNTS_TAUMEAN
    [Counts_t, FLIM_t] = AW_ImageSequences(...
        IRF, Data, aat1, aat2,...
        samplingrate, dim, dt_irf1, dt_irf2, ch);
end
if Sequence_COUNTS_TAUMEAN_Export
    
    for i = 1:size(Counts_t,3)
        tmp=uint16(Counts_t(:,:,i));
        imwrite(tmp, [strtok(FileName,'.'), '_Counts_t_T_',num2str(samplingrate),'s_',...
            num2str(aat1),'s-',num2str(aat2),'s.tiff'],...
            'writemode', 'append')
        
        tmp2=uint16(FLIM_t(:,:,i));
        imwrite(tmp2, [strtok(FileName,'.'), '_FLIM_t_ps_T_',num2str(samplingrate),'s_',...
            num2str(aat1),'s-',num2str(aat2),'s.tiff'],...
            'writemode', 'append')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do sometihing Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% can change treshold for Masking interesting part or use ROI %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%   figure(3);
threshold = 200;% %  in photons
    mask_img = img>threshold; % set automatic mask by threshold
    
    tau_mean_range = round([min(flat(FLIM(mask_img)))...
        max(flat(FLIM(mask_img)))+1]/1000, 0);
    figure ('Name', 'Thresholded Lifetimes', 'NumberTitle', 'off');
    imagesc (FLIM .*mask_img/1000, [tau_mean_range]);
    colorbar
    axis square
    FigStyle
   
%     imagesc(FLIM/1000,tau_mean_range); colorbar
%     FigStyle; grid off
    IntTau(img, FLIM/1000, [1 max(img(:))/10], tau_mean_range,0 )
    title('Linear Intensity')
    if EXPORT_FIGURE
        exportFigure([strtok(FileName,'_'),'_IntTauLin','.png'], 300)
    end
    IntTau(log10(img), FLIM/1000, [1 max(img(:))/10], tau_mean_range,0 )
    title('Logarithmic Intensity')
    if EXPORT_FIGURE
        exportFigure([strtok(FileName,'_'),'_IntTauLog','.png'], 300)
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do sometihing Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Region of interest ROI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% loop for multiple ROIs over number i.e. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% for number = 1:3 .... decaysFromMask... end %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        number=1; %% number of ROIS use loop for multiple 
        if MakeROI
    %         figure(11)
    %         imagesc(img, [10 max(img(:))])
    %         axis square
            title('mark ROI, then doubleklick')
            h_im = imfreehand(gca);
            position = wait(h_im);
            mask_img(:,:,number) = logical(createMask(h_im));
        end
    % Getting Decays from Sample and IRF
         [DataDecayMask(:,number)]=decaysFromMask(Data_p,dim,...
                                         squeeze(mask_img(:,:,number)))';
         [IRFdecayMask(:,number)]=decaysFromMask(IRF_p,dim,...
                                         squeeze(mask_img(:,:,number)))';
    %    
    figure ('Name', 'Think about range to cut Decay in channels');
        semilogy(DataDecayMask); hold on;
        semilogy(IRFdecayMask); hold on;
        legend('Decay data' , 'Decay IRF')
        ylabel('Counts')
        xlabel('Time Channels')
        FigStyle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do sometihing Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%% Fitting Log likelihood  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%% initial conditions for EM Fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    range_tac=550:2900; % cut out Decay by channels: range_tac (from:to)

    decay = DataDecayMask(range_tac); % Decays to Fit
    irf   = IRFdecayMask(range_tac);  % Decays to Fit

    tau_ref=0; %% for Reference Dye with known lifetime for IRF
    tau = [0.4 1.2 3.2]'/ch*1000; % Lifetime i.e. 3 exp Lifetime
    a = [ones(size(tau))]' * 100; % Contribution i.e. 3 exp Lifetime
                                  % a = All photons (area under decay)
    bg = 0.1;    % background
    irf_shift=-43.9 % missmatch IRF data in ch

        clear x0 lo up % initial fitting parameters and limits
        x0.tau = tau;
        lo.tau = x0.tau * 0.1;
        up.tau = x0.tau * 1000;

        x0.a = a';
        lo.a = x0.a * 0;
        up.a = x0.a * 1000000000;

        x0.irf_shift =irf_shift;
        lo.irf_shift =  x0.irf_shift-10;
        up.irf_shift = x0.irf_shift+ 10;
        %
        x0.background = bg;
        lo.background = 0;
        up.background = 1e9;

        x0.tau_ref = tau_ref;
    %         lo.tau_ref = 0;
    %         up.tau_ref = 10;

    FitResults = zoo.FitDecay(irf, decay, x0, lo, up);
    %         FitModel = zeros(size(decay));
    %         I2_chi = zeros(2, length(FitResults.tau_ref));
    [FitModel, I2_chi, Residuals]   = AW_ZooResiduals(irf, decay, FitResults, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% singlePixelFIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% can take a while for many components!!! CPU 100% %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SINGLE_PIXEL_FITTING   
    Decay_Data_dim = AW_3DtacPixel(Data_p,dim,mask_img);
    Decay_Irf_dim = AW_3DtacPixel(IRF_p,dim,mask_img);
    % set Initial conditions
    clear tau
    tau = FitResults.tau; % takes Taus from Decayfit
    a = [ones(size(tau))]' * 100;
    bg = 0.1;

        clear x0 lo up FitResults_dim
        x0.tau = tau;
                    lo.tau = x0.tau * 0.1; %% comment out for fixing values
                    up.tau = x0.tau * 1000;%% comment out for fixing values
        x0.a = a';
                    lo.a = x0.a * 0;%% comment out for fixing values
                    up.a = x0.a * 1000000000;%% comment out for fixing values

        x0.irf_shift =FitResults.irf_shift;
        x0.background =0;
        x0.tau_ref = 0;


    % Fit
    FitResults_dim = zoo.FitDecay(Decay_Irf_dim, Decay_Data_dim, x0, lo, up);
    clear FitAlphas_dim
    FitAlphas_dim=AW_ReshapeMaskDim(FitResults_dim.a, mask_img, dim ,1:size(tau,1));
    %% Figure Alphas
    limits_ROI(1,:) = ceil(min(position))-5
    limits_ROI(2,:) = ceil(max(position))+5
    
        fh_a = figure('Name',...
'Single Pixel Fitting: Contributions Alpha for fixed Lifetimes of ROIfit', ...
                      'NumberTitle', 'off');
        h_sb=subplot(1,size(tau,1),1);

        for i_sub = 1:size(tau,1)
            subplot(1,size(tau,1),i_sub)
            imagesc(FitAlphas_dim(limits_ROI(1,2):limits_ROI(2,2),...
                                limits_ROI(1,1):limits_ROI(2,1) ,i_sub),...
                       [nanmin(FitAlphas_dim(:)) nanmax(FitAlphas_dim(:))])
            axis square; set(gca,'XTickLabel', [],'YTickLabel', [])
            title(['\tau_',num2str(i_sub),'=',...
                    num2str(FitResults_dim.tau(i_sub,1)*ch/1000),'ns'])
        end
        cbar=colorbar;
        cbar.Location='north';
        cbar.Position=[ .35 h_sb.Position(2) .3 .04];
        cbar.AxisLocationMode;
        cbar.Label.String= 'counts';
        cbar.Label.FontSize= 25;
        cbar.FontSize=20;
end