function [DataDecayMask, IRFDecayMask, mask_img, limits_ROI]=...
    AW_UI_ButtonAddROI(Data_pSort, IRF_pSort,SaveResultsPath,SaveNAMEfile)
%% [DataDecayMask, IRFDecayMask, mask_img, limits_ROI]=...
%%     AW_UI_ButtonAddROI(Data_pSort, IRF_pSort,SaveResultsPath,SaveNAMEfile)
%% Makes Figure with Buttons to select ROIS from Intensity image. 
%% Output: - Decays of ROI for IRF and Data
%%         - Binary ROIs (mask_img) and limits
%%         - Saves Results in SaveFolder

clear h_ROI
fh_ROI= figure('Name', 'Mark ROI', 'NumberTitle', 'off', 'Visible','off');

subplot(2,2,[1,3])
imagesc(Data_pSort.image)
axis image; FigStyle;
subplot(222)
%         semilogy(DataDecayMask);
xlim([500 4000]);
xlabel('Time [channels]'); ylabel('Counts'); FigStyle;
title('Decay of ROI')
subplot(224)
%         semilogy(IRFDecayMask,'.');
title('IRF of ROI')
xlim([500 4000]);
ylim([0 1])
xlabel('Time [channels]'); ylabel('Counts'); FigStyle;

ROI_nr=1;
mask_ROI=false(size(Data_pSort.image));
Stop_AW_AddROI = true;
j=1;
DataDecayMask=[];
IRFDecayMask=[];

while Stop_AW_AddROI
    %     Stop_AW_AddROI
    UIstr.Data_pSort = Data_pSort;
    UIstr.ROI_nr     = ROI_nr;
    UIstr.SaveResultsPath = SaveResultsPath;
    UIstr.mask_img = mask_ROI;
    UIstr.IRF_pSort = IRF_pSort;
    outData =0;
    % Create push button
    btn2 = uicontrol('Style', 'pushbutton', 'String', 'Continue Analysis',...
        'Tag','pushbutton','FontSize',15,...
        'Position', [(fh_ROI.Position(1)+250) ...
        (fh_ROI.Position(2)+fh_ROI.Position(4)-150) ...
        150 50],...
        'UserData',Stop_AW_AddROI,...
        'Callback',{@AW_AddROIContinue});
    
    btn = uicontrol('Style', 'pushbutton', 'String', 'Add ROI',...
        'Tag','pushbutton','FontSize',20,...
        'Position', [(fh_ROI.Position(1)+50) ...
        (fh_ROI.Position(2)+fh_ROI.Position(4)-150) ...
        150 50],...
        'UserData',outData,...
        'Callback',{@AW_AddROI,UIstr});
    
    fh_ROI.Visible = 'on';
    waitforbuttonpress
    pause(.2)
    h_ROI = findobj('Tag','pushbutton')
    %    pause(.2)
    
    if  h_ROI(2).UserData == false
        Stop_AW_AddROI = false
    else
        
        if ~isempty(h_ROI(1))
            if  isfield(getfield(h_ROI(1), 'UserData'), 'DataDecayMask')
                for i=1:numel(h_ROI)
                    if isempty(h_ROI(1).UserData)
                        continue;
                    else
                        DataDecayMask(:,j) = h_ROI(1).UserData.DataDecayMask;
                        IRFDecayMask (:,j) = h_ROI(1).UserData.IRFDecayMask;
                        mask_img(:,:,j+1) = h_ROI(1).UserData.mask_img;
                        limits_ROI(:,:,j) = h_ROI(1).UserData.limits_ROI
                    end
                end
            end
        end
        j=j+1;
    end
end

if isempty(DataDecayMask)
    close(fh_ROI)
else
    subplot(2,2,[1,3]); imagesc(sum(mask_img,3)); axis image; ...
        title('ROIs');  FigStyle;...
        subplot(2,2,[2]);   semilogy(DataDecayMask);
    xlim([500 4000]);
    xlabel('Time [channels]'); ylabel('Counts'); FigStyle;
    title('Decay of ROI')
    subplot(224)
    semilogy(IRFDecayMask,'.');
    title('IRF of ROI')
    xlim([500 4000]); %ylim([0 1])
    xlabel('Time [channels]'); ylabel('Counts'); FigStyle;
    save([SaveResultsPath, SaveNAMEfile,'_DecaysROI.mat'],'DataDecayMask','IRFDecayMask')
end
end