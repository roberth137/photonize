function AW_AddROI(src,event,UIstr)
% waitforbuttonpress
subplot(2,2,[1,3])   
UIstr.ROI_nr
[UIstr.mask_img(:,:,:), position]= AW_MaskROI(UIstr.Data_pSort.image);
      
        axis image
        title(['Press Button to add ROI'],'FontSize',20)
        outData.limits_ROI(1,:) = ceil(min(position))-5;
        outData.limits_ROI(2,:) = ceil(max(position))+5;
%         text(-50, -1*size(UIstr.mask_img,1)/3,...
%             'Press keyboard button to go on', 'FontSize', 20,...
%             'Units', 'pixels')
        outData.DataDecayMask = photonscore.flim.decay_from_mask(...
                                  UIstr.Data_pSort, UIstr.mask_img(:,:,UIstr.ROI_nr),...
                                  0, 4096, 4096);
                           
        outData.IRFDecayMask  = photonscore.flim.decay_from_mask(...
                                  UIstr.IRF_pSort, UIstr.mask_img(:,:,UIstr.ROI_nr),...
                                  0, 4096, 4096);          
        subplot(222)  
        semilogy(outData.DataDecayMask); 
        xlim([500 4000]);
        xlabel('Time [channels]'); ylabel('Counts'); FigStyle;       
        title('Decay of ROI')
        subplot(224)  
        semilogy(outData.IRFDecayMask,'.'); 
        title('IRF of ROI')
        xlim([500 4000]);
        ylim([0 max(outData.IRFDecayMask(:))])
        xlabel('Time [channels]'); ylabel('Counts'); FigStyle;
        
        outData.mask_img=UIstr.mask_img;
        outData.ROI_nr = UIstr.ROI_nr;
        
        xxx = dir([UIstr.SaveResultsPath, '*MaskROI*.mat']);
        
        if isempty(xxx) 
            save([UIstr.SaveResultsPath, 'MaskROI_',num2str(UIstr.ROI_nr),'.mat'],...
            'outData', 'position')
        else UIstr.ROI_nr=numel(xxx)+1;
            save([UIstr.SaveResultsPath, 'MaskROI_',num2str(UIstr.ROI_nr),'.mat'],...
            'outData', 'position')
        end
        
        src.UserData = outData;
        
end