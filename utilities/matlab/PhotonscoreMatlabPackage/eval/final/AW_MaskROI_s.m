function [maskGFP, position] =  AW_MaskROI_s(img)
%% [maskGFP, position] =  AW_MaskROI (img)
figure
img=double(img);
imagesc(img, [0 max(img(:))]);
axis image;
colorbar; colormap(jet);
cbar=colorbar;
        cbar.Location='Southoutside';
%             cbar.Position=[ .35 h_sb.Position(2) .3 .04];
%             cbar.AxisLocationMode;
        cbar.Label.String= 'Photons';
        cbar.Label.FontSize= 25;
%         cbar.Ticks=[0:10^(round(log10(round(max(img(:))+1)))-1):round(max(img(:))+1)];
%         cbar.TickLabel=[0:10^(round(log10(round(max(img(:))+1)))-1):round(max(img(:))+1)];
        cbar.TickDirection ='out';
%             cbar.FontSize=20;
title('Draw the area of interest, then doubleclick mouse');
h_im = imrect(gca, [size(img,1)/3 size(img,2)/3 size(img,1)/4 size(img,2)/4]);
position = wait(h_im); 
maskGFP(:,:) = logical(createMask(h_im));
title('thank you');
end
