function [wimg] = AW_IntTauTiff (data1, data2, lim1, lim2, cmap, fname, varargin)
% function AW_IntTauTiff (data1, data2, lim1, lim2, cmap, fname, varargin)

assert(nargin == 6 || nargin == 8);
rin1 = []; rin2 = [];
if nargin == 8; rin1 = varargin{1}; rin2 = varargin{2}; end

data1 = clipData(data1, [lim1(1) NaN]);
data2 = cutData(data2, [lim2(1) NaN], 0);

data1 = clipData(data1, [NaN lim1(2)]);
data2 = clipData(data2, [NaN lim2(2)]);

if ~isempty(rin1)
    rgb = cmapData(data1, cmap, rin1);
else
    rgb = cmapData(data1, cmap);
end

if ~isempty(rin2)
    a = rescaleData(data2, [0 1], rin2);
else
    a = rescaleData(data2, [0 1]);
end

rgba=uint8(round(rgb.*a*255));
wimg=rgba;
imwrite(rgba,[fname,'.tiff'], 'writemode', 'append')
% rgba(:,:,4) = a;
% rgba = uint8(round(flipud(rgba*255)));
% %# create a tiff object
% % tob = Tiff([fname,'.tif'],'w');
% tob = Tiff([fname,'.tif'],'a');
% tob.SubFileType.Page;
% %# you need to set Photometric before Compression
% tob.setTag('Photometric',Tiff.Photometric.RGB)
% % tob.setTag('Compression',Tiff.Compression.None);
% tob.setTag('Compression',Tiff.Compression.LZW)
% %# tell the program that channel 4 is alpha
% tob.setTag('ExtraSamples',Tiff.ExtraSamples.AssociatedAlpha)
% %# set additional tags (you may want to use the structure
% %# version of this for convenience)
% tob.setTag('ImageLength',size(rgba,1));
% tob.setTag('ImageWidth',size(rgba,2));
% tob.setTag('BitsPerSample',8);
% tob.setTag('RowsPerStrip',16);
% tob.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
% tob.setTag('Software','MATLAB')
% tob.setTag('SamplesPerPixel',4);
% %# write and close the file
% tob.write(rgba,'WriteMode','append')
% tob.close
end