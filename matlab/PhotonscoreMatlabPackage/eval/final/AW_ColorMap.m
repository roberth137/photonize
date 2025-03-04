function [cmap]= AW_ColorMap
dim = 1357;
uG = round(192*dim/512);
oG=round(450*dim/512);
cmap = jet(dim);
cmap2 = cmap(uG:oG,:,:);%rgb
cmap3 = flipud(cool(round(dim/4)));
clear cmap
cmap = [cmap3;cmap2(:, :, :)];
end