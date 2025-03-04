function [FLIM_Bild]=IntTau(Counts, FLIM, alphaLim, tauLim, time)
%% [FLIM_Bild]= IntTau(Counts, FLIM, alphaLim, tauLim, time)
lim1 = alphaLim(1);
lim2 = alphaLim(2); % max(img(:))-mean(img(:))
minTau=tauLim(1);
maxTau=tauLim(2);
img=double(Counts);
img2=double(FLIM);
% figure
h = imagesc(img2,[minTau maxTau]);
axis image
cmap= jet(512);
cmap= cmap(90:450,:,:);
colormap(cmap);
img(img < lim1) = lim1; img(img > lim2) = lim2;
img = rescaleData(img, [0 1]);
set(h,'AlphaData',img);
set(gca,'Color',[0,0,0]);
set(gca, 'XTick', [],'YTick', [])
% if time~=0;
%     title (['Time = ', num2str(time), ' s'], 'FontSize', 22)
% end
% pause(2)
cbar=colorbar; 
%  cb = colorbar('hor');
zlab = get(cbar,'xlabel');
set(zlab,'String','\tau [ns]', 'FontSize',35);
set(gca, 'FontSize', 25)
FLIM_Bild=getframe(gcf);
end