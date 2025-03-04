function [FLIM_Bild]=IntTauLab(Counts, FLIM, alphaLim, tauLim, time, CBLab)
%% [FLIM_Bild]=IntTauLab(Counts, FLIM, alphaLim, tauLim, time, CBLab)

lim1 = alphaLim(1);
lim2 = alphaLim(2); % max(img(:))-mean(img(:))
minTau=tauLim(1);
maxTau=tauLim(2);
img=double(Counts);
img2=double(FLIM);
% figure
h = imagesc(img2,[minTau maxTau]);
axis image
colormap jet;
%
img(img < lim1) = lim1; img(img > lim2) = lim2;
img = rescaleData(img, [0 1]);
set(h,'AlphaData',img);
set(gca,'Color',[0,0,0]);
set(gca, 'XTick', [],'YTick', []);

if nargin>4 && time~=0;
    title (['Time = ', num2str(time), ' s'], 'FontSize', 22)
end

cbar=colorbar;
%  cb = colorbar('hor');
if nargin>5 && exist('CBLab', 'var')

    zlab = get(cbar,'xlabel');
%     CBLab='\tau_{mean} [ns]';
    set(zlab,'String',CBLab, 'FontSize', cbar.Label.FontSize*1.8);

end

% set(gca, 'FontSize', 30-(30*h.Parent.Position(3)));
FLIM_Bild=getframe(gcf);
end