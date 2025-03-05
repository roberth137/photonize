%% Figure Properties
cmap= jet(512);
cmap= cmap(90:450,:,:);
colormap(cmap);set(0, 'defaultFigurePosition',[50 50 1275 850])
set(gcf, 'Position',[50 50 1275 900])
set(gca, 'FontSize', 20)
set(gca, 'TickDir', 'out', 'LineWidth', .5)
grid on;
grid minor;
% ylabel('FontSize', 20)
