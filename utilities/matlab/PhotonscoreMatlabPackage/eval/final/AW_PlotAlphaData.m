function [hp, hpm]=AW_PlotAlphaData(x, y, Marker, MarkerSize,...
                              MarkerFaceAlpha, MarkerEdgeAlpha)
%% [hp, hpm]=AW_PlotAlphaData(x, y, Marker, MarkerSize,...
%                               MarkerFaceAlpha, MarkerEdgeAlpha)
    hp = plot(flat(x),flat(y),'LineStyle','none','Marker', Marker);
    drawnow;
    % hp.MarkerEdgeColor = 'none';
    hpm = hp.MarkerHandle;
    hpm.FaceColorType = 'truecoloralpha';
    hpm.FaceColorData = uint8(255*[0;0;0; MarkerFaceAlpha]);
    hpm.EdgeColorData = uint8(255*[0;0;0; MarkerEdgeAlpha]);
    hpm.Size = MarkerSize;
    hold on;

end